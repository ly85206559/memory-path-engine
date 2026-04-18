from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from memory_engine.activation import (
    ActivatedNode,
    ActivationSignal,
    DefaultPropagationPolicy,
)
from memory_engine.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity,
    lexical_overlap,
)
from memory_engine.memory.application.bridge import palace_to_store
from memory_engine.memory.domain.retrieval_result import (
    ActivationSnapshot,
    ActivationSnapshotEntry,
    RecallRoute,
    RetrievedMemory,
)
from memory_engine.memory.domain.seed_selection import SeedActivation
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory_state import MemoryStatePolicy, StaticMemoryStatePolicy
from memory_engine.schema import ActivationContext, ActivationTraceStep, MemoryEdge, MemoryNode
from memory_engine.scoring import StructureOnlyScoringStrategy, WeightedSumScoringStrategy
from memory_engine.semantics import contradiction_candidates


@dataclass(frozen=True, slots=True)
class ActivationRuntimeResult:
    retrieved_memories: tuple[RetrievedMemory, ...] = ()
    routes: tuple[RecallRoute, ...] = ()
    activation_snapshot: ActivationSnapshot = ActivationSnapshot()


class NativeActivationService:
    """Palace-native activation and expansion runtime used by recall orchestration."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        max_activated_nodes: int = 12,
    ) -> None:
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self.max_activated_nodes = max_activated_nodes

    def activate(
        self,
        palace: MemoryPalace,
        *,
        query_text: str,
        seeds: tuple[SeedActivation, ...],
        allowed_space_ids: tuple[str, ...],
        retriever_mode: str,
        max_hops: int,
        top_k: int,
    ) -> ActivationRuntimeResult:
        if not seeds:
            return ActivationRuntimeResult()

        store = palace_to_store(palace)
        memory_state_policy = self._memory_policy_for_mode(retriever_mode)
        scoring_strategy = (
            StructureOnlyScoringStrategy()
            if retriever_mode == "structure_only"
            else WeightedSumScoringStrategy(memory_state_policy=memory_state_policy)
        )
        propagation_policy = self._propagation_policy_for_mode(retriever_mode)
        contradiction_pairs = contradiction_candidates(store.nodes(), store.edges())
        context = ActivationContext(query=query_text, max_hops=max_hops)

        best_by_id: dict[str, RetrievedMemory] = {}
        activation_entries: list[ActivationSnapshotEntry] = []
        routes: list[RecallRoute] = []
        allowed_spaces = set(allowed_space_ids)

        for seed in seeds:
            explored = self._activate_from_seed(
                store=store,
                query_text=query_text,
                seed=seed,
                context=context,
                allowed_spaces=allowed_spaces,
                scoring_strategy=scoring_strategy,
                memory_state_policy=memory_state_policy,
                propagation_policy=propagation_policy,
                contradiction_pairs=contradiction_pairs,
            )
            if explored is None:
                continue
            routes.append(explored.route)
            activation_entries.extend(explored.activation_trace)
            for memory in explored.retrieved_memories:
                current = best_by_id.get(memory.memory_id)
                if current is None or memory.score > current.score:
                    best_by_id[memory.memory_id] = memory

        retrieved = tuple(
            sorted(best_by_id.values(), key=lambda item: item.score, reverse=True)[: max(1, top_k)]
        )
        ordered_routes = tuple(sorted(routes, key=lambda route: route.score, reverse=True)[: max(1, top_k)])
        return ActivationRuntimeResult(
            retrieved_memories=retrieved,
            routes=ordered_routes,
            activation_snapshot=ActivationSnapshot(tuple(activation_entries)),
        )

    def _activate_from_seed(
        self,
        *,
        store,
        query_text: str,
        seed: SeedActivation,
        context: ActivationContext,
        allowed_spaces: set[str],
        scoring_strategy,
        memory_state_policy,
        propagation_policy,
        contradiction_pairs,
    ):
        seed_activation = propagation_policy.seed_activation(seed_score=seed.score)
        if seed_activation <= 0.0:
            return None

        activation_trace: list[ActivationTraceStep] = [
            ActivationTraceStep(
                node_id=seed.memory_id,
                hop=0,
                incoming_activation=seed_activation,
                propagated_activation=seed_activation,
                is_seed=True,
            )
        ]
        queue = deque(
            [
                ActivationSignal(
                    node_id=seed.memory_id,
                    activation=seed_activation,
                    hop=0,
                )
            ]
        )
        visited: set[str] = set()
        activated_nodes: dict[str, ActivatedNode] = {}

        while queue and len(activated_nodes) < self.max_activated_nodes:
            signal = queue.popleft()
            node = store.get_node(signal.node_id)
            if not self._node_is_allowed(node, allowed_spaces):
                continue
            existing = activated_nodes.get(signal.node_id)
            if existing is not None and existing.activation >= signal.activation:
                continue

            semantic_score = (
                seed.score
                if signal.hop == 0
                else self._semantic_similarity(query_text, node.content)
            )
            breakdown = scoring_strategy.score_node(
                query=query_text,
                node=node,
                semantic_score=semantic_score,
                context=context,
                depth=signal.hop,
                source_node_id=signal.source_node_id,
            )
            activated_score = min(1.0, max(signal.activation, breakdown.total_score))
            activated_nodes[signal.node_id] = ActivatedNode(
                node_id=signal.node_id,
                activation=signal.activation,
                score=activated_score,
                hop=signal.hop,
                source_node_id=signal.source_node_id,
                via_edge_type=signal.via_edge_type,
            )
            self._update_trace_score(
                activation_trace,
                node_id=signal.node_id,
                source_node_id=signal.source_node_id,
                hop=signal.hop,
                activated_score=activated_score,
            )

            if signal.hop >= context.max_hops:
                continue

            visited.add(signal.node_id)
            for edge in store.neighbors(signal.node_id):
                if edge.to_id in visited:
                    continue
                destination_node = store.get_node(edge.to_id)
                if not self._node_is_allowed(destination_node, allowed_spaces):
                    activation_trace.append(
                        ActivationTraceStep(
                            node_id=edge.to_id,
                            source_node_id=signal.node_id,
                            edge_type=edge.edge_type,
                            hop=signal.hop + 1,
                            incoming_activation=signal.activation,
                            propagated_activation=0.0,
                            stopped_reason="outside_space_scope",
                        )
                    )
                    continue

                propagation = propagation_policy.propagate(signal=signal, edge=edge)
                if propagation.stopped_reason is not None:
                    activation_trace.append(
                        ActivationTraceStep(
                            node_id=edge.to_id,
                            source_node_id=signal.node_id,
                            edge_type=edge.edge_type,
                            hop=propagation.hop,
                            incoming_activation=propagation.incoming_activation,
                            propagated_activation=propagation.propagated_activation,
                            stopped_reason=propagation.stopped_reason,
                        )
                    )
                    continue

                propagated_activation = (
                    propagation.propagated_activation
                    * memory_state_policy.propagation_factor(destination_node)
                )
                if propagated_activation < propagation_policy.activation_threshold:
                    activation_trace.append(
                        ActivationTraceStep(
                            node_id=edge.to_id,
                            source_node_id=signal.node_id,
                            edge_type=edge.edge_type,
                            hop=propagation.hop,
                            incoming_activation=propagation.incoming_activation,
                            propagated_activation=propagated_activation,
                            stopped_reason="below_threshold",
                        )
                    )
                    continue

                propagated_activation = propagation_policy.adjust_propagated_activation(
                    propagated_activation=propagated_activation,
                    edge=edge,
                    destination_node=destination_node,
                    source_node_id=signal.node_id,
                    contradiction_candidates=contradiction_pairs,
                )
                activation_trace.append(
                    ActivationTraceStep(
                        node_id=edge.to_id,
                        source_node_id=signal.node_id,
                        edge_type=edge.edge_type,
                        hop=propagation.hop,
                        incoming_activation=propagation.incoming_activation,
                        propagated_activation=propagated_activation,
                    )
                )
                queue.append(
                    ActivationSignal(
                        node_id=edge.to_id,
                        activation=propagated_activation,
                        hop=propagation.hop,
                        source_node_id=signal.node_id,
                        via_edge_type=edge.edge_type,
                    )
                )

        if not activated_nodes:
            return None

        terminal_candidates = [
            activated
            for activated in activated_nodes.values()
            if activated.node_id != seed.memory_id
        ] or list(activated_nodes.values())
        terminal = max(
            terminal_candidates,
            key=lambda activated: (activated.score, activated.activation, -activated.hop),
        )
        ordered_path = self._reconstruct_path(
            seed_id=seed.memory_id,
            terminal_id=terminal.node_id,
            activated_nodes=activated_nodes,
        )
        edge_types = tuple(
            activated.via_edge_type
            for activated in ordered_path
            if activated.via_edge_type is not None
        )
        route_kind = self._route_kind_from_edges(edge_types)
        explanation = self._route_explanation(
            seed_id=seed.memory_id,
            terminal_id=terminal.node_id,
            route_kind=route_kind,
            edge_types=edge_types,
            hop_count=max(0, len(ordered_path) - 1),
        )
        route = RecallRoute(
            route_id=f"native-path-{seed.memory_id}",
            route_kind=route_kind,
            step_memory_ids=tuple(item.node_id for item in ordered_path),
            score=terminal.score,
            route_source="native_activation",
            explanation=explanation,
        )

        retrieved_memories = tuple(
            RetrievedMemory(
                memory_id=activated.node_id,
                score=activated.score,
                reason=(
                    f"seed activation={activated.activation:.3f}"
                    if activated.hop == 0
                    else f"native activation hop={activated.hop} activation={activated.activation:.3f}"
                ),
                retrieval_role="seed" if activated.hop == 0 else "support",
                memory_kind=store.get_node(activated.node_id).attributes.get("memory_kind"),
                lifecycle_state=store.get_node(activated.node_id).attributes.get("lifecycle_state"),
                consolidation_kind=store.get_node(activated.node_id).attributes.get("consolidation_kind"),
            )
            for activated in sorted(
                activated_nodes.values(),
                key=lambda item: (item.score, item.activation),
                reverse=True,
            )
        )
        snapshot = tuple(
            ActivationSnapshotEntry(
                memory_id=step.node_id,
                source_memory_id=step.source_node_id,
                edge_type=step.edge_type,
                hop=step.hop,
                incoming_activation=step.incoming_activation,
                propagated_activation=step.propagated_activation,
                activated_score=step.activated_score,
                stopped_reason=step.stopped_reason,
                is_seed=step.is_seed,
            )
            for step in activation_trace
        )

        return _PerSeedActivationResult(
            retrieved_memories=retrieved_memories,
            route=route,
            activation_trace=snapshot,
        )

    def _semantic_similarity(self, query_text: str, content: str) -> float:
        return max(
            cosine_similarity(
                self.embedding_provider.embed(query_text),
                self.embedding_provider.embed(content),
            ),
            lexical_overlap(query_text, content),
        )

    def _node_is_allowed(self, node: MemoryNode, allowed_spaces: set[str]) -> bool:
        if node.attributes.get("memory_kind") == "route":
            return False
        if not allowed_spaces:
            return True
        return node.attributes.get("space_id") in allowed_spaces

    def _reconstruct_path(
        self,
        *,
        seed_id: str,
        terminal_id: str,
        activated_nodes: dict[str, ActivatedNode],
    ) -> list[ActivatedNode]:
        ordered_path: list[ActivatedNode] = []
        current_id = terminal_id
        while True:
            activated = activated_nodes[current_id]
            ordered_path.append(activated)
            if current_id == seed_id or activated.source_node_id is None:
                break
            current_id = activated.source_node_id
        ordered_path.reverse()
        return ordered_path

    def _update_trace_score(
        self,
        activation_trace: list[ActivationTraceStep],
        *,
        node_id: str,
        source_node_id: str | None,
        hop: int,
        activated_score: float,
    ) -> None:
        for step in reversed(activation_trace):
            if (
                step.node_id == node_id
                and step.source_node_id == source_node_id
                and step.hop == hop
                and step.activated_score is None
            ):
                step.activated_score = activated_score
                return

    def _memory_policy_for_mode(self, retriever_mode: str):
        if retriever_mode in {
            "lexical_baseline",
            "embedding_baseline",
            "structure_only",
            "weighted_graph_static",
            "activation_spreading_static",
        }:
            return StaticMemoryStatePolicy()
        return MemoryStatePolicy()

    def _propagation_policy_for_mode(self, retriever_mode: str) -> DefaultPropagationPolicy:
        if retriever_mode == "structure_only":
            return DefaultPropagationPolicy(activation_decay=0.82, activation_threshold=0.05)
        if retriever_mode.startswith("activation_spreading"):
            return DefaultPropagationPolicy(activation_decay=0.75, activation_threshold=0.15)
        if retriever_mode == "embedding_baseline":
            return DefaultPropagationPolicy(activation_decay=0.7, activation_threshold=0.2)
        return DefaultPropagationPolicy(activation_decay=0.8, activation_threshold=0.1)

    def _route_kind_from_edges(self, edge_types: tuple[str, ...]) -> str:
        edge_set = set(edge_types)
        if {"exception_to", "contradicts"} & edge_set:
            return "exception"
        if {"depends_on", "part_of", "causes"} & edge_set:
            return "dependency"
        if {"summarizes", "recalls"} & edge_set:
            return "semantic-summary"
        if "next" in edge_set:
            return "timeline"
        return "graph_expansion"

    def _route_explanation(
        self,
        *,
        seed_id: str,
        terminal_id: str,
        route_kind: str,
        edge_types: tuple[str, ...],
        hop_count: int,
    ) -> str:
        if edge_types:
            edge_summary = " -> ".join(edge_types)
            return (
                f"expanded from seed {seed_id} to {terminal_id} across {hop_count} hops"
                f" using {edge_summary} edges ({route_kind})"
            )
        if seed_id == terminal_id:
            return f"seed {seed_id} remained the strongest recalled node"
        return f"expanded from seed {seed_id} to {terminal_id} ({route_kind})"


@dataclass(frozen=True, slots=True)
class _PerSeedActivationResult:
    retrieved_memories: tuple[RetrievedMemory, ...]
    route: RecallRoute
    activation_trace: tuple[ActivationSnapshotEntry, ...]
