from __future__ import annotations

from collections import deque
import time

from memory_engine.activation import (
    ActivatedNode,
    ActivationSignal,
    DefaultPropagationPolicy,
    PropagationPolicy,
)
from memory_engine.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity,
    lexical_overlap,
)
from memory_engine.memory_state import MemoryStatePolicy, decay_unvisited_nodes, reinforce_result_paths
from memory_engine.replay import path_answer
from memory_engine.schema import ActivationContext, ActivationTraceStep, RetrievalResult
from memory_engine.scoring import ScoringStrategy, StructureOnlyScoringStrategy, WeightedSumScoringStrategy
from memory_engine.semantics import (
    contradiction_bonus,
    contradiction_candidates,
    semantic_activation_bonus,
)
from memory_engine.store import MemoryStore


class BaselineTopKRetriever:
    def __init__(self, store: MemoryStore, memory_state_policy: MemoryStatePolicy | None = None) -> None:
        self.store = store
        self.memory_state_policy = memory_state_policy or MemoryStatePolicy()

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        ranked = sorted(
            self.store.nodes(),
            key=lambda node: lexical_overlap(query, node.content),
            reverse=True,
        )[:top_k]
        paths = [
            path_answer(query, [(node, lexical_overlap(query, node.content), "baseline lexical hit", None)])
            for node in ranked
            if lexical_overlap(query, node.content) > 0
        ]
        reinforce_result_paths(self.store, paths=paths, policy=self.memory_state_policy)
        decay_unvisited_nodes(
            self.store,
            visited_node_ids={step.node_id for path in paths for step in path.steps},
            policy=self.memory_state_policy,
        )
        return RetrievalResult(query=query, paths=paths)


class EmbeddingTopKRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
        memory_state_policy: MemoryStatePolicy | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self._embedding_cache: dict[str, list[float]] = {}
        self.memory_state_policy = memory_state_policy or MemoryStatePolicy()

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        ranked = self.rank_candidates(query, top_k=top_k)
        paths = [
            path_answer(query, [(node, score, "embedding semantic hit", None)])
            for node, score in ranked
            if score > 0
        ]
        reinforce_result_paths(self.store, paths=paths, policy=self.memory_state_policy)
        decay_unvisited_nodes(
            self.store,
            visited_node_ids={step.node_id for path in paths for step in path.steps},
            policy=self.memory_state_policy,
        )
        return RetrievalResult(query=query, paths=paths)

    def rank_candidates(self, query: str, top_k: int = 3) -> list[tuple]:
        query_embedding = self._embed(query)
        ranked = sorted(
            (
                (node, cosine_similarity(query_embedding, self._embed(node.content)))
                for node in self.store.nodes()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]

    def semantic_similarity(self, query: str, text: str) -> float:
        return cosine_similarity(self._embed(query), self._embed(text))

    def _embed(self, text: str) -> list[float]:
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.embedding_provider.embed(text)
        return self._embedding_cache[text]


class WeightedGraphRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
        scoring_strategy: ScoringStrategy | None = None,
        memory_state_policy: MemoryStatePolicy | None = None,
    ) -> None:
        self.store = store
        self.embedding_retriever = EmbeddingTopKRetriever(
            store=store,
            embedding_provider=embedding_provider,
            memory_state_policy=memory_state_policy,
        )
        self.scoring_strategy = scoring_strategy or WeightedSumScoringStrategy()
        self.memory_state_policy = memory_state_policy or MemoryStatePolicy()
        self.contradiction_candidates = contradiction_candidates(
            self.store.nodes(),
            self.store.edges(),
        )
        self._annotate_contradiction_targets()

    def _annotate_contradiction_targets(self) -> None:
        by_node: dict[str, set[str]] = {}
        for candidate in self.contradiction_candidates:
            by_node.setdefault(candidate.left_node_id, set()).add(candidate.right_node_id)
            by_node.setdefault(candidate.right_node_id, set()).add(candidate.left_node_id)
        for node_id, targets in by_node.items():
            self.store.get_node(node_id).attributes["contradiction_targets"] = sorted(targets)

    def search(self, query: str, top_k: int = 3, context: ActivationContext | None = None) -> RetrievalResult:
        context = context or ActivationContext(query=query)
        start = time.perf_counter()
        seeds = self._rank_seed_candidates(query, top_k=top_k)

        paths = []
        for seed, seed_similarity in seeds:
            path = self._expand_from_seed(
                query=query,
                seed_id=seed.id,
                seed_similarity=seed_similarity,
                context=context,
            )
            if path is not None:
                paths.append(path)

        elapsed_ms = (time.perf_counter() - start) * 1000
        for path in paths:
            path.final_answer = f"{path.final_answer} [latency_ms={elapsed_ms:.2f}]"
        reinforce_result_paths(self.store, paths=paths, policy=self.memory_state_policy)
        decay_unvisited_nodes(
            self.store,
            visited_node_ids={step.node_id for path in paths for step in path.steps},
            policy=self.memory_state_policy,
        )
        return RetrievalResult(query=query, paths=paths)

    def _rank_seed_candidates(self, query: str, top_k: int) -> list[tuple]:
        combined: dict[str, tuple] = {}

        for node, score in self.embedding_retriever.rank_candidates(query, top_k=top_k):
            combined[node.id] = (node, score)

        for node in self._lexical_rank_candidates(query, top_k=top_k):
            score = self._semantic_similarity(query, node.content)
            existing = combined.get(node.id)
            if existing is None or score > existing[1]:
                combined[node.id] = (node, score)

        ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
        return ranked[: max(top_k, min(len(ranked), top_k * 2))]

    def _lexical_rank_candidates(self, query: str, top_k: int) -> list:
        return sorted(
            self.store.nodes(),
            key=lambda node: lexical_overlap(query, node.content),
            reverse=True,
        )[:top_k]

    def _semantic_similarity(self, query: str, text: str) -> float:
        embedding_score = self.embedding_retriever.semantic_similarity(query, text)
        lexical_score = lexical_overlap(query, text)
        return max(embedding_score, lexical_score)

    def _expand_from_seed(
        self,
        query: str,
        seed_id: str,
        seed_similarity: float,
        context: ActivationContext,
    ):
        visited = {seed_id}
        queue = deque([(seed_id, 0, None, None)])
        chain: list[tuple] = []

        while queue:
            node_id, depth, via_edge, source_node_id = queue.popleft()
            node = self.store.get_node(node_id)
            semantic_score = (
                seed_similarity
                if depth == 0
                else self._semantic_similarity(query, node.content)
            )
            breakdown = self._score_node(
                query=query,
                node=node,
                semantic_score=semantic_score,
                context=context,
                depth=depth,
                source_node_id=source_node_id,
            )
            reason = (
                f"seed hit semantic={breakdown.semantic_score:.3f}"
                if depth == 0
                else (
                    f"expanded at hop {depth} total={breakdown.total_score:.3f} "
                    f"exception={breakdown.exception_score:.3f} "
                    f"contradiction={breakdown.contradiction_score:.3f}"
                )
            )
            chain.append((node, breakdown.total_score, reason, via_edge))

            if depth >= context.max_hops:
                continue

            for edge in self.store.neighbors(node_id):
                if edge.to_id in visited:
                    continue
                visited.add(edge.to_id)
                queue.append((edge.to_id, depth + 1, edge.edge_type, node_id))

        chain.sort(key=lambda item: item[1], reverse=True)
        if not chain:
            return None
        return path_answer(query, chain[:3])

    def _score_node(
        self,
        *,
        query: str,
        node,
        semantic_score: float,
        context: ActivationContext,
        depth: int,
        source_node_id: str | None,
    ):
        try:
            return self.scoring_strategy.score_node(
                query=query,
                node=node,
                semantic_score=semantic_score,
                context=context,
                depth=depth,
                source_node_id=source_node_id,
            )
        except TypeError:
            return self.scoring_strategy.score_node(
                query=query,
                node=node,
                semantic_score=semantic_score,
                context=context,
                depth=depth,
            )


class StructureAwareRetriever(WeightedGraphRetriever):
    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
        memory_state_policy: MemoryStatePolicy | None = None,
    ) -> None:
        super().__init__(
            store=store,
            embedding_provider=embedding_provider,
            scoring_strategy=StructureOnlyScoringStrategy(),
            memory_state_policy=memory_state_policy,
        )


class ActivationSpreadingRetriever(WeightedGraphRetriever):
    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
        scoring_strategy: ScoringStrategy | None = None,
        propagation_policy: PropagationPolicy | None = None,
        max_activated_nodes: int = 12,
        memory_state_policy: MemoryStatePolicy | None = None,
    ) -> None:
        super().__init__(
            store=store,
            embedding_provider=embedding_provider,
            scoring_strategy=scoring_strategy,
            memory_state_policy=memory_state_policy,
        )
        self.propagation_policy = propagation_policy or DefaultPropagationPolicy()
        self.max_activated_nodes = max_activated_nodes

    def search(self, query: str, top_k: int = 3, context: ActivationContext | None = None) -> RetrievalResult:
        context = context or ActivationContext(query=query)
        start = time.perf_counter()
        seeds = self._rank_seed_candidates(query, top_k=top_k)

        paths = []
        for seed, seed_similarity in seeds:
            path = self._activate_from_seed(
                query=query,
                seed_id=seed.id,
                seed_similarity=seed_similarity,
                context=context,
            )
            if path is not None:
                paths.append(path)

        elapsed_ms = (time.perf_counter() - start) * 1000
        for path in paths:
            path.final_answer = f"{path.final_answer} [latency_ms={elapsed_ms:.2f}]"
        reinforce_result_paths(self.store, paths=paths, policy=self.memory_state_policy)
        decay_unvisited_nodes(
            self.store,
            visited_node_ids={step.node_id for path in paths for step in path.steps},
            policy=self.memory_state_policy,
        )
        return RetrievalResult(query=query, paths=paths)

    def _activate_from_seed(
        self,
        *,
        query: str,
        seed_id: str,
        seed_similarity: float,
        context: ActivationContext,
    ):
        seed_activation = self.propagation_policy.seed_activation(seed_score=seed_similarity)
        if seed_activation <= 0.0:
            return None

        activation_trace = [
            ActivationTraceStep(
                node_id=seed_id,
                hop=0,
                incoming_activation=seed_activation,
                propagated_activation=seed_activation,
                is_seed=True,
            )
        ]
        queue = deque(
            [
                ActivationSignal(
                    node_id=seed_id,
                    activation=seed_activation,
                    hop=0,
                )
            ]
        )
        visited: set[str] = set()
        activated_nodes: dict[str, ActivatedNode] = {}

        while queue and len(activated_nodes) < self.max_activated_nodes:
            signal = queue.popleft()
            existing = activated_nodes.get(signal.node_id)
            if existing is not None and existing.activation >= signal.activation:
                continue

            node = self.store.get_node(signal.node_id)
            semantic_score = (
                seed_similarity
                if signal.hop == 0
                else self._semantic_similarity(query, node.content)
            )
            breakdown = self._score_node(
                query=query,
                node=node,
                semantic_score=semantic_score,
                context=context,
                depth=signal.hop,
                source_node_id=signal.source_node_id,
            )
            activated_score = max(signal.activation, breakdown.total_score)
            activated_nodes[signal.node_id] = ActivatedNode(
                node_id=signal.node_id,
                activation=signal.activation,
                score=min(activated_score, 1.0),
                hop=signal.hop,
                source_node_id=signal.source_node_id,
                via_edge_type=signal.via_edge_type,
            )
            self._update_trace_score(
                activation_trace,
                node_id=signal.node_id,
                source_node_id=signal.source_node_id,
                hop=signal.hop,
                activated_score=min(activated_score, 1.0),
            )

            if signal.hop >= context.max_hops:
                continue

            visited.add(signal.node_id)
            for edge in self.store.neighbors(signal.node_id):
                if edge.to_id in visited:
                    continue
                propagation = self.propagation_policy.propagate(signal=signal, edge=edge)
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
                destination_node = self.store.get_node(edge.to_id)
                propagated_activation = propagation.propagated_activation
                if edge.edge_type == "exception_to":
                    propagated_activation += 0.12
                propagated_activation += semantic_activation_bonus(destination_node)
                propagated_activation += contradiction_bonus(
                    node_id=edge.to_id,
                    candidates=self.contradiction_candidates,
                    source_node_id=signal.node_id,
                )
                activation_trace.append(
                    ActivationTraceStep(
                        node_id=edge.to_id,
                        source_node_id=signal.node_id,
                        edge_type=edge.edge_type,
                        hop=propagation.hop,
                        incoming_activation=propagation.incoming_activation,
                        propagated_activation=min(propagated_activation, 1.0),
                    )
                )
                queue.append(
                    ActivationSignal(
                        node_id=edge.to_id,
                        activation=min(propagated_activation, 1.0),
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
            if activated.node_id != seed_id
        ] or list(activated_nodes.values())
        terminal = max(
            terminal_candidates,
            key=lambda activated: (activated.score, activated.activation, activated.hop),
        )
        ordered_path = self._reconstruct_activated_path(
            seed_id=seed_id,
            terminal_id=terminal.node_id,
            activated_nodes=activated_nodes,
        )
        chain = []
        for activated in ordered_path:
            node = self.store.get_node(activated.node_id)
            reason = (
                f"seed activation={activated.activation:.3f}"
                if activated.hop == 0
                else f"propagated hop={activated.hop} activation={activated.activation:.3f}"
            )
            chain.append((node, activated.score, reason, activated.via_edge_type))
        return path_answer(query, chain, activation_trace=activation_trace)

    def _reconstruct_activated_path(
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
