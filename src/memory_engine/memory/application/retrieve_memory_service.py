from __future__ import annotations

from dataclasses import dataclass, field, replace

from memory_engine.memory.application.bridge import palace_to_store
from memory_engine.memory.application.activation_service import NativeActivationService
from memory_engine.memory.application.query_models import RecallQuery
from memory_engine.memory.application.result_ranker_service import (
    DefaultRecallResultRanker,
    RecallResultRanker,
)
from memory_engine.memory.application.route_planner_service import DefaultRoutePlanner
from memory_engine.memory.application.seed_selection_service import (
    EmbeddingSeedSelector,
    HybridSeedSelector,
    LexicalSeedSelector,
    default_hybrid_seed_selector,
)
from memory_engine.memory.application.space_selection_service import (
    HybridSpaceSelector,
    KeywordSpaceSelector,
    MetadataSpaceSelector,
)
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.retrieval_result import PalaceRecallResult, RecallRoute
from memory_engine.memory.domain.route_planning import RoutePlanner, RoutePlanningInput
from memory_engine.memory.domain.seed_selection import SeedSelectionInput, SeedSelector
from memory_engine.memory.domain.space_selection import SpaceSelectionInput, SpaceSelector
from memory_engine.memory_state import MemoryStatePolicy, StaticMemoryStatePolicy
from memory_engine.retrieval_factory import build_legacy_retriever
from memory_engine.schema import ActivationContext


def _default_hybrid_space_selector() -> HybridSpaceSelector:
    return HybridSpaceSelector(
        keyword_selector=KeywordSpaceSelector(),
        metadata_selector=MetadataSpaceSelector(),
    )


@dataclass(slots=True)
class RetrieveMemoryService:
    """Palace-aware recall orchestration with optional legacy graph fallback."""

    space_selector: SpaceSelector = field(default_factory=_default_hybrid_space_selector)
    seed_selector: SeedSelector = field(default_factory=default_hybrid_seed_selector)
    activation_service: NativeActivationService = field(default_factory=NativeActivationService)
    route_planner: RoutePlanner = field(default_factory=DefaultRoutePlanner)
    result_ranker: RecallResultRanker = field(default_factory=DefaultRecallResultRanker)

    def recall(self, palace: MemoryPalace, query: RecallQuery) -> PalaceRecallResult:
        state_policy = self._memory_policy_for_mode(query.policy.retriever_mode)
        space_in = SpaceSelectionInput(
            text=query.text,
            preferred_space_ids=query.preferred_space_ids,
            max_spaces=query.policy.max_spaces,
        )
        space_candidates = self.space_selector.select_spaces(palace, space_in)
        if space_candidates:
            allowed_spaces = tuple(c.space_id for c in space_candidates)
        else:
            allowed_spaces = tuple(palace.spaces.keys())

        seed_in = SeedSelectionInput(
            text=query.text,
            allowed_space_ids=allowed_spaces,
            max_seeds=query.policy.max_seeds,
            allowed_memory_kinds=query.policy.allowed_memory_kinds,
        )
        seeds = self._seed_selector_for_mode(state_policy).select_seeds(palace, seed_in)
        activation_result = self.activation_service.activate(
            palace,
            query_text=query.text,
            seeds=seeds,
            allowed_space_ids=allowed_spaces,
            retriever_mode=query.policy.retriever_mode,
            max_hops=query.policy.max_hops,
            top_k=query.policy.top_k,
        )

        plan_in = RoutePlanningInput(
            text=query.text,
            top_k=query.policy.top_k,
            max_hops=query.policy.max_hops,
        )
        planned_routes = self.route_planner.plan_routes(palace, plan_in, seeds)
        explicit_routes = tuple(
            route for route in planned_routes if route.route_source == "route_memory"
        )
        routes = self._merge_routes(
            explicit_routes=explicit_routes,
            activation_routes=activation_result.routes,
            planned_routes=planned_routes,
            top_k=query.policy.top_k,
        )

        bundle = self._result_ranker_for_mode(state_policy).rank(
            palace,
            query.text,
            seeds,
            activation_result.retrieved_memories,
            routes,
            top_k=query.policy.top_k,
        )

        orchestration_meta: dict[str, object] = {
            "selected_space_ids": list(allowed_spaces),
            "space_candidates": [
                {"space_id": c.space_id, "score": c.score, "reason": c.reason}
                for c in space_candidates
            ],
            "seed_ids": [s.memory_id for s in seeds],
            "recall_orchestration": True,
        }

        native = PalaceRecallResult(
            query=query.text,
            retrieved_memories=bundle.retrieved_memories,
            routes=bundle.routes,
            activation_snapshot=activation_result.activation_snapshot,
            metadata=orchestration_meta,
        )

        if query.policy.allow_legacy_fallback and not native.retrieved_memories:
            return self._legacy_fallback(palace, query, orchestration_meta)

        return native

    def _merge_routes(
        self,
        *,
        explicit_routes: tuple[RecallRoute, ...],
        activation_routes: tuple[RecallRoute, ...],
        planned_routes: tuple[RecallRoute, ...],
        top_k: int,
    ) -> tuple[RecallRoute, ...]:
        candidates = [
            *explicit_routes,
            *activation_routes,
            *(
                route
                for route in planned_routes
                if route.route_source != "route_memory"
            ),
        ]
        if not candidates:
            return ()

        by_shape: dict[tuple[tuple[str, ...], str], RecallRoute] = {}
        for route in candidates:
            key = (route.step_memory_ids, route.route_kind)
            existing = by_shape.get(key)
            if existing is None:
                by_shape[key] = route
                continue
            if self._route_priority(route) > self._route_priority(existing):
                by_shape[key] = route
            elif (
                self._route_priority(route) == self._route_priority(existing)
                and route.score > existing.score
            ):
                by_shape[key] = route

        ordered = sorted(
            by_shape.values(),
            key=lambda route: (
                self._route_priority(route),
                route.score,
                len(route.step_memory_ids),
            ),
            reverse=True,
        )
        return tuple(ordered[: max(1, top_k)])

    def _route_priority(self, route: RecallRoute) -> int:
        if route.route_source == "route_memory":
            return 3
        if route.route_source == "native_activation":
            return 2
        if route.route_source == "legacy_path":
            return 1
        return 0

    def _memory_policy_for_mode(self, retriever_mode: str) -> MemoryStatePolicy:
        if retriever_mode in {
            "lexical_baseline",
            "embedding_baseline",
            "structure_only",
            "weighted_graph_static",
            "activation_spreading_static",
        }:
            return StaticMemoryStatePolicy()
        return MemoryStatePolicy()

    def _seed_selector_for_mode(self, state_policy: MemoryStatePolicy) -> SeedSelector:
        if isinstance(self.seed_selector, HybridSeedSelector):
            return replace(
                self.seed_selector,
                embedding_selector=replace(
                    self.seed_selector.embedding_selector,
                    state_policy=state_policy,
                ),
                lexical_selector=replace(
                    self.seed_selector.lexical_selector,
                    state_policy=state_policy,
                ),
            )
        if isinstance(self.seed_selector, (EmbeddingSeedSelector, LexicalSeedSelector)):
            return replace(self.seed_selector, state_policy=state_policy)
        return self.seed_selector

    def _result_ranker_for_mode(self, state_policy: MemoryStatePolicy) -> RecallResultRanker:
        if isinstance(self.result_ranker, DefaultRecallResultRanker):
            return replace(self.result_ranker, state_policy=state_policy)
        return self.result_ranker

    def _legacy_fallback(
        self,
        palace: MemoryPalace,
        query: RecallQuery,
        orchestration_meta: dict[str, object],
    ) -> PalaceRecallResult:
        store = palace_to_store(palace)
        retriever = build_legacy_retriever(query.policy.retriever_mode, store)
        try:
            result = retriever.search(
                query.text,
                top_k=query.policy.top_k,
                context=ActivationContext(
                    query=query.text,
                    max_hops=query.policy.max_hops,
                ),
            )
        except TypeError:
            result = retriever.search(
                query.text,
                top_k=query.policy.top_k,
            )
        legacy = PalaceRecallResult.from_legacy_result(result)
        merged_meta = {
            **legacy.metadata,
            "fallback_reason": "empty_native_recall",
            **orchestration_meta,
        }
        return replace(legacy, metadata=merged_meta)
