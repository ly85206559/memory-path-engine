from __future__ import annotations

from dataclasses import dataclass, field, replace

from memory_engine.memory.application.bridge import palace_to_store
from memory_engine.memory.application.query_models import RecallQuery
from memory_engine.memory.application.result_ranker_service import (
    DefaultRecallResultRanker,
    RecallResultRanker,
)
from memory_engine.memory.application.route_planner_service import DefaultRoutePlanner
from memory_engine.memory.application.seed_selection_service import default_hybrid_seed_selector
from memory_engine.memory.application.space_selection_service import (
    HybridSpaceSelector,
    KeywordSpaceSelector,
    MetadataSpaceSelector,
)
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.retrieval_result import ActivationSnapshot, PalaceRecallResult
from memory_engine.memory.domain.route_planning import RoutePlanner, RoutePlanningInput
from memory_engine.memory.domain.seed_selection import SeedSelectionInput, SeedSelector
from memory_engine.memory.domain.space_selection import SpaceSelectionInput, SpaceSelector
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
    route_planner: RoutePlanner = field(default_factory=DefaultRoutePlanner)
    result_ranker: RecallResultRanker = field(default_factory=DefaultRecallResultRanker)

    def recall(self, palace: MemoryPalace, query: RecallQuery) -> PalaceRecallResult:
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
        seeds = self.seed_selector.select_seeds(palace, seed_in)

        plan_in = RoutePlanningInput(
            text=query.text,
            top_k=query.policy.top_k,
            max_hops=query.policy.max_hops,
        )
        routes = self.route_planner.plan_routes(palace, plan_in, seeds)

        bundle = self.result_ranker.rank(
            palace,
            query.text,
            seeds,
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
            activation_snapshot=ActivationSnapshot(),
            metadata=orchestration_meta,
        )

        if query.policy.allow_legacy_fallback and not native.retrieved_memories:
            return self._legacy_fallback(palace, query, orchestration_meta)

        return native

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
