from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.ranking import RankedRecallBundle
from memory_engine.memory.domain.retrieval_result import RecallRoute, RetrievedMemory
from memory_engine.memory.domain.seed_selection import SeedActivation
from memory_engine.memory_state import MemoryStatePolicy


class RecallResultRanker(Protocol):
    def rank(
        self,
        palace: MemoryPalace,
        query_text: str,
        seeds: tuple[SeedActivation, ...],
        activated_memories: tuple[RetrievedMemory, ...],
        routes: tuple[RecallRoute, ...],
        *,
        top_k: int,
    ) -> RankedRecallBundle:
        ...


@dataclass(slots=True)
class DefaultRecallResultRanker:
    """Fuse seeds and route steps into a deduplicated ranked memory list."""

    state_policy: MemoryStatePolicy = field(default_factory=MemoryStatePolicy)

    def rank(
        self,
        palace: MemoryPalace,
        query_text: str,
        seeds: tuple[SeedActivation, ...],
        activated_memories: tuple[RetrievedMemory, ...],
        routes: tuple[RecallRoute, ...],
        *,
        top_k: int,
    ) -> RankedRecallBundle:
        del query_text
        by_id: dict[str, RetrievedMemory] = {}

        for seed in seeds:
            mem = palace.memories.get(seed.memory_id)
            kind = mem.kind.value if mem is not None else None
            lifecycle_state = mem.state.state.value if mem is not None else None
            consolidation_kind = (
                str(mem.metadata.get("consolidation_kind"))
                if mem is not None and mem.metadata.get("consolidation_kind") is not None
                else None
            )
            score = (
                min(1.0, seed.score * self.state_policy.recall_multiplier_for_state(mem.state))
                if mem is not None
                else seed.score
            )
            prev = by_id.get(seed.memory_id)
            candidate = RetrievedMemory(
                memory_id=seed.memory_id,
                score=score,
                reason=seed.reason,
                retrieval_role="seed",
                memory_kind=kind,
                lifecycle_state=lifecycle_state,
                consolidation_kind=consolidation_kind,
            )
            if prev is None or candidate.score > prev.score:
                by_id[seed.memory_id] = candidate

        for activated in activated_memories:
            prev = by_id.get(activated.memory_id)
            if prev is None or activated.score > prev.score:
                by_id[activated.memory_id] = activated

        for route in routes:
            for idx, mid in enumerate(route.step_memory_ids):
                mem = palace.memories.get(mid)
                kind = mem.kind.value if mem is not None else None
                lifecycle_state = mem.state.state.value if mem is not None else None
                consolidation_kind = (
                    str(mem.metadata.get("consolidation_kind"))
                    if mem is not None and mem.metadata.get("consolidation_kind") is not None
                    else None
                )
                step_score = route.score * (1.0 / (1 + idx))
                if mem is not None:
                    step_score = min(
                        1.0,
                        step_score * self.state_policy.recall_multiplier_for_state(mem.state),
                    )
                candidate = RetrievedMemory(
                    memory_id=mid,
                    score=step_score,
                    reason=f"route:{route.route_id}",
                    retrieval_role="support" if idx > 0 else "seed",
                    memory_kind=kind,
                    lifecycle_state=lifecycle_state,
                    consolidation_kind=consolidation_kind,
                )
                prev = by_id.get(mid)
                if prev is None or candidate.score > prev.score:
                    by_id[mid] = candidate

        ranked = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
        retrieved = tuple(ranked[: max(1, top_k)])
        ordered_routes = tuple(sorted(routes, key=lambda r: r.score, reverse=True))
        return RankedRecallBundle(retrieved_memories=retrieved, routes=ordered_routes)
