from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.ranking import RankedRecallBundle
from memory_engine.memory.domain.retrieval_result import RecallRoute, RetrievedMemory
from memory_engine.memory.domain.seed_selection import SeedActivation


class RecallResultRanker(Protocol):
    def rank(
        self,
        palace: MemoryPalace,
        query_text: str,
        seeds: tuple[SeedActivation, ...],
        routes: tuple[RecallRoute, ...],
        *,
        top_k: int,
    ) -> RankedRecallBundle:
        ...


@dataclass(slots=True)
class DefaultRecallResultRanker:
    """Fuse seeds and route steps into a deduplicated ranked memory list."""

    def rank(
        self,
        palace: MemoryPalace,
        query_text: str,
        seeds: tuple[SeedActivation, ...],
        routes: tuple[RecallRoute, ...],
        *,
        top_k: int,
    ) -> RankedRecallBundle:
        del query_text
        by_id: dict[str, RetrievedMemory] = {}

        for seed in seeds:
            mem = palace.memories.get(seed.memory_id)
            kind = mem.kind.value if mem is not None else None
            prev = by_id.get(seed.memory_id)
            candidate = RetrievedMemory(
                memory_id=seed.memory_id,
                score=seed.score,
                reason=seed.reason,
                retrieval_role="seed",
                memory_kind=kind,
            )
            if prev is None or candidate.score > prev.score:
                by_id[seed.memory_id] = candidate

        for route in routes:
            for idx, mid in enumerate(route.step_memory_ids):
                mem = palace.memories.get(mid)
                kind = mem.kind.value if mem is not None else None
                step_score = route.score * (1.0 / (1 + idx))
                candidate = RetrievedMemory(
                    memory_id=mid,
                    score=step_score,
                    reason=f"route:{route.route_id}",
                    retrieval_role="support" if idx > 0 else "seed",
                    memory_kind=kind,
                )
                prev = by_id.get(mid)
                if prev is None or candidate.score > prev.score:
                    by_id[mid] = candidate

        ranked = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
        retrieved = tuple(ranked[: max(1, top_k)])
        ordered_routes = tuple(sorted(routes, key=lambda r: r.score, reverse=True))
        return RankedRecallBundle(retrieved_memories=retrieved, routes=ordered_routes)
