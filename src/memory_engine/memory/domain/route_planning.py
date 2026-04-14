from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.retrieval_result import RecallRoute
from memory_engine.memory.domain.seed_selection import SeedActivation


@dataclass(frozen=True, slots=True)
class RoutePlanningInput:
    text: str
    top_k: int = 3
    max_hops: int = 2


class RoutePlanner(Protocol):
    def plan_routes(
        self,
        palace: MemoryPalace,
        planning: RoutePlanningInput,
        seeds: tuple[SeedActivation, ...],
    ) -> tuple[RecallRoute, ...]:
        """Produce recall routes, preferring explicit RouteMemory when available."""
