from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memory_engine.memory.domain.palace import MemoryPalace


@dataclass(frozen=True, slots=True)
class SpaceSelectionInput:
    """Inputs for palace space ranking, kept in the domain layer to avoid importing application DTOs."""

    text: str
    preferred_space_ids: tuple[str, ...] = ()
    max_spaces: int = 3


@dataclass(frozen=True, slots=True)
class SpaceCandidate:
    space_id: str
    score: float
    reason: str


class SpaceSelector(Protocol):
    def select_spaces(
        self,
        palace: MemoryPalace,
        selection: SpaceSelectionInput,
    ) -> tuple[SpaceCandidate, ...]:
        """Return ordered space candidates, highest score first."""
