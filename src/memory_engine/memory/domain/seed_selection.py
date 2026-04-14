from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memory_engine.memory.domain.enums import MemoryKind
from memory_engine.memory.domain.palace import MemoryPalace


@dataclass(frozen=True, slots=True)
class SeedSelectionInput:
    text: str
    allowed_space_ids: tuple[str, ...]
    max_seeds: int = 5
    allowed_memory_kinds: tuple[MemoryKind, ...] = ()


@dataclass(frozen=True, slots=True)
class SeedActivation:
    memory_id: str
    score: float
    reason: str
    space_id: str | None = None


class SeedSelector(Protocol):
    def select_seeds(
        self,
        palace: MemoryPalace,
        selection: SeedSelectionInput,
    ) -> tuple[SeedActivation, ...]:
        """Return ordered seed activations, highest score first."""
