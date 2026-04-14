from __future__ import annotations

from dataclasses import dataclass

from memory_engine.memory.domain.retrieval_result import RecallRoute, RetrievedMemory


@dataclass(frozen=True, slots=True)
class RankedRecallBundle:
    """Ranked outputs ready to assemble into PalaceRecallResult."""

    retrieved_memories: tuple[RetrievedMemory, ...]
    routes: tuple[RecallRoute, ...]
