from __future__ import annotations

from dataclasses import dataclass

from memory_engine.memory.domain.enums import MemoryLifecycleState


@dataclass(frozen=True, slots=True)
class ConsolidationPolicy:
    minimum_group_size: int = 2
    max_source_memories_per_group: int = 6
    allowed_source_states: tuple[MemoryLifecycleState, ...] = (
        MemoryLifecycleState.ACTIVE,
        MemoryLifecycleState.STABILIZING,
        MemoryLifecycleState.CONSOLIDATED,
    )


@dataclass(frozen=True, slots=True)
class ConsolidationGroup:
    palace_id: str
    scenario_tag: str
    space_id: str
    source_memory_ids: tuple[str, ...]
    shared_symbolic_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConsolidationArtifact:
    memory_id: str
    concept_type: str
    source_memory_ids: tuple[str, ...]
