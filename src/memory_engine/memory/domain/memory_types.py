from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory_engine.memory.domain.encoding import EncodingProfile
from memory_engine.memory.domain.enums import MemoryKind
from memory_engine.memory.domain.memory_state import DomainMemoryState
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile
from memory_engine.schema import EvidenceRef


@dataclass(slots=True)
class Memory:
    memory_id: str
    palace_id: str
    kind: MemoryKind = field(init=False)
    location: PalaceLocation
    content: str
    salience: SalienceProfile
    source: EvidenceRef | None = None
    state: DomainMemoryState = field(default_factory=DomainMemoryState)
    encoding: EncodingProfile = field(default_factory=EncodingProfile)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodicMemory(Memory):
    episode_id: str = ""
    timestamp: str | None = None
    participants: tuple[str, ...] = ()
    event_type: str = "event"

    def __post_init__(self) -> None:
        self.kind = MemoryKind.EPISODIC


@dataclass(slots=True)
class SemanticMemory(Memory):
    concept_id: str = ""
    concept_type: str = "concept"
    canonical_form: str = ""
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.kind = MemoryKind.SEMANTIC


@dataclass(slots=True)
class RouteMemory(Memory):
    route_id: str = ""
    start_memory_id: str = ""
    ordered_waypoints: tuple[str, ...] = ()
    route_kind: str = "timeline"

    def __post_init__(self) -> None:
        self.kind = MemoryKind.ROUTE
