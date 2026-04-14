from __future__ import annotations

from dataclasses import dataclass, field

from memory_engine.memory.domain.enums import MemoryLinkType
from memory_engine.memory.domain.memory_types import Memory
from memory_engine.memory.domain.value_objects import PalaceLocation
from memory_engine.schema import EvidenceRef


@dataclass(slots=True)
class PalaceSpace:
    space_id: str
    name: str
    location: PalaceLocation
    parent_space_id: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class MemoryLink:
    from_memory_id: str
    to_memory_id: str
    link_type: MemoryLinkType
    strength: float = 1.0
    confidence: float = 1.0
    source: EvidenceRef | None = None
    bidirectional: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryPalace:
    palace_id: str
    spaces: dict[str, PalaceSpace] = field(default_factory=dict)
    memories: dict[str, Memory] = field(default_factory=dict)
    links: list[MemoryLink] = field(default_factory=list)

    def add_space(self, space: PalaceSpace) -> None:
        self.spaces[space.space_id] = space

    def add_memory(self, memory: Memory) -> None:
        self.memories[memory.memory_id] = memory

    def add_link(self, link: MemoryLink) -> None:
        self.links.append(link)

    def memory(self, memory_id: str) -> Memory:
        return self.memories[memory_id]

    def outbound_links(self, memory_id: str) -> list[MemoryLink]:
        return [link for link in self.links if link.from_memory_id == memory_id]
