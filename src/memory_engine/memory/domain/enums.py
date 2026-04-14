from __future__ import annotations

from enum import StrEnum


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ROUTE = "route"


class MemoryLifecycleState(StrEnum):
    ENCODED = "encoded"
    ACTIVE = "active"
    STABILIZING = "stabilizing"
    CONSOLIDATED = "consolidated"
    FADING = "fading"
    ARCHIVED = "archived"


class MemoryLinkType(StrEnum):
    NEXT = "next"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    EXCEPTION_TO = "exception_to"
    CONTRADICTS = "contradicts"
    CAUSES = "causes"
    RECALLS = "recalls"
    SUMMARIZES = "summarizes"
    LOCATED_IN = "located_in"
    ROUTE_TO = "route_to"
