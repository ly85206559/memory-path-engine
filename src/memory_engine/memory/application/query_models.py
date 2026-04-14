from __future__ import annotations

from dataclasses import dataclass, field

from memory_engine.memory.domain.enums import MemoryKind


@dataclass(frozen=True, slots=True)
class RecallSeed:
    memory_id: str
    score: float
    reason: str = "seed"


@dataclass(frozen=True, slots=True)
class RecallPolicy:
    retriever_mode: str = "weighted_graph"
    top_k: int = 3
    max_hops: int = 2
    max_spaces: int = 3
    max_seeds: int = 5
    allow_legacy_fallback: bool = True
    allowed_memory_kinds: tuple[MemoryKind, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallQuery:
    palace_id: str
    text: str
    preferred_space_ids: tuple[str, ...] = ()
    policy: RecallPolicy = field(default_factory=RecallPolicy)
