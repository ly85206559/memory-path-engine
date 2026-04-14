from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(frozen=True, slots=True)
class RecallQuery:
    palace_id: str
    text: str
    policy: RecallPolicy = field(default_factory=RecallPolicy)
