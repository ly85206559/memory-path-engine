from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceRef:
    source_path: str
    section_id: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryWeight:
    importance: float = 0.0
    risk: float = 0.0
    novelty: float = 0.0
    confidence: float = 1.0
    usage_count: int = 0
    decay_factor: float = 1.0

    def bounded_score(self) -> float:
        score = (
            self.importance * 0.35
            + self.risk * 0.35
            + self.novelty * 0.15
            + self.confidence * 0.15
        )
        return max(0.0, min(score, 1.0))


@dataclass(slots=True)
class MemoryNode:
    id: str
    type: str
    content: str
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    weights: MemoryWeight = field(default_factory=MemoryWeight)
    source_ref: EvidenceRef | None = None


@dataclass(slots=True)
class MemoryEdge:
    from_id: str
    to_id: str
    edge_type: str
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    source_ref: EvidenceRef | None = None


@dataclass(slots=True)
class ActivationContext:
    query: str
    semantic_weight: float = 0.55
    structural_weight: float = 0.2
    anomaly_weight: float = 0.15
    importance_weight: float = 0.1
    max_hops: int = 2


@dataclass(slots=True)
class PathStep:
    node_id: str
    reason: str
    score: float
    via_edge_type: str | None = None


@dataclass(slots=True)
class MemoryPath:
    query: str
    steps: list[PathStep] = field(default_factory=list)
    supporting_evidence: list[EvidenceRef] = field(default_factory=list)
    final_answer: str = ""
    final_score: float = 0.0


@dataclass(slots=True)
class RetrievalResult:
    query: str
    paths: list[MemoryPath] = field(default_factory=list)

    def best_path(self) -> MemoryPath:
        if not self.paths:
            raise ValueError("No retrieval paths available.")
        return max(self.paths, key=lambda path: path.final_score)
