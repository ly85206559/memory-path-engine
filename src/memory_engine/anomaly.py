from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from memory_engine.schema import MemoryNode


class AnomalyKind(str, Enum):
    WEIGHT_THRESHOLD = "weight_threshold"


@dataclass(frozen=True, slots=True)
class AnomalySignal:
    kind: AnomalyKind
    severity: float
    source: str
    explanation: str = ""
    rule_id: str | None = None


class AnomalyPolicy(Protocol):
    def signals_for_node(self, *, node: MemoryNode) -> tuple[AnomalySignal, ...]:
        """Return anomaly signals for a node."""


class ThresholdAnomalyPolicy:
    def __init__(
        self,
        *,
        risk_threshold: float = 0.8,
        novelty_threshold: float = 0.8,
    ) -> None:
        self.risk_threshold = risk_threshold
        self.novelty_threshold = novelty_threshold

    def signals_for_node(self, *, node: MemoryNode) -> tuple[AnomalySignal, ...]:
        signals: list[AnomalySignal] = []
        if node.weights.risk >= self.risk_threshold:
            signals.append(
                AnomalySignal(
                    kind=AnomalyKind.WEIGHT_THRESHOLD,
                    severity=node.weights.risk,
                    source="memory_weight",
                    explanation="risk threshold exceeded",
                    rule_id="risk_threshold",
                )
            )
        if node.weights.novelty >= self.novelty_threshold:
            signals.append(
                AnomalySignal(
                    kind=AnomalyKind.WEIGHT_THRESHOLD,
                    severity=node.weights.novelty,
                    source="memory_weight",
                    explanation="novelty threshold exceeded",
                    rule_id="novelty_threshold",
                )
            )
        return tuple(signals)
