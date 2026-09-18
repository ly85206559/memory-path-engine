from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from memory_engine.schema import MemoryNode


class AnomalyKind(str, Enum):
    WEIGHT_THRESHOLD = "weight_threshold"
    LEXICAL_CONFLICT = "lexical_conflict"
    EXCEPTION_MARKER = "exception_marker"
    CONTRADICTION_ATTRIBUTE = "contradiction_attribute"


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


class LexicalConflictAnomalyPolicy:
    """Emit signals when node text carries conflict or exception wording."""

    CONFLICT_TOKENS = (
        "conflict",
        "conflicts",
        "contradict",
        "contradicts",
        "contrary to",
        "inconsistent",
    )
    EXCEPTION_TOKENS = (
        "unless",
        "except",
        "notwithstanding",
        "override",
        "overrides",
    )

    def signals_for_node(self, *, node: MemoryNode) -> tuple[AnomalySignal, ...]:
        lowered = node.content.lower()
        signals: list[AnomalySignal] = []
        if any(token in lowered for token in self.CONFLICT_TOKENS):
            signals.append(
                AnomalySignal(
                    kind=AnomalyKind.LEXICAL_CONFLICT,
                    severity=0.9,
                    source="lexical",
                    explanation="lexical conflict or tension wording",
                    rule_id="lexical_conflict",
                )
            )
        if any(token in lowered for token in self.EXCEPTION_TOKENS):
            signals.append(
                AnomalySignal(
                    kind=AnomalyKind.EXCEPTION_MARKER,
                    severity=0.85,
                    source="lexical",
                    explanation="exception or override wording",
                    rule_id="exception_marker",
                )
            )
        return tuple(signals)


class ContradictionAttributeAnomalyPolicy:
    """Emit signals when ingest annotated contradiction or exception targets."""

    def signals_for_node(self, *, node: MemoryNode) -> tuple[AnomalySignal, ...]:
        signals: list[AnomalySignal] = []
        contradiction_targets = node.attributes.get("contradiction_targets") or []
        if contradiction_targets:
            signals.append(
                AnomalySignal(
                    kind=AnomalyKind.CONTRADICTION_ATTRIBUTE,
                    severity=0.95,
                    source="node_attributes",
                    explanation="node carries contradiction targets",
                    rule_id="contradiction_targets",
                )
            )
        if node.attributes.get("exception_target") or node.attributes.get("contradicts_target"):
            signals.append(
                AnomalySignal(
                    kind=AnomalyKind.CONTRADICTION_ATTRIBUTE,
                    severity=0.9,
                    source="node_attributes",
                    explanation="node is linked by exception or contradicts edge",
                    rule_id="semantic_override_target",
                )
            )
        return tuple(signals)


class CompositeAnomalyPolicy:
    """Combine multiple anomaly policies and de-duplicate by rule_id."""

    def __init__(self, policies: tuple[AnomalyPolicy, ...]) -> None:
        self.policies = policies

    def signals_for_node(self, *, node: MemoryNode) -> tuple[AnomalySignal, ...]:
        signals: list[AnomalySignal] = []
        seen_rule_ids: set[str] = set()
        for policy in self.policies:
            for signal in policy.signals_for_node(node=node):
                rule_key = signal.rule_id or f"{signal.kind.value}:{signal.explanation}"
                if rule_key in seen_rule_ids:
                    continue
                seen_rule_ids.add(rule_key)
                signals.append(signal)
        return tuple(signals)


def default_anomaly_policy(
    *,
    risk_threshold: float = 0.8,
    novelty_threshold: float = 0.8,
) -> CompositeAnomalyPolicy:
    return CompositeAnomalyPolicy(
        (
            ThresholdAnomalyPolicy(
                risk_threshold=risk_threshold,
                novelty_threshold=novelty_threshold,
            ),
            LexicalConflictAnomalyPolicy(),
            ContradictionAttributeAnomalyPolicy(),
        )
    )
