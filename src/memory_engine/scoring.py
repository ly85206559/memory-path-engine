from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memory_engine.anomaly import AnomalyPolicy, ThresholdAnomalyPolicy
from memory_engine.memory_state import MemoryStatePolicy
from memory_engine.schema import ActivationContext, MemoryNode


@dataclass(slots=True)
class ScoreBreakdown:
    semantic_score: float
    structural_score: float
    anomaly_score: float
    importance_score: float
    total_score: float


class ScoringStrategy(Protocol):
    def score_node(
        self,
        *,
        query: str,
        node: MemoryNode,
        semantic_score: float,
        context: ActivationContext,
        depth: int,
    ) -> ScoreBreakdown:
        """Return a score breakdown for a node."""


class WeightedSumScoringStrategy:
    def __init__(
        self,
        anomaly_threshold: float = 0.8,
        depth_penalty: float = 0.25,
        anomaly_policy: AnomalyPolicy | None = None,
        memory_state_policy: MemoryStatePolicy | None = None,
    ) -> None:
        self.anomaly_threshold = anomaly_threshold
        self.depth_penalty = depth_penalty
        self.anomaly_policy = anomaly_policy or ThresholdAnomalyPolicy(
            risk_threshold=anomaly_threshold,
            novelty_threshold=anomaly_threshold,
        )
        self.memory_state_policy = memory_state_policy or MemoryStatePolicy()

    def score_node(
        self,
        *,
        query: str,
        node: MemoryNode,
        semantic_score: float,
        context: ActivationContext,
        depth: int,
    ) -> ScoreBreakdown:
        del query
        structural_score = max(0.0, 1.0 - depth * self.depth_penalty)
        anomaly_score = 1.0 if self._is_anomalous(node) else 0.0
        importance_score = self.memory_state_policy.effective_weight_score(node.weights)
        weighted_bonus_gate = semantic_score
        total_score = (
            semantic_score * context.semantic_weight
            + structural_score * context.structural_weight
            + anomaly_score * context.anomaly_weight * weighted_bonus_gate
            + importance_score * context.importance_weight * weighted_bonus_gate
        )
        return ScoreBreakdown(
            semantic_score=semantic_score,
            structural_score=structural_score,
            anomaly_score=anomaly_score,
            importance_score=importance_score,
            total_score=total_score,
        )

    def _is_anomalous(self, node: MemoryNode) -> bool:
        return bool(self.anomaly_policy.signals_for_node(node=node))


class StructureOnlyScoringStrategy:
    def __init__(self, depth_penalty: float = 0.25) -> None:
        self.depth_penalty = depth_penalty

    def score_node(
        self,
        *,
        query: str,
        node: MemoryNode,
        semantic_score: float,
        context: ActivationContext,
        depth: int,
    ) -> ScoreBreakdown:
        del query, node
        structural_score = max(0.0, 1.0 - depth * self.depth_penalty)
        total_score = (
            semantic_score * context.semantic_weight
            + structural_score * context.structural_weight
        )
        return ScoreBreakdown(
            semantic_score=semantic_score,
            structural_score=structural_score,
            anomaly_score=0.0,
            importance_score=0.0,
            total_score=total_score,
        )
