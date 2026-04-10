import unittest

from memory_engine.anomaly import AnomalyKind, ThresholdAnomalyPolicy
from memory_engine.scoring import WeightedSumScoringStrategy
from memory_engine.schema import ActivationContext, MemoryNode, MemoryWeight


class AnomalyPolicyTests(unittest.TestCase):
    def test_threshold_policy_emits_signal_for_high_risk_node(self):
        node = MemoryNode(
            id="node-1",
            type="clause",
            content="Escalate immediately on repeated breach.",
            weights=MemoryWeight(risk=0.91, novelty=0.2),
        )

        signals = ThresholdAnomalyPolicy().signals_for_node(node=node)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].kind, AnomalyKind.WEIGHT_THRESHOLD)

    def test_weighted_sum_strategy_uses_anomaly_policy(self):
        node = MemoryNode(
            id="node-2",
            type="clause",
            content="Unexpected exception path.",
            weights=MemoryWeight(risk=0.2, novelty=0.95, confidence=0.8),
        )

        breakdown = WeightedSumScoringStrategy().score_node(
            query="unexpected exception path",
            node=node,
            semantic_score=0.8,
            context=ActivationContext(query="unexpected exception path"),
            depth=0,
        )

        self.assertEqual(breakdown.anomaly_score, 1.0)

    def test_weighted_sum_strategy_emits_semantic_scores(self):
        node = MemoryNode(
            id="node-3",
            type="clause",
            content="Unless goods are defective, Buyer must pay all invoices within 30 days.",
            attributes={
                "semantic_role": "exception",
                "contradiction_targets": ["node-4"],
            },
            weights=MemoryWeight(risk=0.2, novelty=0.85, confidence=0.8),
        )

        breakdown = WeightedSumScoringStrategy().score_node(
            query="What overrides the payment rule?",
            node=node,
            semantic_score=0.9,
            context=ActivationContext(query="What overrides the payment rule?"),
            depth=1,
            source_node_id="node-4",
        )

        self.assertGreater(breakdown.exception_score, 0.0)
        self.assertGreater(breakdown.contradiction_score, 0.0)
