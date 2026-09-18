import tempfile
import unittest
from pathlib import Path

from memory_engine.anomaly import (
    AnomalyKind,
    CompositeAnomalyPolicy,
    ContradictionAttributeAnomalyPolicy,
    LexicalConflictAnomalyPolicy,
    ThresholdAnomalyPolicy,
    default_anomaly_policy,
)
from memory_engine.domain_pack import ExampleContractPack
from memory_engine.scoring import WeightedSumScoringStrategy
from memory_engine.schema import ActivationContext, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.semantics import contradiction_candidates
from memory_engine.store import MemoryStore


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

    def test_lexical_policy_detects_conflict_wording(self):
        node = MemoryNode(
            id="n1",
            type="clause",
            content="Order Form payment terms conflict with Agreement payment terms.",
            weights=MemoryWeight(risk=0.2, novelty=0.2),
        )

        signals = LexicalConflictAnomalyPolicy().signals_for_node(node=node)

        kinds = {signal.kind for signal in signals}
        self.assertIn(AnomalyKind.LEXICAL_CONFLICT, kinds)

    def test_lexical_policy_detects_exception_markers(self):
        node = MemoryNode(
            id="n2",
            type="clause",
            content="Unless goods are defective, Buyer must pay within 30 days.",
            weights=MemoryWeight(risk=0.2, novelty=0.2),
        )

        signals = LexicalConflictAnomalyPolicy().signals_for_node(node=node)

        kinds = {signal.kind for signal in signals}
        self.assertIn(AnomalyKind.EXCEPTION_MARKER, kinds)

    def test_attribute_policy_detects_contradiction_targets(self):
        node = MemoryNode(
            id="n3",
            type="clause",
            content="Buyer must pay within 15 days.",
            attributes={"contradiction_targets": ["n4"], "contradicts_target": "n4"},
            weights=MemoryWeight(risk=0.2, novelty=0.2),
        )

        signals = ContradictionAttributeAnomalyPolicy().signals_for_node(node=node)

        self.assertTrue(signals)
        self.assertTrue(all(signal.kind == AnomalyKind.CONTRADICTION_ATTRIBUTE for signal in signals))

    def test_default_policy_composites_threshold_and_lexical(self):
        node = MemoryNode(
            id="n5",
            type="clause",
            content="These terms conflict with the prior schedule.",
            weights=MemoryWeight(risk=0.95, novelty=0.2),
        )

        signals = default_anomaly_policy().signals_for_node(node=node)
        kinds = {signal.kind for signal in signals}

        self.assertIn(AnomalyKind.WEIGHT_THRESHOLD, kinds)
        self.assertIn(AnomalyKind.LEXICAL_CONFLICT, kinds)

    def test_composite_deduplicates_by_rule_id(self):
        policy = CompositeAnomalyPolicy(
            (
                ThresholdAnomalyPolicy(risk_threshold=0.5, novelty_threshold=0.99),
                ThresholdAnomalyPolicy(risk_threshold=0.5, novelty_threshold=0.99),
            )
        )
        node = MemoryNode(
            id="n6",
            type="clause",
            content="High risk clause.",
            weights=MemoryWeight(risk=0.9, novelty=0.1),
        )

        signals = policy.signals_for_node(node=node)

        self.assertEqual(len(signals), 1)


class ContradictionEdgeTests(unittest.TestCase):
    def test_contradicts_edges_produce_candidates(self):
        nodes = [
            MemoryNode(id="a:1", type="clause", content="Pay within 30 days."),
            MemoryNode(id="a:2", type="clause", content="Pay within 15 days."),
        ]
        edges = [
            MemoryEdge(from_id="a:1", to_id="a:2", edge_type="contradicts", weight=0.85),
        ]

        candidates = contradiction_candidates(nodes, edges)

        self.assertEqual(len(candidates), 1)
        pair = {candidates[0].left_node_id, candidates[0].right_node_id}
        self.assertEqual(pair, {"a:1", "a:2"})

    def test_contract_pack_ingests_explicit_contradicts_edges(self):
        contract_text = "\n".join(
            [
                "# Conflict Demo",
                "",
                "## Agreement",
                "1 Buyer must pay all invoices within 30 days of receipt.",
                "2 Buyer payment of invoices under this Agreement conflicts with Order Form payment of invoices.",
                "",
                "## Order Form",
                "3 Buyer must pay all invoices within 15 days of receipt.",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "conflict_demo.md"
            path.write_text(contract_text, encoding="utf-8")
            store = MemoryStore()
            ExampleContractPack().ingest_document(path, store)

            contradict_edges = [
                edge for edge in store.edges() if edge.edge_type == "contradicts"
            ]
            self.assertTrue(contradict_edges)
            annotated = store.get_node("conflict_demo:2")
            self.assertEqual(annotated.attributes.get("contradicts_target"), "conflict_demo:1")
            self.assertIn("conflict_demo:1", annotated.attributes.get("contradiction_targets", []))

            candidates = contradiction_candidates(store.nodes(), store.edges())
            self.assertTrue(
                any(
                    {c.left_node_id, c.right_node_id} == {"conflict_demo:1", "conflict_demo:2"}
                    for c in candidates
                )
            )
