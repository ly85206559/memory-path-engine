import unittest

from memory_engine.api import (
    apply_online_memory_step,
    project_dual_views,
    recall_and_reason,
    recall_from_store,
    reason_from_recall,
)
from memory_engine.memory.application.forgetting_policies import (
    AggressiveForgettingPolicy,
    MildForgettingPolicy,
    policy_by_name,
)
from memory_engine.memory.application.multi_representation import (
    episodic_id_for,
    semantic_id_for,
)
from memory_engine.reasoning import PathReasoner
from memory_engine.replay import path_answer
from memory_engine.schema import MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


def _seed_store() -> MemoryStore:
    store = MemoryStore()
    obligation = MemoryNode(
        id="clause:1",
        type="clause",
        content="Buyer must pay all invoices within 30 days.",
        attributes={"semantic_role": "obligation"},
        weights=MemoryWeight(importance=0.6, risk=0.2, novelty=0.2, confidence=0.9),
    )
    exception = MemoryNode(
        id="clause:2",
        type="clause",
        content="Unless goods are defective, Buyer must pay all invoices within 30 days.",
        attributes={"semantic_role": "exception"},
        weights=MemoryWeight(importance=0.75, risk=0.4, novelty=0.85, confidence=0.95),
    )
    remedy = MemoryNode(
        id="clause:3",
        type="clause",
        content="If goods are defective, Buyer may withhold invoice payment until Seller cures.",
        attributes={"semantic_role": "remedy"},
        weights=MemoryWeight(importance=0.8, risk=0.7, novelty=0.5, confidence=0.95),
    )
    unused = MemoryNode(
        id="clause:unused",
        type="clause",
        content="Notices must be delivered in writing to the registered office.",
        attributes={"semantic_role": "action"},
        weights=MemoryWeight(importance=0.4, risk=0.1, novelty=0.1, confidence=0.8),
    )
    for node in (obligation, exception, remedy, unused):
        store.add_node(node)
    store.add_edge(
        MemoryEdge(
            from_id="clause:2",
            to_id="clause:1",
            edge_type="exception_to",
            weight=0.8,
            bidirectional=True,
        )
    )
    store.add_edge(
        MemoryEdge(
            from_id="clause:2",
            to_id="clause:3",
            edge_type="depends_on",
            weight=0.7,
        )
    )
    return store


class MultiRepresentationTests(unittest.TestCase):
    def test_project_dual_views_creates_linked_episodic_and_semantic(self):
        store = _seed_store()
        projected = project_dual_views(store, node_ids=["clause:2"])
        self.assertEqual(len(projected), 1)
        ids = projected[0]
        episodic = store.get_node(ids.episodic_node_id)
        semantic = store.get_node(ids.semantic_node_id)
        source = store.get_node("clause:2")

        self.assertEqual(episodic.attributes["memory_kind"], "episodic")
        self.assertEqual(semantic.attributes["memory_kind"], "semantic")
        self.assertTrue(source.attributes["has_dual_representation"])
        self.assertEqual(episodic_id_for("clause:2"), ids.episodic_node_id)
        self.assertEqual(semantic_id_for("clause:2"), ids.semantic_node_id)

        edge_keys = {
            (edge.from_id, edge.to_id, edge.edge_type) for edge in store.edges()
        }
        self.assertIn((ids.episodic_node_id, "clause:2", "recalls"), edge_keys)
        self.assertIn((ids.semantic_node_id, "clause:2", "summarizes"), edge_keys)
        self.assertIn((ids.episodic_node_id, ids.semantic_node_id, "summarizes"), edge_keys)
        self.assertIn((ids.semantic_node_id, ids.episodic_node_id, "summarizes"), edge_keys)

    def test_project_dual_views_is_idempotent(self):
        store = _seed_store()
        project_dual_views(store, node_ids=["clause:1"])
        before = len(store.nodes())
        project_dual_views(store, node_ids=["clause:1"])
        self.assertEqual(len(store.nodes()), before)


class ForgettingPolicyTests(unittest.TestCase):
    def test_policy_by_name_resolves_mild_and_aggressive(self):
        self.assertIsInstance(policy_by_name("mild"), MildForgettingPolicy)
        self.assertIsInstance(policy_by_name("aggressive"), AggressiveForgettingPolicy)

    def test_aggressive_forgetting_decays_unused_faster_than_mild(self):
        mild_store = _seed_store()
        aggressive_store = _seed_store()

        query = "What if goods are defective and payment is disputed?"
        mild_result = recall_from_store(
            mild_store, query, retriever_mode="weighted_graph", project_palace=False
        )
        aggressive_result = recall_from_store(
            aggressive_store, query, retriever_mode="weighted_graph", project_palace=False
        )
        self.assertIsNotNone(mild_result.legacy)
        self.assertIsNotNone(aggressive_result.legacy)

        for _ in range(4):
            apply_online_memory_step(mild_store, mild_result.legacy, policy="mild")
            apply_online_memory_step(
                aggressive_store, aggressive_result.legacy, policy="aggressive"
            )

        mild_unused = mild_store.get_node("clause:unused").weights.decay_factor
        aggressive_unused = aggressive_store.get_node("clause:unused").weights.decay_factor
        self.assertLess(aggressive_unused, mild_unused)
        self.assertLess(aggressive_unused, 0.5)
        self.assertGreaterEqual(mild_unused, 0.7)


class PathReasoningTests(unittest.TestCase):
    def test_path_reasoner_composes_hop_citations(self):
        store = _seed_store()
        path = path_answer(
            "defective goods payment",
            [
                (store.get_node("clause:2"), 0.9, "seed", None),
                (store.get_node("clause:3"), 0.85, "hop", "depends_on"),
            ],
        )
        answered = PathReasoner().reason(query="defective goods payment", path=path, store=store)
        self.assertIn("clause:2", answered.cited_node_ids)
        self.assertIn("clause:3", answered.cited_node_ids)
        self.assertIn("depends_on", answered.path_edge_types)
        self.assertIn("Given that dependency,", answered.answer)
        self.assertIn("withhold invoice payment", answered.answer)
        self.assertGreater(answered.confidence, 0.0)
        self.assertEqual(len(answered.hop_explanations), 2)

    def test_recall_and_reason_closes_the_loop(self):
        store = _seed_store()
        unified, answered, snapshots = recall_and_reason(
            store,
            "What if goods are defective?",
            retriever_mode="weighted_graph",
            forgetting_policy="mild",
        )
        self.assertTrue(unified.legacy and unified.legacy.paths)
        self.assertTrue(answered.cited_node_ids)
        self.assertTrue(answered.answer)
        self.assertIsNotNone(snapshots)
        self.assertIn("clause:unused", snapshots or {})

    def test_reason_from_recall_handles_empty_paths(self):
        from memory_engine.schema import RetrievalResult

        answered = reason_from_recall("empty", RetrievalResult(query="empty", paths=[]))
        self.assertEqual(answered.answer, "")
        self.assertEqual(answered.cited_node_ids, ())


if __name__ == "__main__":
    unittest.main()
