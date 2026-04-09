import unittest

from memory_engine.retrieve import BaselineTopKRetriever, WeightedGraphRetriever
from memory_engine.schema import EvidenceRef, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


def build_store() -> MemoryStore:
    store = MemoryStore()
    store.add_node(
        MemoryNode(
            id="contract:1",
            type="clause",
            content="Supplier shall deliver goods within 10 days.",
            weights=MemoryWeight(importance=0.5, risk=0.2, novelty=0.1, confidence=0.9),
            source_ref=EvidenceRef(source_path="contract.md", section_id="1"),
        )
    )
    store.add_node(
        MemoryNode(
            id="contract:2",
            type="clause",
            content="If Supplier misses delivery, Buyer may issue notice and a 5 day cure period applies.",
            weights=MemoryWeight(importance=0.8, risk=0.7, novelty=0.4, confidence=0.9),
            source_ref=EvidenceRef(source_path="contract.md", section_id="2"),
        )
    )
    store.add_node(
        MemoryNode(
            id="contract:3",
            type="clause",
            content="If Supplier fails to cure, Buyer may terminate and recover direct damages.",
            weights=MemoryWeight(importance=0.9, risk=0.95, novelty=0.85, confidence=0.95),
            source_ref=EvidenceRef(source_path="contract.md", section_id="3"),
        )
    )
    store.add_edge(MemoryEdge(from_id="contract:2", to_id="contract:3", edge_type="depends_on", weight=0.8))
    return store


class RetrieveTests(unittest.TestCase):
    def test_baseline_returns_lexical_hit(self):
        result = BaselineTopKRetriever(build_store()).search("delivery cure period", top_k=1)
        self.assertEqual(result.best_path().steps[0].node_id, "contract:2")

    def test_weighted_graph_retriever_replays_neighbor_path(self):
        result = WeightedGraphRetriever(build_store()).search(
            "What happens if delivery is late and not cured?",
            top_k=1,
        )
        node_ids = [step.node_id for step in result.best_path().steps]
        self.assertIn("contract:3", node_ids)
        self.assertTrue(result.best_path().supporting_evidence)
