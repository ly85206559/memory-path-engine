import unittest

from memory_engine.benchmarking.application.public_benchmarks import (
    compute_ndcg_at_k,
    ranked_node_ids_from_result,
)
from memory_engine.embeddings import NgramHashingEmbeddingProvider
from memory_engine.retrieve import HybridRetriever, WeightedGraphRetriever
from memory_engine.schema import MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


def _store_with_scrambled_importance() -> MemoryStore:
    """Haystack where graph scoring bonuses would bury the true lexical winner."""
    store = MemoryStore()
    nodes = [
        ("gold", "rollback recovers the API gateway after deploy failure", 0.2),
        ("distractor_a", "workers restart paging noise unrelated topic", 0.95),
        ("distractor_b", "database owner paging schedule weekly review", 0.9),
        ("distractor_c", "cache warm-up before traffic spike window", 0.85),
        ("near", "rollback plan recovers gateway services carefully", 0.3),
    ]
    for node_id, content, importance in nodes:
        store.add_node(
            MemoryNode(
                id=node_id,
                type="memo",
                content=content,
                weights=MemoryWeight(importance=importance, confidence=0.5),
            )
        )
    # Chain distractors ahead of gold in graph walk order.
    store.add_edge(MemoryEdge(from_id="distractor_a", to_id="distractor_b", edge_type="next_unit"))
    store.add_edge(MemoryEdge(from_id="distractor_b", to_id="distractor_c", edge_type="next_unit"))
    store.add_edge(MemoryEdge(from_id="distractor_c", to_id="gold", edge_type="next_unit"))
    store.add_edge(MemoryEdge(from_id="gold", to_id="near", edge_type="next_unit"))
    return store


class HybridSeedRerankTests(unittest.TestCase):
    def test_hybrid_preserves_seed_order_for_public_ranking(self) -> None:
        store = _store_with_scrambled_importance()
        query = "rollback recovers the API gateway"
        hybrid = HybridRetriever(
            store,
            embedding_provider=NgramHashingEmbeddingProvider(dimension=64),
        )
        result = hybrid.search(query, top_k=5)
        ranked = ranked_node_ids_from_result(result, top_k=5)
        self.assertEqual(ranked[0], "gold")
        self.assertTrue(result.palace_result.metadata.get("hybrid_seed_rerank"))
        self.assertGreaterEqual(compute_ndcg_at_k(["gold"], ranked, k=5), 0.99)

    def test_weighted_graph_does_not_force_seed_rerank(self) -> None:
        # Layer B path: WeightedGraphRetriever must not gain hybrid_seed_rerank.
        store = _store_with_scrambled_importance()
        result = WeightedGraphRetriever(store).search(
            "rollback recovers the API gateway",
            top_k=5,
        )
        self.assertFalse(
            bool((result.palace_result.metadata or {}).get("hybrid_seed_rerank"))
        )


if __name__ == "__main__":
    unittest.main()
