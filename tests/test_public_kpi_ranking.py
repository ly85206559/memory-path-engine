import unittest

from memory_engine.benchmarking.application.public_benchmarks import ranked_node_ids_from_result
from memory_engine.embeddings import NgramHashingEmbeddingProvider, bm25_score, cosine_similarity
from memory_engine.memory.domain.retrieval_result import (
    PalaceRecallResult,
    RecallRoute,
    RetrievedMemory,
)
from memory_engine.schema import MemoryNode, MemoryPath, MemoryWeight, PathStep, RetrievalResult


class PublicKpiRankingTests(unittest.TestCase):
    def test_ranked_node_ids_prefer_score_over_path_walk_order(self):
        palace_result = PalaceRecallResult(
            query="q",
            retrieved_memories=(
                RetrievedMemory(memory_id="noise-a", score=0.4, reason="support", retrieval_role="support"),
                RetrievedMemory(memory_id="noise-b", score=0.35, reason="support", retrieval_role="support"),
                RetrievedMemory(memory_id="gold", score=0.91, reason="seed", retrieval_role="seed"),
                RetrievedMemory(memory_id="alt", score=0.88, reason="seed", retrieval_role="seed"),
            ),
            routes=(
                RecallRoute(
                    route_id="r1",
                    route_kind="legacy",
                    step_memory_ids=("noise-a", "noise-b", "gold"),
                    score=0.95,
                ),
            ),
        )
        result = palace_result.to_legacy_retrieval_result()
        self.assertEqual(
            ranked_node_ids_from_result(result, top_k=3),
            ["gold", "alt", "noise-a"],
        )

    def test_from_legacy_result_keeps_best_score_per_node(self):
        low = MemoryNode(id="n1", type="clause", content="a", weights=MemoryWeight())
        high = MemoryNode(id="n1", type="clause", content="a", weights=MemoryWeight())
        other = MemoryNode(id="n2", type="clause", content="b", weights=MemoryWeight())
        paths = [
            MemoryPath(
                query="q",
                steps=(
                    PathStep(node_id="n1", score=0.4, reason="weak"),
                    PathStep(node_id="n2", score=0.3, reason="support"),
                ),
                final_answer="a",
                final_score=0.9,
            ),
            MemoryPath(
                query="q",
                steps=(PathStep(node_id="n1", score=0.95, reason="strong"),),
                final_answer="a",
                final_score=0.8,
            ),
        ]
        # Keep nodes referenced so reconstruction stays valid.
        del low, high, other
        result = RetrievalResult(query="q", paths=paths)
        palace = PalaceRecallResult.from_legacy_result(result)
        by_id = {item.memory_id: item for item in palace.retrieved_memories}
        self.assertAlmostEqual(by_id["n1"].score, 0.95)
        self.assertEqual(by_id["n1"].retrieval_role, "seed")
        ranked = ranked_node_ids_from_result(
            RetrievalResult(query="q", paths=paths, palace_result=palace),
            top_k=2,
        )
        self.assertEqual(ranked[0], "n1")


class EmbeddingProductProviderTests(unittest.TestCase):
    def test_ngram_provider_is_deterministic_and_nonzero(self):
        provider = NgramHashingEmbeddingProvider(dimension=64)
        left = provider.embed("yoga classes downtown studio")
        right = provider.embed("yoga classes downtown studio")
        other = provider.embed("completely unrelated bicycle inventory")
        self.assertEqual(left, right)
        self.assertGreater(cosine_similarity(left, right), 0.99)
        self.assertGreater(cosine_similarity(left, left), cosine_similarity(left, other))

    def test_bm25_prefers_informative_overlap(self):
        docs = [
            "I take yoga classes at the riverside studio every Tuesday.",
            "We talked about weather, traffic, and lunch plans.",
        ]
        doc_freq = {"yoga": 1, "class": 1, "riversid": 1, "studio": 1, "tuesday": 1, "weather": 1}
        score_gold = bm25_score(
            "Where do I take yoga classes?",
            docs[0],
            avgdl=10.0,
            doc_freq=doc_freq,
            doc_count=2,
        )
        score_noise = bm25_score(
            "Where do I take yoga classes?",
            docs[1],
            avgdl=10.0,
            doc_freq=doc_freq,
            doc_count=2,
        )
        self.assertGreater(score_gold, score_noise)


if __name__ == "__main__":
    unittest.main()
