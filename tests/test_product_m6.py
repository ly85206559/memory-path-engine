import unittest

from memory_engine.embeddings import (
    HashingEmbeddingProvider,
    NgramHashingEmbeddingProvider,
    clear_embedding_provider_cache,
    embed_many,
    resolve_embedding_provider,
    _truncate_for_dense_embed,
)
from memory_engine.retrieve import EmbeddingTopKRetriever
from memory_engine.schema import MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


class TruncateDenseEmbedTests(unittest.TestCase):
    def test_short_text_unchanged(self) -> None:
        self.assertEqual(_truncate_for_dense_embed("hello", 100), "hello")

    def test_long_text_keeps_head_and_tail(self) -> None:
        text = "A" * 1200 + "MID" + "B" * 1200
        out = _truncate_for_dense_embed(text, 200)
        self.assertEqual(len(out), 200 + len("\n...\n"))
        self.assertTrue(out.startswith("A"))
        self.assertTrue(out.endswith("B"))
        self.assertIn("...", out)


class EmbedManyTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_embedding_provider_cache()

    def tearDown(self) -> None:
        clear_embedding_provider_cache()

    def test_ngram_and_hash_batch_match_single(self) -> None:
        texts = ["rollback recovers the API", "workers restart before paging"]
        for name in ("ngram", "hash"):
            provider = resolve_embedding_provider(name)
            batched = embed_many(provider, texts)
            singles = [provider.embed(text) for text in texts]
            self.assertEqual(batched, singles)

    def test_fastembed_batch_when_installed(self) -> None:
        try:
            import fastembed  # noqa: F401
        except ImportError:
            self.skipTest("fastembed not installed")
        provider = resolve_embedding_provider("fastembed")
        texts = [
            "rollback recovers the API gateway",
            "restart workers before paging DB",
            "rollback recovers the API gateway",
        ]
        batched = provider.embed_many(texts)
        self.assertEqual(len(batched), 3)
        self.assertEqual(batched[0], batched[2])
        self.assertEqual(provider.embed(texts[0]), batched[0])


class BatchPrefetchRetrieverTests(unittest.TestCase):
    def test_rank_candidates_prefetches_store(self) -> None:
        store = MemoryStore()
        for index, content in enumerate(
            ("alpha incident rollback", "beta worker restart", "gamma database paging")
        ):
            store.add_node(
                MemoryNode(
                    id=f"n{index}",
                    type="memo",
                    content=content,
                    weights=MemoryWeight(importance=0.5),
                )
            )
        provider = NgramHashingEmbeddingProvider(dimension=64)
        retriever = EmbeddingTopKRetriever(store, embedding_provider=provider)
        ranked = retriever.rank_candidates("rollback restart", top_k=2)
        self.assertEqual(len(ranked), 2)
        # Query + all node contents should be cached after one rank call.
        self.assertIn("rollback restart", retriever._embedding_cache)
        for node in store.nodes():
            self.assertIn(node.content, retriever._embedding_cache)


if __name__ == "__main__":
    unittest.main()
