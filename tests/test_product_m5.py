import os
import unittest
from unittest import mock

from memory_engine.embeddings import (
    HashingEmbeddingProvider,
    NgramHashingEmbeddingProvider,
    clear_embedding_provider_cache,
    resolve_embedding_provider,
)
from memory_engine.retrieval_factory import build_legacy_retriever
from memory_engine.store import MemoryStore


class ResolveEmbeddingProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_embedding_provider_cache()

    def tearDown(self) -> None:
        clear_embedding_provider_cache()

    def test_default_is_ngram(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MPE_EMBEDDING", None)
            provider = resolve_embedding_provider()
        self.assertIsInstance(provider, NgramHashingEmbeddingProvider)

    def test_caches_same_backend_instance(self) -> None:
        left = resolve_embedding_provider("ngram")
        right = resolve_embedding_provider("ngram")
        self.assertIs(left, right)

    def test_explicit_hash_and_ngram(self) -> None:
        self.assertIsInstance(resolve_embedding_provider("hash"), HashingEmbeddingProvider)
        self.assertIsInstance(resolve_embedding_provider("ngram"), NgramHashingEmbeddingProvider)

    def test_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"MPE_EMBEDDING": "hash"}):
            self.assertIsInstance(resolve_embedding_provider(), HashingEmbeddingProvider)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_embedding_provider("not-a-backend")

    def test_fastembed_missing_package_message(self) -> None:
        with mock.patch.dict("sys.modules", {"fastembed": None}):
            # Force import path to fail inside provider __init__.
            import builtins

            real_import = builtins.__import__

            def blocked(name, *args, **kwargs):
                if name == "fastembed" or name.startswith("fastembed."):
                    raise ImportError("blocked for test")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=blocked):
                with self.assertRaises(ImportError) as ctx:
                    resolve_embedding_provider("fastembed")
        self.assertIn("memory-path-engine[embed]", str(ctx.exception))


class FactoryEmbeddingWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_embedding_provider_cache()

    def tearDown(self) -> None:
        clear_embedding_provider_cache()

    def test_hybrid_defaults_to_product_provider(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MPE_EMBEDDING", None)
            retriever = build_legacy_retriever("hybrid", MemoryStore())
        self.assertIsInstance(
            retriever.embedding_provider,
            NgramHashingEmbeddingProvider,
        )

    def test_hybrid_honors_explicit_embedding(self) -> None:
        retriever = build_legacy_retriever("hybrid", MemoryStore(), embedding="hash")
        self.assertIsInstance(retriever.embedding_provider, HashingEmbeddingProvider)

    def test_weighted_graph_keeps_hash_unless_explicit(self) -> None:
        # Layer B stability: non-product modes must not silently switch to ngram/env.
        retriever = build_legacy_retriever("weighted_graph", MemoryStore())
        # LegacyModeRetriever only injects provider when explicitly set for non-product modes.
        self.assertIsNone(retriever.embedding_provider)

    def test_weighted_graph_accepts_explicit_embedding(self) -> None:
        retriever = build_legacy_retriever(
            "weighted_graph",
            MemoryStore(),
            embedding="ngram",
        )
        self.assertIsInstance(retriever.embedding_provider, NgramHashingEmbeddingProvider)


class FastEmbedOptionalTests(unittest.TestCase):
    def test_fastembed_smoke_when_installed(self) -> None:
        try:
            import fastembed  # noqa: F401
        except ImportError:
            self.skipTest("fastembed not installed")
        provider = resolve_embedding_provider("fastembed")
        vector = provider.embed("rollback recovers the API gateway")
        self.assertGreater(len(vector), 0)
        # Second call hits cache and stays deterministic.
        self.assertEqual(provider.embed("rollback recovers the API gateway"), vector)


if __name__ == "__main__":
    unittest.main()
