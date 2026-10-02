import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_engine.embeddings import DiskEmbeddingCache, NgramHashingEmbeddingProvider


class DiskEmbeddingCacheTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskEmbeddingCache(Path(tmp), model_name="unit-test")
            self.assertIsNone(cache.get("hello"))
            cache.put("hello", [0.1, 0.2, 0.3])
            self.assertEqual(cache.get("hello"), [0.1, 0.2, 0.3])

    def test_optional_env_enables_disk_cache_on_dense_init_path(self) -> None:
        # Ngram has no disk cache; verify helper path used by dense providers.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"MPE_EMBEDDING_CACHE_DIR": tmp}):
                from memory_engine.embeddings import _optional_disk_cache

                cache = _optional_disk_cache("demo-model")
                self.assertIsNotNone(cache)
                assert cache is not None
                cache.put("x", [1.0])
                self.assertEqual(cache.get("x"), [1.0])
            # Without env → disabled
            with patch.dict("os.environ", {}, clear=False):
                import os

                os.environ.pop("MPE_EMBEDDING_CACHE_DIR", None)
                from memory_engine.embeddings import _optional_disk_cache as opt

                self.assertIsNone(opt("demo-model"))
            # Sanity: ngram still works
            self.assertTrue(len(NgramHashingEmbeddingProvider(dimension=8).embed("hi")) == 8)


if __name__ == "__main__":
    unittest.main()
