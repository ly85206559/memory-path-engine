import unittest


class RetrievalFactoryTests(unittest.TestCase):
    def test_build_legacy_retriever_returns_mode_wrapper(self) -> None:
        from memory_engine.retrieval_factory import LegacyModeRetriever, build_legacy_retriever
        from memory_engine.store import MemoryStore

        retriever = build_legacy_retriever("lexical_baseline", MemoryStore())

        self.assertIsInstance(retriever, LegacyModeRetriever)
        self.assertEqual(retriever.retriever_mode, "lexical_baseline")
