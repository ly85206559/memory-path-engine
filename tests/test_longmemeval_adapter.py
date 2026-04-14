import unittest
from pathlib import Path


class LongMemEvalAdapterTests(unittest.TestCase):
    def test_gold_session_ids_map_to_expected_node_ids(self):
        from memory_engine.benchmarking.adapters.longmemeval import longmemeval_gold_node_ids

        sample = {
            "question_id": "q-001",
            "answer_session_ids": ["sess-2", "sess-4"],
        }
        self.assertEqual(
            longmemeval_gold_node_ids(sample),
            ["q-001:sess-2", "q-001:sess-4"],
        )

    def test_build_longmemeval_memory_store_creates_session_nodes(self):
        from memory_engine.benchmarking.adapters.longmemeval import (
            build_longmemeval_memory_palace,
            build_longmemeval_memory_store,
            longmemeval_session_node_id,
        )

        sample = {
            "question_id": "q-002",
            "question_type": "single-session-user",
            "question": "What tea does Kai prefer?",
            "question_date": "2025-02-01",
            "haystack_session_ids": ["sess-1", "sess-2"],
            "haystack_dates": ["2025-01-01", "2025-01-15"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I switched from coffee to tea."},
                    {"role": "assistant", "content": "Noted."},
                ],
                [
                    {"role": "user", "content": "My favorite tea is jasmine green tea.", "has_answer": True},
                    {"role": "assistant", "content": "Jasmine green tea sounds great."},
                ],
            ],
            "answer_session_ids": ["sess-2"],
        }

        palace = build_longmemeval_memory_palace(sample)
        store = build_longmemeval_memory_store(sample)
        node_ids = {node.id for node in store.nodes()}
        self.assertEqual(len(palace.spaces), 2)
        self.assertIn(longmemeval_session_node_id(sample, "sess-1"), node_ids)
        self.assertIn(longmemeval_session_node_id(sample, "sess-2"), node_ids)
        self.assertTrue(store.neighbors(longmemeval_session_node_id(sample, "sess-1")))

    def test_tiny_fixture_runs_suite(self):
        from memory_engine.benchmarking.adapters.longmemeval import (
            load_longmemeval_json,
            run_longmemeval_benchmark,
        )

        path = Path("benchmarks/external/longmemeval/longmemeval_tiny_fixture.json")
        samples = load_longmemeval_json(path)
        self.assertEqual(len(samples), 2)
        suite = run_longmemeval_benchmark(
            samples,
            retriever_modes=("embedding_baseline", "weighted_graph"),
            top_k=5,
            dataset_id="longmemeval-tiny-ci",
        )
        self.assertEqual(suite.benchmark_name, "LongMemEval")
        self.assertEqual(suite.modes["embedding_baseline"].questions, 2)
        self.assertIn("weighted_graph", suite.modes)
        self.assertGreaterEqual(suite.modes["embedding_baseline"].recall_at_5, 0.5)
        self.assertGreaterEqual(suite.modes["weighted_graph"].recall_at_5, 0.5)
        self.assertTrue(suite.modes["embedding_baseline"].metadata["v1_memory_architecture"])
        self.assertIn("space_count", suite.modes["embedding_baseline"].case_reports[0].metadata)

    def test_validate_longmemeval_sample_rejects_unknown_answer_session(self):
        from memory_engine.benchmarking.adapters.longmemeval import validate_longmemeval_sample

        sample = {
            "question_id": "bad-001",
            "question": "q",
            "haystack_session_ids": ["sess-1"],
            "haystack_sessions": [[{"role": "user", "content": "hello"}]],
            "answer_session_ids": ["sess-2"],
        }
        with self.assertRaises(ValueError):
            validate_longmemeval_sample(sample)
