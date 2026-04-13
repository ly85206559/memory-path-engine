import unittest
from pathlib import Path


class HotpotQAAdapterTests(unittest.TestCase):
    def test_supporting_facts_map_to_expected_node_ids(self):
        from memory_engine.benchmarking.adapters.hotpotqa import (
            supporting_facts_to_evidence_node_ids,
        )

        sample = {
            "_id": "unit-mini",
            "question": "q",
            "answer": "a",
            "context": [
                ["Farm story", ["The sky is blue.", "The grass is green."]],
                ["City story", ["The sky is gray.", "Traffic is loud."]],
            ],
            "supporting_facts": [["Farm story", 0]],
        }
        self.assertEqual(supporting_facts_to_evidence_node_ids(sample), ["farm_story:0"])

    def test_title_match_is_case_insensitive(self):
        from memory_engine.benchmarking.adapters.hotpotqa import (
            supporting_facts_to_evidence_node_ids,
        )

        sample = {
            "_id": "x",
            "question": "q",
            "answer": "a",
            "context": [
                ["Farm Story", ["One.", "Two."]],
            ],
            "supporting_facts": [["farm story", 1]],
        }
        self.assertEqual(supporting_facts_to_evidence_node_ids(sample), ["farm_story:1"])

    def test_lexical_baseline_hits_gold_sentence(self):
        from memory_engine.benchmarking.adapters.hotpotqa import (
            build_hotpot_memory_store,
            hotpot_sample_to_benchmark_case,
        )
        from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
        from memory_engine.benchmarking.application.service import build_retriever
        from memory_engine.benchmarking.domain.models import StructuredBenchmarkDataset

        sample = {
            "_id": "hit-001",
            "question": "sky color farm",
            "answer": "blue",
            "context": [
                ["Farm story", ["The sky is blue.", "The grass is green."]],
                ["City story", ["The sky is gray.", "Traffic is loud."]],
            ],
            "supporting_facts": [["Farm story", 0]],
        }
        store = build_hotpot_memory_store(sample)
        retriever = build_retriever("lexical_baseline", store)
        case = hotpot_sample_to_benchmark_case(sample)
        ds = StructuredBenchmarkDataset(
            dataset_id="one",
            dataset_name="one",
            domain_pack_name="hotpotqa_sentence_pack",
            document_directory="../../benchmarks/external/hotpotqa",
            cases=[case],
        )
        report = StructuredBenchmarkRunner().run(
            dataset=ds,
            retriever_name="lexical_baseline",
            retriever=retriever,
            top_k=4,
        )
        self.assertTrue(report.case_reports[0].evidence_hit)
        self.assertIn("farm_story:0", report.case_reports[0].matched_evidence)

    def test_tiny_fixture_load_and_suite_runs(self):
        from memory_engine.benchmarking.adapters.hotpotqa import (
            load_hotpotqa_json_array,
            run_hotpotqa_benchmark,
        )

        path = Path("benchmarks/external/hotpotqa/hotpot_tiny_fixture.json")
        samples = load_hotpotqa_json_array(path)
        self.assertEqual(len(samples), 2)
        suite = run_hotpotqa_benchmark(
            samples,
            retriever_modes=("lexical_baseline", "embedding_baseline"),
            top_k=8,
            dataset_id="hotpot-tiny-ci",
        )
        self.assertEqual(suite.modes["lexical_baseline"].questions, 2)
        self.assertEqual(suite.modes["embedding_baseline"].questions, 2)
        self.assertGreaterEqual(suite.modes["lexical_baseline"].evidence_hit_rate, 0.5)
        self.assertGreaterEqual(suite.modes["embedding_baseline"].evidence_hit_rate, 0.5)
