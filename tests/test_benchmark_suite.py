import unittest
from pathlib import Path


class StructuredBenchmarkSuiteTests(unittest.TestCase):
    def test_evaluation_service_runs_typed_suite_from_dataset_file(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/example_runbook_benchmark.json")

        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("embedding_baseline", "weighted_graph"),
            top_k=3,
        )

        self.assertEqual(suite_report.dataset_id, "example-runbook-benchmark-v1")
        self.assertIn("embedding_baseline", suite_report.modes)
        self.assertIn("weighted_graph", suite_report.modes)
        self.assertTrue(suite_report.comparison.per_question)
        self.assertIn("weighted_graph", suite_report.comparison.mode_summary)
        self.assertEqual(
            suite_report.comparison.per_question[0].case_id,
            suite_report.modes["weighted_graph"].case_reports[0].case_id,
        )
