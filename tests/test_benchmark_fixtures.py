import unittest
from pathlib import Path


class BenchmarkFixtureTests(unittest.TestCase):
    def test_runbook_benchmark_fixture_loads_and_runs(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )
        from memory_engine.benchmarking.infrastructure.json_repository import (
            JsonStructuredBenchmarkDatasetRepository,
        )

        dataset_path = Path("benchmarks/structured_memory/example_runbook_benchmark.json")
        dataset = JsonStructuredBenchmarkDatasetRepository().load(dataset_path)
        self.assertEqual(dataset.domain_pack_name, "example_runbook_pack")

        report = StructuredBenchmarkEvaluationService().run_from_dataset_path(
            dataset_path=dataset_path,
            retriever_mode="weighted_graph",
            top_k=3,
        )
        self.assertGreaterEqual(report.evidence_recall, 0.5)
