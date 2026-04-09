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

    def test_exception_override_fixture_runs_with_activation_spreading(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/exception_override_benchmark.json")
        report = StructuredBenchmarkEvaluationService().run_from_dataset_path(
            dataset_path=dataset_path,
            retriever_mode="activation_spreading_v1",
            top_k=2,
        )

        self.assertEqual(report.dataset_id, "exception-override-benchmark-v1")
        self.assertTrue(report.case_reports[0].surfaced_semantic_roles)

    def test_multi_hop_chain_fixture_runs_with_activation_spreading(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/multi_hop_chain_benchmark.json")
        report = StructuredBenchmarkEvaluationService().run_from_dataset_path(
            dataset_path=dataset_path,
            retriever_mode="activation_spreading_v1",
            top_k=2,
        )

        self.assertEqual(report.dataset_id, "multi-hop-chain-benchmark-v1")
        self.assertGreaterEqual(report.case_reports[0].best_path_hops, 1)
