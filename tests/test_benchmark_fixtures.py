import unittest
from pathlib import Path


class BenchmarkFixtureTests(unittest.TestCase):
    def test_all_structured_benchmark_json_files_load(self):
        from memory_engine.benchmarking.infrastructure.json_repository import (
            JsonStructuredBenchmarkDatasetRepository,
        )

        repository = JsonStructuredBenchmarkDatasetRepository()
        fixtures_dir = Path("benchmarks/structured_memory")
        dataset_paths = sorted(fixtures_dir.glob("*.json"))

        self.assertTrue(dataset_paths)
        for dataset_path in dataset_paths:
            with self.subTest(dataset_path=dataset_path.name):
                dataset = repository.load(dataset_path)
                self.assertTrue(dataset.dataset_id)
                self.assertTrue(dataset.cases)
                self.assertTrue((dataset_path.parent / dataset.document_directory).resolve().exists())

    def test_contract_benchmark_fixture_loads_and_runs(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )
        from memory_engine.benchmarking.infrastructure.json_repository import (
            JsonStructuredBenchmarkDatasetRepository,
        )

        dataset_path = Path("benchmarks/structured_memory/example_contract_benchmark.json")
        dataset = JsonStructuredBenchmarkDatasetRepository().load(dataset_path)
        self.assertEqual(dataset.domain_pack_name, "example_contract_pack")

        report = StructuredBenchmarkEvaluationService().run_from_dataset_path(
            dataset_path=dataset_path,
            retriever_mode="weighted_graph",
            top_k=3,
        )
        self.assertEqual(report.dataset_id, "example-contract-benchmark-v1")
        self.assertGreaterEqual(report.evidence_recall, 0.5)

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
        self.assertTrue(report.case_reports[0].contradiction_hit)
        self.assertTrue(report.case_reports[0].surfaced_contradictions)
        self.assertTrue(report.case_reports[0].activation_trace_hit)
        self.assertGreater(report.case_reports[0].best_path_contradiction_score, 0.0)
        self.assertEqual(report.case_reports[0].activation_trace_length, 5)
        self.assertEqual(
            report.case_reports[0].activation_stopped_reasons,
            ["below_threshold"],
        )

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
