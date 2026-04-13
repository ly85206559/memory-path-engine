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

    def test_dynamic_memory_priming_fixture_exposes_static_dynamic_divergence(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/dynamic_memory_priming_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("activation_spreading_static", "activation_spreading_dynamic"),
            top_k=1,
        )

        static_probe = suite_report.modes["activation_spreading_static"].case_reports[-1]
        dynamic_probe = suite_report.modes["activation_spreading_dynamic"].case_reports[-1]

        self.assertEqual(static_probe.case_id, "probe-009")
        self.assertEqual(dynamic_probe.case_id, "probe-009")
        self.assertTrue(static_probe.hit)
        self.assertFalse(dynamic_probe.hit)
        self.assertEqual(static_probe.matched_evidence, ["dynamic_memory_priming_runbook:7"])
        self.assertEqual(dynamic_probe.matched_evidence, [])

    def test_contract_exception_priming_fixture_exposes_path_divergence(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/contract_exception_priming_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("activation_spreading_static", "activation_spreading_dynamic"),
            top_k=1,
        )

        static_prime = suite_report.modes["activation_spreading_static"].case_reports[0]
        dynamic_prime = suite_report.modes["activation_spreading_dynamic"].case_reports[0]
        static_probe = suite_report.modes["activation_spreading_static"].case_reports[-1]
        dynamic_probe = suite_report.modes["activation_spreading_dynamic"].case_reports[-1]

        self.assertEqual(static_prime.case_id, "prime-001")
        self.assertEqual(dynamic_prime.case_id, "prime-001")
        self.assertTrue(static_prime.hit)
        self.assertTrue(dynamic_prime.hit)
        self.assertEqual(static_prime.activation_trace_length, dynamic_prime.activation_trace_length)
        self.assertEqual(static_probe.case_id, "probe-009")
        self.assertEqual(dynamic_probe.case_id, "probe-009")
        self.assertTrue(static_probe.evidence_hit)
        self.assertTrue(dynamic_probe.evidence_hit)
        self.assertFalse(static_probe.path_hit)
        self.assertTrue(dynamic_probe.path_hit)
        self.assertFalse(static_probe.hit)
        self.assertTrue(dynamic_probe.hit)
        self.assertEqual(static_probe.matched_evidence, ["dynamic_exception_priming_contract:7"])
        self.assertEqual(dynamic_probe.matched_evidence, ["dynamic_exception_priming_contract:7"])

    def test_exception_override_path_fixture_requires_spreading_for_full_hit(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/exception_override_path_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("weighted_graph", "activation_spreading_v1"),
            top_k=2,
        )

        weighted_case = suite_report.modes["weighted_graph"].case_reports[0]
        spreading_case = suite_report.modes["activation_spreading_v1"].case_reports[0]

        self.assertTrue(weighted_case.evidence_hit)
        self.assertFalse(weighted_case.path_hit)
        self.assertFalse(weighted_case.semantic_hit)
        self.assertTrue(weighted_case.contradiction_hit)
        self.assertFalse(weighted_case.hit)
        self.assertTrue(spreading_case.path_hit)
        self.assertTrue(spreading_case.semantic_hit)
        self.assertTrue(spreading_case.contradiction_hit)
        self.assertTrue(spreading_case.hit)

    def test_dynamic_override_sequence_fixture_exposes_path_divergence(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/dynamic_override_sequence_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("activation_spreading_static", "activation_spreading_dynamic"),
            top_k=2,
        )

        static_probe = suite_report.modes["activation_spreading_static"].case_reports[-1]
        dynamic_probe = suite_report.modes["activation_spreading_dynamic"].case_reports[-1]

        self.assertEqual(static_probe.case_id, "probe-007")
        self.assertEqual(dynamic_probe.case_id, "probe-007")
        self.assertTrue(static_probe.evidence_hit)
        self.assertTrue(dynamic_probe.evidence_hit)
        self.assertFalse(static_probe.path_hit)
        self.assertTrue(dynamic_probe.path_hit)
        self.assertFalse(static_probe.hit)
        self.assertTrue(dynamic_probe.hit)

    def test_structure_ablation_fixture_highlights_graph_semantics(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/structure_ablation_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=(
                "embedding_baseline",
                "structure_only",
                "weighted_graph",
                "activation_spreading_v1",
            ),
            top_k=2,
        )

        embedding_case = suite_report.modes["embedding_baseline"].case_reports[0]
        structure_case = suite_report.modes["structure_only"].case_reports[0]

        self.assertTrue(embedding_case.evidence_hit)
        self.assertFalse(embedding_case.semantic_hit)
        self.assertTrue(structure_case.semantic_hit)
        self.assertGreater(
            suite_report.comparison.mode_summary["structure_only"].avg_activated_nodes,
            suite_report.comparison.mode_summary["embedding_baseline"].avg_activated_nodes,
        )
