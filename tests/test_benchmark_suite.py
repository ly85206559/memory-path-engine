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
            retriever_modes=("embedding_baseline", "weighted_graph", "activation_spreading_v1"),
            top_k=3,
        )

        self.assertEqual(suite_report.dataset_id, "example-runbook-benchmark-v1")
        self.assertIn("embedding_baseline", suite_report.modes)
        self.assertIn("weighted_graph", suite_report.modes)
        self.assertIn("activation_spreading_v1", suite_report.modes)
        self.assertTrue(suite_report.comparison.per_question)
        self.assertIn("activation_spreading_v1", suite_report.comparison.mode_summary)
        self.assertIn(
            "avg_activated_nodes",
            suite_report.comparison.mode_summary["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "avg_activation_trace_length",
            suite_report.comparison.mode_summary["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "activation_trace_hit_rate",
            suite_report.comparison.mode_summary["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "route_hit_rate",
            suite_report.comparison.mode_summary["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "activation_snapshot_hit_rate",
            suite_report.comparison.mode_summary["activation_spreading_v1"].model_dump(),
        )
        self.assertEqual(
            suite_report.comparison.per_question[0].case_id,
            suite_report.modes["weighted_graph"].case_reports[0].case_id,
        )
        first_question = suite_report.comparison.per_question[0]
        self.assertEqual(
            set(first_question.modes.keys()),
            set(suite_report.modes.keys()),
        )
        self.assertEqual(
            set(first_question.best_modes),
            {
                mode_name
                for mode_name, mode_result in first_question.modes.items()
                if mode_result.hit
            },
        )
        self.assertEqual(
            first_question.missed_by_all,
            not any(mode_result.hit for mode_result in first_question.modes.values()),
        )
        self.assertIn(
            "path_hit",
            first_question.modes["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "activation_trace_hit",
            first_question.modes["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "route_hit",
            first_question.modes["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "activation_snapshot_hit",
            first_question.modes["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "semantic_hit",
            first_question.modes["activation_spreading_v1"].model_dump(),
        )
        self.assertIn(
            "contradiction_hit",
            first_question.modes["activation_spreading_v1"].model_dump(),
        )

    def test_suite_reports_activation_trace_hit_rate_for_trace_aware_fixture(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/exception_override_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("weighted_graph", "activation_spreading_v1"),
            top_k=2,
        )

        self.assertEqual(
            suite_report.comparison.mode_summary["weighted_graph"].activation_trace_hit_rate,
            1.0,
        )
        self.assertEqual(
            suite_report.comparison.mode_summary["activation_spreading_v1"].activation_trace_hit_rate,
            1.0,
        )
        first_question = suite_report.comparison.per_question[0]
        self.assertTrue(first_question.modes["weighted_graph"].activation_trace_hit)
        self.assertTrue(first_question.modes["activation_spreading_v1"].activation_trace_hit)
        self.assertEqual(
            suite_report.comparison.mode_summary["weighted_graph"].activation_trace_hit_cases,
            1,
        )
        self.assertEqual(
            suite_report.comparison.mode_summary["activation_spreading_v1"].activation_trace_hit_cases,
            1,
        )

    def test_optional_metric_hit_rate_uses_only_applicable_cases(self):
        from memory_engine.benchmarking.application.service import build_comparison_report
        from memory_engine.benchmarking.domain.models import StructuredBenchmarkCaseReport, StructuredBenchmarkReport

        report = StructuredBenchmarkReport(
            dataset_id="d1",
            retriever_name="fake",
            questions=2,
            evidence_hit_rate=1.0,
            evidence_recall=1.0,
            avg_latency_ms=1.0,
            case_reports=[
                StructuredBenchmarkCaseReport(
                    case_id="c1",
                    query="q1",
                    evidence_hit=True,
                    hit=True,
                    path_hit=True,
                    expected_evidence=["a"],
                    matched_evidence=["a"],
                    missing_evidence=[],
                    returned_node_ids=["a"],
                    latency_ms=1.0,
                ),
                StructuredBenchmarkCaseReport(
                    case_id="c2",
                    query="q2",
                    evidence_hit=True,
                    hit=True,
                    path_hit=None,
                    expected_evidence=["b"],
                    matched_evidence=["b"],
                    missing_evidence=[],
                    returned_node_ids=["b"],
                    latency_ms=1.0,
                ),
            ],
        )

        comparison = build_comparison_report({"fake": report})

        self.assertEqual(comparison.mode_summary["fake"].path_hit_rate, 1.0)
        self.assertEqual(comparison.mode_summary["fake"].path_hit_cases, 1)

    def test_suite_supports_dynamic_memory_experiment_modes(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/example_runbook_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=(
                "weighted_graph_static",
                "weighted_graph_dynamic",
                "activation_spreading_static",
                "activation_spreading_dynamic",
            ),
            top_k=3,
        )

        self.assertIn("weighted_graph_static", suite_report.modes)
        self.assertIn("weighted_graph_dynamic", suite_report.modes)
        self.assertIn("activation_spreading_static", suite_report.modes)
        self.assertIn("activation_spreading_dynamic", suite_report.modes)
        self.assertIn("weighted_graph_static", suite_report.comparison.mode_summary)
        self.assertIn("weighted_graph_dynamic", suite_report.comparison.mode_summary)
        self.assertIn("activation_spreading_static", suite_report.comparison.mode_summary)
        self.assertIn("activation_spreading_dynamic", suite_report.comparison.mode_summary)

    def test_suite_exposes_dynamic_priming_divergence_for_activation_spreading(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/dynamic_memory_priming_benchmark.json")
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
        self.assertEqual(static_prime.best_path_hops, dynamic_prime.best_path_hops)
        self.assertEqual(static_prime.activation_trace_length, dynamic_prime.activation_trace_length)
        self.assertEqual(static_probe.case_id, "probe-009")
        self.assertEqual(dynamic_probe.case_id, "probe-009")
        self.assertTrue(static_probe.evidence_hit)
        self.assertTrue(static_probe.path_hit)
        self.assertTrue(static_probe.hit)
        self.assertFalse(dynamic_probe.evidence_hit)
        self.assertFalse(dynamic_probe.path_hit)
        self.assertFalse(dynamic_probe.hit)
        self.assertEqual(static_probe.matched_evidence, ["dynamic_memory_priming_runbook:7"])
        self.assertEqual(dynamic_probe.matched_evidence, [])
        self.assertGreater(
            suite_report.comparison.mode_summary["activation_spreading_static"].evidence_hit_rate,
            suite_report.comparison.mode_summary["activation_spreading_dynamic"].evidence_hit_rate,
        )
        self.assertGreater(
            suite_report.comparison.mode_summary["activation_spreading_static"].path_hit_rate,
            suite_report.comparison.mode_summary["activation_spreading_dynamic"].path_hit_rate,
        )

    def test_suite_exposes_contract_exception_priming_path_divergence(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/contract_exception_priming_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("activation_spreading_static", "activation_spreading_dynamic"),
            top_k=1,
        )

        static_probe = suite_report.modes["activation_spreading_static"].case_reports[-1]
        dynamic_probe = suite_report.modes["activation_spreading_dynamic"].case_reports[-1]

        self.assertEqual(static_probe.case_id, "probe-009")
        self.assertEqual(dynamic_probe.case_id, "probe-009")
        self.assertTrue(static_probe.evidence_hit)
        self.assertTrue(dynamic_probe.evidence_hit)
        self.assertFalse(static_probe.path_hit)
        self.assertTrue(dynamic_probe.path_hit)
        self.assertFalse(static_probe.hit)
        self.assertTrue(dynamic_probe.hit)
        self.assertLess(
            suite_report.comparison.mode_summary["activation_spreading_static"].path_hit_rate,
            suite_report.comparison.mode_summary["activation_spreading_dynamic"].path_hit_rate,
        )

    def test_suite_exposes_exception_override_path_diagnostics(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/exception_override_path_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("weighted_graph", "activation_spreading_v1"),
            top_k=2,
        )

        first_question = suite_report.comparison.per_question[0]
        self.assertTrue(first_question.modes["weighted_graph"].hit)
        self.assertTrue(first_question.modes["activation_spreading_v1"].hit)
        self.assertEqual(
            suite_report.comparison.mode_summary["activation_spreading_v1"].contradiction_hit_rate,
            suite_report.comparison.mode_summary["weighted_graph"].contradiction_hit_rate,
        )
        self.assertEqual(
            suite_report.comparison.mode_summary["activation_spreading_v1"].path_hit_rate,
            suite_report.comparison.mode_summary["weighted_graph"].path_hit_rate,
        )

    def test_suite_exposes_dynamic_override_sequence_divergence(self):
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
        self.assertFalse(static_probe.hit)
        self.assertTrue(dynamic_probe.hit)
        self.assertGreater(
            suite_report.comparison.mode_summary["activation_spreading_dynamic"].path_hit_rate,
            suite_report.comparison.mode_summary["activation_spreading_static"].path_hit_rate,
        )
