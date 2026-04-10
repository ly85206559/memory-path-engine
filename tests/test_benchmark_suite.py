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
            0.0,
        )
        self.assertEqual(
            suite_report.comparison.mode_summary["activation_spreading_v1"].activation_trace_hit_rate,
            1.0,
        )
        first_question = suite_report.comparison.per_question[0]
        self.assertFalse(first_question.modes["weighted_graph"].activation_trace_hit)
        self.assertTrue(first_question.modes["activation_spreading_v1"].activation_trace_hit)

    def test_suite_supports_dynamic_memory_experiment_modes(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_path = Path("benchmarks/structured_memory/example_runbook_benchmark.json")
        suite_report = StructuredBenchmarkEvaluationService().run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=("weighted_graph_static", "weighted_graph_dynamic"),
            top_k=3,
        )

        self.assertIn("weighted_graph_static", suite_report.modes)
        self.assertIn("weighted_graph_dynamic", suite_report.modes)
        self.assertIn("weighted_graph_static", suite_report.comparison.mode_summary)
        self.assertIn("weighted_graph_dynamic", suite_report.comparison.mode_summary)
