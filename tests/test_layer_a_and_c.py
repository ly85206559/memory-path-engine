import unittest
from pathlib import Path

from memory_engine.benchmarking.application.layer_a_report import (
    LAYER_A_METRIC_SCOPE,
    annotate_external_summary,
    build_layer_a_report,
    render_layer_a_markdown,
)
from scripts.generate_layer_a_report import SLICE_PROFILES
from scripts.run_layer_c_benchmark import DEFAULT_LAYER_C_FIXTURES, build_layer_c_report


class LayerAReportTests(unittest.TestCase):
    def test_annotate_external_summary_marks_layer_a_scope(self):
        payload = annotate_external_summary({"samples": 2}, dataset_kind="hotpotqa")
        self.assertEqual(payload["metric_layer"], "A")
        self.assertEqual(payload["metric_scope"], LAYER_A_METRIC_SCOPE)
        self.assertIn("Layer B", payload["architecture_metrics_location"])

    def test_build_layer_a_report_on_tiny_fixtures(self):
        report = build_layer_a_report(
            hotpot_dataset=Path("benchmarks/external/hotpotqa/hotpot_tiny_fixture.json"),
            longmem_dataset=Path(
                "benchmarks/external/longmemeval/longmemeval_tiny_fixture.json"
            ),
            slice_profile="tiny",
            top_k=5,
        )
        self.assertEqual(report["report_kind"], "layer_a_positioning")
        self.assertEqual(report["hotpotqa"]["metric_scope"], LAYER_A_METRIC_SCOPE)
        self.assertEqual(report["longmemeval"]["metric_scope"], LAYER_A_METRIC_SCOPE)
        self.assertIn("weighted_graph", report["hotpotqa"]["modes"])
        self.assertIn("recall_at_10", next(iter(report["longmemeval"]["modes"].values())))
        markdown = render_layer_a_markdown(report)
        self.assertIn("Layer A Positioning Report", markdown)
        self.assertIn("Metric Scope Map", markdown)
        self.assertIn("medium", SLICE_PROFILES)
        self.assertIn("full", SLICE_PROFILES)


class LayerCBenchmarkTests(unittest.TestCase):
    def test_default_layer_c_fixtures_exist(self):
        root = Path("benchmarks/layer_c_minimal")
        for name in DEFAULT_LAYER_C_FIXTURES:
            self.assertTrue((root / name).exists())

    def test_layer_c_weighted_graph_hits_all_public_standin_cases(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        service = StructuredBenchmarkEvaluationService()
        for fixture_name in DEFAULT_LAYER_C_FIXTURES:
            with self.subTest(fixture=fixture_name):
                report = service.run_from_dataset_path(
                    dataset_path=Path("benchmarks/layer_c_minimal") / fixture_name,
                    retriever_mode="weighted_graph",
                    top_k=3,
                )
                self.assertEqual(report.evidence_hit_rate, 1.0)
                self.assertTrue(all(case.hit for case in report.case_reports))

    def test_build_layer_c_report_includes_transfer_scope(self):
        report = build_layer_c_report(
            fixtures=DEFAULT_LAYER_C_FIXTURES,
            modes=("weighted_graph",),
            top_k=3,
        )
        self.assertEqual(report["metric_layer"], "C")
        self.assertEqual(report["metric_scope"], "real_world_transfer")
        self.assertEqual(len(report["per_fixture"]), 2)
