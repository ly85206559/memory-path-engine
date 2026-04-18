import unittest
from pathlib import Path

from scripts.generate_layer_b_report import DEFAULT_FIXTURES, render_markdown_report
from scripts.run_longmemeval_benchmark import build_summary_payload


class ReportingScriptsTests(unittest.TestCase):
    def test_render_layer_b_markdown_report_includes_required_metrics(self) -> None:
        report = {
            "fixtures": ["spatial_recall_benchmark.json"],
            "modes": ["weighted_graph"],
            "overall": {
                "weighted_graph": {
                    "path_hit_rate": 0.5,
                    "path_hit_cases": 2,
                    "route_hit_rate": 0.75,
                    "route_hit_cases": 4,
                    "space_hit_rate": 1.0,
                    "space_hit_cases": 1,
                    "lifecycle_hit_rate": 0.25,
                    "lifecycle_hit_cases": 1,
                    "activation_trace_hit_rate": 0.5,
                    "activation_trace_hit_cases": 2,
                    "activation_snapshot_hit_rate": 0.5,
                    "activation_snapshot_hit_cases": 2,
                }
            },
            "per_fixture": [
                {
                    "datasets": "spatial_recall_benchmark.json",
                    "modes": {
                        "weighted_graph": {
                            "path_hit_rate": 0.5,
                            "path_hit_cases": 2,
                            "route_hit_rate": 0.75,
                            "route_hit_cases": 4,
                            "space_hit_rate": 1.0,
                            "space_hit_cases": 1,
                            "lifecycle_hit_rate": 0.25,
                            "lifecycle_hit_cases": 1,
                            "activation_trace_hit_rate": 0.5,
                            "activation_trace_hit_cases": 2,
                            "activation_snapshot_hit_rate": 0.5,
                            "activation_snapshot_hit_cases": 2,
                        }
                    },
                }
            ],
        }

        markdown = render_markdown_report(report)

        self.assertIn("path_hit_rate", markdown)
        self.assertIn("route_hit_rate", markdown)
        self.assertIn("space_hit_rate", markdown)
        self.assertIn("lifecycle_hit_rate", markdown)
        self.assertIn("activation_trace_hit_rate", markdown)
        self.assertIn("activation_snapshot_hit_rate", markdown)
        self.assertIn("path_cases", markdown)
        self.assertIn("trace_cases", markdown)
        self.assertIn("snapshot_cases", markdown)

    def test_default_layer_b_fixtures_include_path_and_trace_coverage(self) -> None:
        self.assertIn("exception_override_benchmark.json", DEFAULT_FIXTURES)
        self.assertIn("exception_override_path_benchmark.json", DEFAULT_FIXTURES)
        self.assertIn("multi_hop_chain_benchmark.json", DEFAULT_FIXTURES)
        self.assertIn("activation_snapshot_benchmark.json", DEFAULT_FIXTURES)

    def test_build_longmemeval_summary_payload_compacts_mode_metrics(self) -> None:
        class _ModeReport:
            questions = 12
            recall_at_5 = 0.5
            recall_at_10 = 0.75
            ndcg_at_10 = 0.61
            avg_latency_ms = 12.3

        class _Suite:
            modes = {"weighted_graph": _ModeReport()}

        summary = build_summary_payload(
            _Suite(),
            dataset_path=Path("benchmarks/external/longmemeval/data/sample.json"),
            sample_count=12,
            granularity="session",
        )

        self.assertEqual(summary["samples"], 12)
        self.assertEqual(summary["granularity"], "session")
        self.assertIn("weighted_graph", summary["modes"])
        self.assertEqual(summary["modes"]["weighted_graph"]["recall_at_10"], 0.75)
