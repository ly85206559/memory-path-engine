import unittest
from pathlib import Path

from scripts.generate_layer_b_report import render_markdown_report
from scripts.run_longmemeval_benchmark import build_summary_payload


class ReportingScriptsTests(unittest.TestCase):
    def test_render_layer_b_markdown_report_includes_required_metrics(self) -> None:
        report = {
            "fixtures": ["spatial_recall_benchmark.json"],
            "modes": ["weighted_graph"],
            "overall": {
                "weighted_graph": {
                    "path_hit_rate": 0.5,
                    "route_hit_rate": 0.75,
                    "space_hit_rate": 1.0,
                    "lifecycle_hit_rate": 0.25,
                    "activation_snapshot_hit_rate": 0.5,
                }
            },
            "per_fixture": [
                {
                    "datasets": "spatial_recall_benchmark.json",
                    "modes": {
                        "weighted_graph": {
                            "path_hit_rate": 0.5,
                            "route_hit_rate": 0.75,
                            "space_hit_rate": 1.0,
                            "lifecycle_hit_rate": 0.25,
                            "activation_snapshot_hit_rate": 0.5,
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
        self.assertIn("activation_snapshot_hit_rate", markdown)

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
