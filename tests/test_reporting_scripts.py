import unittest
from pathlib import Path

from scripts.generate_ablation_report import (
    DEFAULT_ABLATION_FIXTURES,
    render_markdown_report as render_ablation_markdown_report,
)
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
        self.assertIn("contradiction_tension_benchmark.json", DEFAULT_FIXTURES)
        self.assertIn("multi_hop_chain_benchmark.json", DEFAULT_FIXTURES)
        self.assertIn("activation_snapshot_benchmark.json", DEFAULT_FIXTURES)

    def test_render_ablation_markdown_report_includes_family_and_latency_sections(self) -> None:
        report = {
            "fixtures": ["structure_ablation_benchmark.json"],
            "modes": ["embedding_baseline", "structure_only", "weighted_graph"],
            "ablation_families": ["no_structure", "no_weight", "no_path_expansion"],
            "overall_modes": {
                "weighted_graph": {
                    "count": 1.0,
                    "avg_latency_ms": 1.2,
                    "median_latency_ms": 1.2,
                    "p95_latency_ms": 1.2,
                    "max_latency_ms": 1.2,
                    "avg_evidence_hit_rate": 1.0,
                    "fixture_count": 1,
                }
            },
            "overall_families": [
                {
                    "family_id": "no_structure",
                    "label": "Remove structure",
                    "baseline_mode": "embedding_baseline",
                    "full_mode": "structure_only",
                    "primary_metric": "semantic_hit_rate",
                    "fixtures_evaluated": 1,
                    "fixtures_meeting_direction": 1,
                    "direction_pass_rate": 1.0,
                    "avg_primary_delta": 1.0,
                    "avg_latency_delta_ms": 0.2,
                }
            ],
            "per_fixture": [
                {
                    "fixture": "structure_ablation_benchmark.json",
                    "modes": {
                        "weighted_graph": {
                            "evidence_hit_rate": 1.0,
                            "path_hit_rate": 1.0,
                            "semantic_hit_rate": 1.0,
                            "contradiction_hit_rate": 0.0,
                            "avg_latency_ms": 1.2,
                            "median_latency_ms": 1.2,
                            "p95_latency_ms": 1.2,
                        }
                    },
                    "ablation_families": [
                        {
                            "family_id": "no_structure",
                            "delta": {
                                "primary_metric": 1.0,
                                "evidence_hit_rate": 0.0,
                                "path_hit_rate": 1.0,
                                "semantic_hit_rate": 1.0,
                                "avg_latency_ms": 0.2,
                            },
                            "expected_direction_met": True,
                        }
                    ],
                }
            ],
        }

        markdown = render_ablation_markdown_report(report)

        self.assertIn("Ablation and Latency Report", markdown)
        self.assertIn("Overall Latency by Mode", markdown)
        self.assertIn("Ablation Family Summary", markdown)
        self.assertIn("no_structure", markdown)
        self.assertIn("median_ms", markdown)

    def test_default_ablation_fixtures_cover_required_families(self) -> None:
        self.assertIn("structure_ablation_benchmark.json", DEFAULT_ABLATION_FIXTURES)
        self.assertIn("multi_hop_chain_benchmark.json", DEFAULT_ABLATION_FIXTURES)
        self.assertIn("contradiction_tension_benchmark.json", DEFAULT_ABLATION_FIXTURES)

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
        self.assertEqual(summary["metric_layer"], "A")
        self.assertEqual(summary["metric_scope"], "external_positioning")
        self.assertIn("weighted_graph", summary["modes"])
        self.assertEqual(summary["modes"]["weighted_graph"]["recall_at_10"], 0.75)
