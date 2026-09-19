import unittest

from memory_engine.benchmarking.application.ablation import (
    ABLATION_FAMILIES,
    ABLATION_MODES,
    DEFAULT_ABLATION_FIXTURES,
    aggregate_ablation_report,
    evaluate_ablation_family,
    mode_metric_snapshot,
    summarize_latencies,
)
from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkCaseReport,
    StructuredBenchmarkReport,
)


class AblationHelpersTests(unittest.TestCase):
    def test_summarize_latencies_reports_avg_median_and_p95(self):
        summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 100.0])

        self.assertEqual(summary["count"], 5.0)
        self.assertEqual(summary["avg_latency_ms"], 22.0)
        self.assertEqual(summary["median_latency_ms"], 3.0)
        self.assertEqual(summary["p95_latency_ms"], 100.0)
        self.assertEqual(summary["max_latency_ms"], 100.0)

    def test_evaluate_ablation_family_marks_positive_primary_delta(self):
        family = next(item for item in ABLATION_FAMILIES if item.family_id == "no_structure")
        result = evaluate_ablation_family(
            family=family,
            baseline_metrics={
                "evidence_hit_rate": 1.0,
                "path_hit_rate": 0.0,
                "semantic_hit_rate": 0.0,
                "contradiction_hit_rate": 0.0,
                "avg_latency_ms": 1.0,
                "median_latency_ms": 1.0,
                "path_hit_cases": 0,
            },
            full_metrics={
                "evidence_hit_rate": 1.0,
                "path_hit_rate": 1.0,
                "semantic_hit_rate": 1.0,
                "contradiction_hit_rate": 0.0,
                "avg_latency_ms": 1.5,
                "median_latency_ms": 1.4,
                "path_hit_cases": 1,
            },
        )

        self.assertTrue(result["expected_direction_met"])
        self.assertEqual(result["delta"]["primary_metric"], 1.0)
        self.assertEqual(result["delta"]["avg_latency_ms"], 0.5)

    def test_mode_metric_snapshot_includes_latency_percentiles(self):
        report = StructuredBenchmarkReport(
            dataset_id="demo",
            retriever_name="weighted_graph",
            questions=2,
            evidence_hit_rate=1.0,
            evidence_recall=1.0,
            avg_latency_ms=2.0,
            case_reports=[
                StructuredBenchmarkCaseReport(
                    case_id="c1",
                    query="q1",
                    evidence_hit=True,
                    hit=True,
                    path_hit=True,
                    semantic_hit=True,
                    expected_evidence=["n1"],
                    matched_evidence=["n1"],
                    missing_evidence=[],
                    returned_node_ids=["n1"],
                    latency_ms=1.0,
                ),
                StructuredBenchmarkCaseReport(
                    case_id="c2",
                    query="q2",
                    evidence_hit=True,
                    hit=True,
                    path_hit=False,
                    semantic_hit=False,
                    expected_evidence=["n2"],
                    matched_evidence=["n2"],
                    missing_evidence=[],
                    returned_node_ids=["n2"],
                    latency_ms=3.0,
                ),
            ],
        )

        snapshot = mode_metric_snapshot(report)

        self.assertEqual(snapshot["path_hit_rate"], 0.5)
        self.assertEqual(snapshot["semantic_hit_rate"], 0.5)
        self.assertEqual(snapshot["median_latency_ms"], 2.0)
        self.assertEqual(snapshot["p95_latency_ms"], 3.0)

    def test_aggregate_ablation_report_builds_family_pass_rates(self):
        per_fixture = [
            {
                "fixture": "structure_ablation_benchmark.json",
                "dataset_id": "structure-ablation-benchmark-v1",
                "modes": {
                    mode: {
                        "questions": 1,
                        "evidence_hit_rate": 1.0 if mode != "lexical_baseline" else 0.0,
                        "evidence_recall": 1.0,
                        "path_hit_rate": 1.0 if mode in {"structure_only", "weighted_graph"} else 0.0,
                        "path_hit_cases": 1,
                        "semantic_hit_rate": 1.0 if mode != "embedding_baseline" else 0.0,
                        "semantic_hit_cases": 1,
                        "contradiction_hit_rate": 0.0,
                        "contradiction_hit_cases": 0,
                        "avg_latency_ms": 1.0,
                        "median_latency_ms": 1.0,
                        "p95_latency_ms": 1.0,
                        "max_latency_ms": 1.0,
                    }
                    for mode in ABLATION_MODES
                },
                "ablation_families": [
                    evaluate_ablation_family(
                        family=family,
                        baseline_metrics={
                            "evidence_hit_rate": 0.5,
                            "path_hit_rate": 0.0,
                            "semantic_hit_rate": 0.0,
                            "contradiction_hit_rate": 0.0,
                            "avg_latency_ms": 1.0,
                            "median_latency_ms": 1.0,
                            "path_hit_cases": 1,
                        },
                        full_metrics={
                            "evidence_hit_rate": 1.0,
                            "path_hit_rate": 1.0,
                            "semantic_hit_rate": 1.0,
                            "contradiction_hit_rate": 0.0,
                            "avg_latency_ms": 1.2,
                            "median_latency_ms": 1.2,
                            "path_hit_cases": 1,
                        },
                    )
                    for family in ABLATION_FAMILIES
                ],
                "latency_summary": summarize_latencies([1.0]),
            }
        ]

        report = aggregate_ablation_report(per_fixture, modes=ABLATION_MODES)

        self.assertEqual(report["fixtures"], ["structure_ablation_benchmark.json"])
        self.assertEqual(len(report["overall_families"]), 3)
        self.assertTrue(all(family["direction_pass_rate"] == 1.0 for family in report["overall_families"]))
        self.assertIn("structure_ablation_benchmark.json", DEFAULT_ABLATION_FIXTURES)
        self.assertIn("contradiction_tension_benchmark.json", DEFAULT_ABLATION_FIXTURES)
