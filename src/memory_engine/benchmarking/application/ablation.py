from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkReport,
    StructuredBenchmarkSuiteReport,
)

# Modes used for the evaluation.md ablation families.
ABLATION_MODES: tuple[str, ...] = (
    "lexical_baseline",
    "embedding_baseline",
    "structure_only",
    "weighted_graph",
    "activation_spreading_v1",
)

DEFAULT_ABLATION_FIXTURES: tuple[str, ...] = (
    "structure_ablation_benchmark.json",
    "multi_hop_chain_benchmark.json",
    "exception_override_path_benchmark.json",
    "contradiction_tension_benchmark.json",
)


@dataclass(frozen=True, slots=True)
class AblationFamily:
    family_id: str
    label: str
    baseline_mode: str
    full_mode: str
    primary_metric: str
    description: str


ABLATION_FAMILIES: tuple[AblationFamily, ...] = (
    AblationFamily(
        family_id="no_structure",
        label="Remove structure",
        baseline_mode="embedding_baseline",
        full_mode="structure_only",
        primary_metric="semantic_hit_rate",
        description="Flat embedding retrieval vs edge-aware structure-only expansion.",
    ),
    AblationFamily(
        family_id="no_weight",
        label="Remove weights",
        baseline_mode="structure_only",
        full_mode="weighted_graph",
        primary_metric="evidence_hit_rate",
        description="Structure without anomaly/importance weights vs weighted graph scoring.",
    ),
    AblationFamily(
        family_id="no_path_expansion",
        label="Remove path expansion",
        baseline_mode="embedding_baseline",
        full_mode="weighted_graph",
        primary_metric="path_hit_rate",
        description="No neighbor expansion vs weighted graph path expansion and replay.",
    ),
)


def summarize_latencies(latencies_ms: Iterable[float]) -> dict[str, float]:
    values = sorted(float(value) for value in latencies_ms)
    if not values:
        return {
            "count": 0.0,
            "avg_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "max_latency_ms": 0.0,
        }
    index = min(len(values) - 1, max(0, int(round(0.95 * (len(values) - 1)))))
    return {
        "count": float(len(values)),
        "avg_latency_ms": round(sum(values) / len(values), 3),
        "median_latency_ms": round(float(median(values)), 3),
        "p95_latency_ms": round(values[index], 3),
        "max_latency_ms": round(values[-1], 3),
    }


def mode_metric_snapshot(report: StructuredBenchmarkReport) -> dict[str, float | int]:
    latencies = [case.latency_ms for case in report.case_reports]
    latency = summarize_latencies(latencies)
    path_cases = sum(1 for case in report.case_reports if case.path_hit is not None)
    path_hits = sum(1 for case in report.case_reports if case.path_hit is True)
    semantic_cases = sum(1 for case in report.case_reports if case.semantic_hit is not None)
    semantic_hits = sum(1 for case in report.case_reports if case.semantic_hit is True)
    contradiction_cases = sum(
        1 for case in report.case_reports if case.contradiction_hit is not None
    )
    contradiction_hits = sum(
        1 for case in report.case_reports if case.contradiction_hit is True
    )
    return {
        "questions": report.questions,
        "evidence_hit_rate": round(report.evidence_hit_rate, 6),
        "evidence_recall": round(report.evidence_recall, 6),
        "path_hit_rate": round(path_hits / path_cases, 6) if path_cases else 0.0,
        "path_hit_cases": path_cases,
        "semantic_hit_rate": round(semantic_hits / semantic_cases, 6) if semantic_cases else 0.0,
        "semantic_hit_cases": semantic_cases,
        "contradiction_hit_rate": (
            round(contradiction_hits / contradiction_cases, 6) if contradiction_cases else 0.0
        ),
        "contradiction_hit_cases": contradiction_cases,
        "avg_latency_ms": latency["avg_latency_ms"],
        "median_latency_ms": latency["median_latency_ms"],
        "p95_latency_ms": latency["p95_latency_ms"],
        "max_latency_ms": latency["max_latency_ms"],
    }


def _delta(full_value: float, baseline_value: float) -> float:
    return round(full_value - baseline_value, 6)


def evaluate_ablation_family(
    *,
    family: AblationFamily,
    baseline_metrics: dict[str, float | int],
    full_metrics: dict[str, float | int],
) -> dict:
    primary = family.primary_metric
    baseline_primary = float(baseline_metrics.get(primary, 0.0))
    full_primary = float(full_metrics.get(primary, 0.0))
    primary_delta = _delta(full_primary, baseline_primary)
    expected_direction_met = primary_delta > 0.0 or (
        primary_delta == 0.0 and full_primary >= baseline_primary and full_primary > 0.0
    )
    # Strict: full must not be worse on the primary metric when both are defined.
    if family.family_id == "no_path_expansion" and int(full_metrics.get("path_hit_cases", 0)) == 0:
        # Path expectations may be absent on some fixtures; fall back to evidence.
        baseline_primary = float(baseline_metrics.get("evidence_hit_rate", 0.0))
        full_primary = float(full_metrics.get("evidence_hit_rate", 0.0))
        primary = "evidence_hit_rate"
        primary_delta = _delta(full_primary, baseline_primary)
        expected_direction_met = primary_delta >= 0.0

    latency_delta = _delta(
        float(full_metrics.get("avg_latency_ms", 0.0)),
        float(baseline_metrics.get("avg_latency_ms", 0.0)),
    )
    return {
        "family_id": family.family_id,
        "label": family.label,
        "description": family.description,
        "baseline_mode": family.baseline_mode,
        "full_mode": family.full_mode,
        "primary_metric": primary,
        "baseline": {
            "evidence_hit_rate": baseline_metrics.get("evidence_hit_rate", 0.0),
            "path_hit_rate": baseline_metrics.get("path_hit_rate", 0.0),
            "semantic_hit_rate": baseline_metrics.get("semantic_hit_rate", 0.0),
            "contradiction_hit_rate": baseline_metrics.get("contradiction_hit_rate", 0.0),
            "avg_latency_ms": baseline_metrics.get("avg_latency_ms", 0.0),
            "median_latency_ms": baseline_metrics.get("median_latency_ms", 0.0),
        },
        "full": {
            "evidence_hit_rate": full_metrics.get("evidence_hit_rate", 0.0),
            "path_hit_rate": full_metrics.get("path_hit_rate", 0.0),
            "semantic_hit_rate": full_metrics.get("semantic_hit_rate", 0.0),
            "contradiction_hit_rate": full_metrics.get("contradiction_hit_rate", 0.0),
            "avg_latency_ms": full_metrics.get("avg_latency_ms", 0.0),
            "median_latency_ms": full_metrics.get("median_latency_ms", 0.0),
        },
        "delta": {
            "evidence_hit_rate": _delta(
                float(full_metrics.get("evidence_hit_rate", 0.0)),
                float(baseline_metrics.get("evidence_hit_rate", 0.0)),
            ),
            "path_hit_rate": _delta(
                float(full_metrics.get("path_hit_rate", 0.0)),
                float(baseline_metrics.get("path_hit_rate", 0.0)),
            ),
            "semantic_hit_rate": _delta(
                float(full_metrics.get("semantic_hit_rate", 0.0)),
                float(baseline_metrics.get("semantic_hit_rate", 0.0)),
            ),
            "contradiction_hit_rate": _delta(
                float(full_metrics.get("contradiction_hit_rate", 0.0)),
                float(baseline_metrics.get("contradiction_hit_rate", 0.0)),
            ),
            "avg_latency_ms": latency_delta,
            "primary_metric": primary_delta,
        },
        "expected_direction_met": expected_direction_met,
    }


def build_fixture_ablation_row(suite: StructuredBenchmarkSuiteReport, *, fixture_name: str) -> dict:
    mode_metrics = {
        mode_name: mode_metric_snapshot(mode_report)
        for mode_name, mode_report in suite.modes.items()
    }
    families = []
    for family in ABLATION_FAMILIES:
        if family.baseline_mode not in mode_metrics or family.full_mode not in mode_metrics:
            continue
        families.append(
            evaluate_ablation_family(
                family=family,
                baseline_metrics=mode_metrics[family.baseline_mode],
                full_metrics=mode_metrics[family.full_mode],
            )
        )
    all_latencies = [
        case.latency_ms
        for mode_report in suite.modes.values()
        for case in mode_report.case_reports
    ]
    return {
        "fixture": fixture_name,
        "dataset_id": suite.dataset_id,
        "modes": mode_metrics,
        "ablation_families": families,
        "latency_summary": summarize_latencies(all_latencies),
    }


def aggregate_ablation_report(per_fixture: list[dict], *, modes: tuple[str, ...]) -> dict:
    overall_mode_latencies: dict[str, list[float]] = {mode: [] for mode in modes}
    overall_mode_evidence: dict[str, list[float]] = {mode: [] for mode in modes}
    family_results: dict[str, list[dict]] = {family.family_id: [] for family in ABLATION_FAMILIES}

    for fixture_row in per_fixture:
        for mode_name, metrics in fixture_row["modes"].items():
            if mode_name not in overall_mode_latencies:
                overall_mode_latencies[mode_name] = []
                overall_mode_evidence[mode_name] = []
            # Reconstruct approximate case latencies via avg * questions for aggregates
            # Prefer recompute from stored avg when questions known.
            questions = int(metrics.get("questions", 0))
            avg_latency = float(metrics.get("avg_latency_ms", 0.0))
            if questions > 0:
                overall_mode_latencies[mode_name].extend([avg_latency] * questions)
            overall_mode_evidence[mode_name].append(float(metrics.get("evidence_hit_rate", 0.0)))
        for family_row in fixture_row["ablation_families"]:
            family_results[family_row["family_id"]].append(family_row)

    overall_modes = {}
    for mode_name in modes:
        latency = summarize_latencies(overall_mode_latencies.get(mode_name, []))
        evidence_rates = overall_mode_evidence.get(mode_name, [])
        overall_modes[mode_name] = {
            **latency,
            "avg_evidence_hit_rate": (
                round(sum(evidence_rates) / len(evidence_rates), 6) if evidence_rates else 0.0
            ),
            "fixture_count": len(evidence_rates),
        }

    overall_families = []
    for family in ABLATION_FAMILIES:
        rows = family_results.get(family.family_id, [])
        if not rows:
            continue
        met = sum(1 for row in rows if row["expected_direction_met"])
        overall_families.append(
            {
                "family_id": family.family_id,
                "label": family.label,
                "baseline_mode": family.baseline_mode,
                "full_mode": family.full_mode,
                "primary_metric": family.primary_metric,
                "fixtures_evaluated": len(rows),
                "fixtures_meeting_direction": met,
                "direction_pass_rate": round(met / len(rows), 6),
                "avg_primary_delta": round(
                    sum(float(row["delta"]["primary_metric"]) for row in rows) / len(rows),
                    6,
                ),
                "avg_latency_delta_ms": round(
                    sum(float(row["delta"]["avg_latency_ms"]) for row in rows) / len(rows),
                    6,
                ),
            }
        )

    return {
        "modes": list(modes),
        "fixtures": [row["fixture"] for row in per_fixture],
        "ablation_families": [family.family_id for family in ABLATION_FAMILIES],
        "overall_modes": overall_modes,
        "overall_families": overall_families,
        "per_fixture": per_fixture,
    }
