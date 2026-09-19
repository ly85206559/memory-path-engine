from __future__ import annotations

from pathlib import Path
from typing import Any

from memory_engine.benchmarking.adapters.hotpotqa import (
    load_hotpotqa_json_array,
    run_hotpotqa_benchmark,
    summarize_hotpotqa_suite,
)
from memory_engine.benchmarking.adapters.longmemeval import (
    load_longmemeval_json,
    run_longmemeval_benchmark,
)

LAYER_A_METRIC_SCOPE = "external_positioning"
LAYER_B_METRIC_SCOPE = "architecture_mechanism"
LAYER_C_METRIC_SCOPE = "real_world_transfer"

LAYER_A_DISCLAIMER = (
    "Layer A metrics are for external positioning only. "
    "They do not validate path shape, semantic roles, contradiction pairs, "
    "or dynamic-memory behavior. Use Layer B / Layer C for those claims."
)

DEFAULT_HOTPOT_MODES: tuple[str, ...] = (
    "lexical_baseline",
    "embedding_baseline",
    "weighted_graph",
    "activation_spreading_v1",
)
DEFAULT_LONGMEM_MODES: tuple[str, ...] = (
    "embedding_baseline",
    "weighted_graph",
    "activation_spreading_v1",
)

SLICE_PROFILES: dict[str, dict[str, int]] = {
    "tiny": {"hotpot_limit": 0, "longmem_limit": 0},  # 0 = all samples in the chosen file
    "medium": {"hotpot_limit": 64, "longmem_limit": 50},
    "full": {"hotpot_limit": 0, "longmem_limit": 0},
}


def annotate_external_summary(payload: dict[str, Any], *, dataset_kind: str) -> dict[str, Any]:
    """Attach Layer A scope metadata so dashboards do not confuse external vs architecture metrics."""
    annotated = dict(payload)
    annotated["metric_layer"] = "A"
    annotated["metric_scope"] = LAYER_A_METRIC_SCOPE
    annotated["dataset_kind"] = dataset_kind
    annotated["disclaimer"] = LAYER_A_DISCLAIMER
    annotated["architecture_metrics_location"] = "benchmarks/structured_memory (Layer B)"
    return annotated


def build_hotpot_layer_a_section(
    *,
    dataset_path: Path,
    modes: tuple[str, ...],
    top_k: int,
    limit: int = 0,
) -> dict[str, Any]:
    samples = load_hotpotqa_json_array(dataset_path)
    if limit > 0:
        samples = samples[:limit]
    suite = run_hotpotqa_benchmark(
        samples,
        retriever_modes=modes,
        top_k=top_k,
        dataset_id=f"hotpotqa::{dataset_path.stem}",
    )
    summary = summarize_hotpotqa_suite(samples, suite)
    payload = {
        "dataset": str(dataset_path),
        "samples": len(samples),
        "top_k": top_k,
        "modes": {
            mode_name: {
                "questions": report.questions,
                "evidence_hit_rate": report.evidence_hit_rate,
                "evidence_recall": report.evidence_recall,
                "avg_latency_ms": report.avg_latency_ms,
                "breakdown_by_type": {
                    case_type: bucket.model_dump()
                    for case_type, bucket in summary.modes[mode_name].breakdown_by_type.items()
                },
            }
            for mode_name, report in suite.modes.items()
        },
    }
    return annotate_external_summary(payload, dataset_kind="hotpotqa")


def build_longmem_layer_a_section(
    *,
    dataset_path: Path,
    modes: tuple[str, ...],
    top_k: int,
    limit: int = 0,
    granularity: str = "session",
) -> dict[str, Any]:
    samples = load_longmemeval_json(dataset_path)
    if limit > 0:
        samples = samples[:limit]
    suite = run_longmemeval_benchmark(
        samples,
        retriever_modes=modes,
        top_k=top_k,
        granularity=granularity,
        dataset_id=f"longmemeval::{dataset_path.stem}",
    )
    payload = {
        "dataset": str(dataset_path),
        "samples": len(samples),
        "top_k": top_k,
        "granularity": granularity,
        "modes": {
            mode_name: {
                "questions": report.questions,
                "recall_at_5": report.recall_at_5,
                "recall_at_10": report.recall_at_10,
                "ndcg_at_10": report.ndcg_at_10,
                "avg_latency_ms": report.avg_latency_ms,
            }
            for mode_name, report in suite.modes.items()
        },
    }
    return annotate_external_summary(payload, dataset_kind="longmemeval")


def build_layer_a_report(
    *,
    hotpot_dataset: Path,
    longmem_dataset: Path,
    slice_profile: str = "tiny",
    hotpot_modes: tuple[str, ...] = DEFAULT_HOTPOT_MODES,
    longmem_modes: tuple[str, ...] = DEFAULT_LONGMEM_MODES,
    top_k: int = 10,
    hotpot_limit: int | None = None,
    longmem_limit: int | None = None,
) -> dict[str, Any]:
    if slice_profile not in SLICE_PROFILES:
        raise ValueError(f"Unknown slice_profile: {slice_profile}")
    profile = SLICE_PROFILES[slice_profile]
    resolved_hotpot_limit = profile["hotpot_limit"] if hotpot_limit is None else hotpot_limit
    resolved_longmem_limit = profile["longmem_limit"] if longmem_limit is None else longmem_limit

    hotpot_section = build_hotpot_layer_a_section(
        dataset_path=hotpot_dataset,
        modes=hotpot_modes,
        top_k=top_k,
        limit=resolved_hotpot_limit,
    )
    longmem_section = build_longmem_layer_a_section(
        dataset_path=longmem_dataset,
        modes=longmem_modes,
        top_k=top_k,
        limit=resolved_longmem_limit,
    )
    return {
        "report_kind": "layer_a_positioning",
        "metric_layer": "A",
        "metric_scope": LAYER_A_METRIC_SCOPE,
        "disclaimer": LAYER_A_DISCLAIMER,
        "slice_profile": slice_profile,
        "layers": {
            "A": {
                "purpose": "external positioning and public retrieval sanity",
                "metrics": [
                    "evidence_hit_rate",
                    "evidence_recall",
                    "R@5",
                    "R@10",
                    "NDCG@10",
                    "avg_latency_ms",
                ],
            },
            "B": {
                "purpose": "architecture mechanism validation",
                "metric_scope": LAYER_B_METRIC_SCOPE,
                "location": "benchmarks/structured_memory",
                "metrics": [
                    "path_hit_rate",
                    "semantic_hit_rate",
                    "contradiction_hit_rate",
                    "activation_trace_hit_rate",
                    "route_hit_rate",
                ],
            },
            "C": {
                "purpose": "real-world transfer / private golden sets",
                "metric_scope": LAYER_C_METRIC_SCOPE,
                "location": "benchmarks/layer_c_minimal",
            },
        },
        "hotpotqa": hotpot_section,
        "longmemeval": longmem_section,
    }


def render_layer_a_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Layer A Positioning Report",
        "",
        f"Slice profile: `{report['slice_profile']}`",
        "",
        f"> {report['disclaimer']}",
        "",
        "## Metric Scope Map",
        "",
        "| Layer | Scope | Purpose |",
        "| --- | --- | --- |",
        f"| A | `{report['layers']['A']['purpose']}` | external positioning |",
        f"| B | `{report['layers']['B']['metric_scope']}` | {report['layers']['B']['purpose']} |",
        f"| C | `{report['layers']['C']['metric_scope']}` | {report['layers']['C']['purpose']} |",
        "",
        "## HotpotQA (evidence retrieval)",
        "",
        f"Dataset: `{report['hotpotqa']['dataset']}`",
        f"Samples: {report['hotpotqa']['samples']}",
        "",
        "| Mode | evidence_hit_rate | evidence_recall | avg_latency_ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode_name, metrics in report["hotpotqa"]["modes"].items():
        lines.append(
            f"| {mode_name} | {metrics['evidence_hit_rate']:.3f} | "
            f"{metrics['evidence_recall']:.3f} | {metrics['avg_latency_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## LongMemEval (session recall)",
            "",
            f"Dataset: `{report['longmemeval']['dataset']}`",
            f"Samples: {report['longmemeval']['samples']}",
            "",
            "| Mode | R@5 | R@10 | NDCG@10 | avg_latency_ms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode_name, metrics in report["longmemeval"]["modes"].items():
        lines.append(
            f"| {mode_name} | {metrics['recall_at_5']:.3f} | {metrics['recall_at_10']:.3f} | "
            f"{metrics['ndcg_at_10']:.3f} | {metrics['avg_latency_ms']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)
