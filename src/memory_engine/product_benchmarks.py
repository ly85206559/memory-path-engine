from __future__ import annotations

"""Product-facing public benchmark helpers (Layer A KPI reports)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from memory_engine.benchmarking.adapters.longmemeval import (
    load_longmemeval_json,
    run_longmemeval_benchmark,
)
from memory_engine.benchmarking.application.layer_a_report import annotate_external_summary

DEFAULT_MODES: tuple[str, ...] = (
    "lexical_baseline",
    "embedding_baseline",
    "weighted_graph",
    "hybrid",
    "activation_spreading_v1",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_longmemeval_dataset() -> Path:
    return (
        repo_root()
        / "benchmarks"
        / "external"
        / "longmemeval"
        / "longmemeval_tiny_fixture.json"
    )


def default_baseline_dir() -> Path:
    return repo_root() / "benchmarks" / "external" / "longmemeval" / "baselines"


def run_longmemeval_baseline(
    *,
    dataset: Path | None = None,
    limit: int = 0,
    modes: Sequence[str] = DEFAULT_MODES,
    top_k: int = 10,
    output_dir: Path | None = None,
    label: str = "tiny",
    granularity: str = "session",
) -> dict[str, Any]:
    """
    Run LongMemEval retrieval baseline and write JSON + Markdown artifacts.

    This is the Product Layer A KPI runner: reproducible session/turn recall
    reports for tiny fixtures and full downloaded LongMemEval-S corpora.
    """
    dataset_path = dataset or default_longmemeval_dataset()
    if not dataset_path.is_absolute():
        dataset_path = (repo_root() / dataset_path).resolve()
    samples = load_longmemeval_json(dataset_path)
    if limit > 0:
        samples = samples[:limit]

    suite = run_longmemeval_benchmark(
        samples,
        retriever_modes=tuple(modes),
        top_k=top_k,
        granularity=granularity,
        dataset_id=f"longmemeval::{dataset_path.stem}::{label}",
    )
    summary = annotate_external_summary(
        {
            "product_kpi": True,
            "label": label,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "dataset": str(dataset_path),
            "samples": len(samples),
            "granularity": granularity,
            "top_k": top_k,
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
            "notes": [
                "Product KPI baseline for Layer A public recall.",
                "granularity=session aggregates each session; granularity=turn stores drawer-like turn units.",
                "hybrid mode blends lexical+embedding seeds then graph-expands.",
                "Full LongMemEval-S: download the cleaned file and run with --label full.",
                "Layer B path/contradiction metrics remain the architecture proof surface.",
            ],
        },
        dataset_kind="longmemeval",
    )

    out_dir = output_dir or default_baseline_dir()
    if not out_dir.is_absolute():
        out_dir = (repo_root() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"longmemeval_baseline_{label}.json"
    md_path = out_dir / f"longmemeval_baseline_{label}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_longmemeval_baseline_markdown(summary) + "\n", encoding="utf-8")
    return {
        "dataset": str(dataset_path),
        "samples": len(samples),
        "summary": summary,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


def format_longmemeval_baseline_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# LongMemEval baseline ({summary.get('label', 'unknown')})",
        "",
        f"- generated_at: `{summary.get('generated_at', '')}`",
        f"- dataset: `{summary.get('dataset', '')}`",
        f"- samples: **{summary.get('samples', 0)}**",
        f"- granularity: `{summary.get('granularity', 'session')}`",
        f"- metric_scope: `{summary.get('metric_scope', 'external_positioning')}`",
        f"- product_kpi: `{summary.get('product_kpi', False)}`",
        "",
        "| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    modes = summary.get("modes") or {}
    for mode_name, metrics in modes.items():
        lines.append(
            "| {mode} | {r5:.3f} | {r10:.3f} | {ndcg:.3f} | {latency:.3f} | {questions} |".format(
                mode=mode_name,
                r5=float(metrics.get("recall_at_5", 0.0)),
                r10=float(metrics.get("recall_at_10", 0.0)),
                ndcg=float(metrics.get("ndcg_at_10", 0.0)),
                latency=float(metrics.get("avg_latency_ms", 0.0)),
                questions=int(metrics.get("questions", 0)),
            )
        )
    notes = summary.get("notes") or []
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    return "\n".join(lines)
