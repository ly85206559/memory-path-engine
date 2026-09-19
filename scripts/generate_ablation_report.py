from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_engine.benchmarking.application.ablation import (
    ABLATION_MODES,
    DEFAULT_ABLATION_FIXTURES,
    aggregate_ablation_report,
    build_fixture_ablation_row,
)
from memory_engine.benchmarking.application.service import StructuredBenchmarkEvaluationService


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_ablation_report(
    *,
    dataset_names: tuple[str, ...],
    retriever_modes: tuple[str, ...],
    top_k: int = 3,
) -> dict:
    service = StructuredBenchmarkEvaluationService()
    per_fixture: list[dict] = []
    for dataset_name in dataset_names:
        dataset_path = repo_root() / "benchmarks" / "structured_memory" / dataset_name
        suite = service.run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=retriever_modes,
            top_k=top_k,
        )
        per_fixture.append(build_fixture_ablation_row(suite, fixture_name=dataset_name))
    return aggregate_ablation_report(per_fixture, modes=retriever_modes)


def render_markdown_report(report: dict) -> str:
    lines = [
        "# Ablation and Latency Report",
        "",
        f"Fixtures: {', '.join(report['fixtures'])}",
        f"Modes: {', '.join(report['modes'])}",
        "",
        "Ablation families follow `docs/evaluation.md`: remove structure, remove weights, remove path expansion.",
        "",
        "## Overall Latency by Mode",
        "",
        "| Mode | avg_ms | median_ms | p95_ms | max_ms | avg_evidence_hit_rate | fixtures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode_name, metrics in report["overall_modes"].items():
        lines.append(
            f"| {mode_name} | {metrics['avg_latency_ms']:.3f} | {metrics['median_latency_ms']:.3f} | "
            f"{metrics['p95_latency_ms']:.3f} | {metrics['max_latency_ms']:.3f} | "
            f"{metrics['avg_evidence_hit_rate']:.3f} | {metrics['fixture_count']} |"
        )

    lines.extend(
        [
            "",
            "## Ablation Family Summary",
            "",
            "| Family | baseline → full | primary metric | fixtures met | pass rate | avg primary Δ | avg latency Δ ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family in report["overall_families"]:
        lines.append(
            f"| {family['family_id']} | {family['baseline_mode']} → {family['full_mode']} | "
            f"{family['primary_metric']} | {family['fixtures_meeting_direction']}/{family['fixtures_evaluated']} | "
            f"{family['direction_pass_rate']:.3f} | {family['avg_primary_delta']:+.3f} | "
            f"{family['avg_latency_delta_ms']:+.3f} |"
        )

    lines.extend(["", "## Per Fixture", ""])
    for fixture in report["per_fixture"]:
        lines.append(f"### {fixture['fixture']}")
        lines.append("")
        lines.append(
            "| Mode | evidence_hit_rate | path_hit_rate | semantic_hit_rate | contradiction_hit_rate | avg_ms | median_ms | p95_ms |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mode_name, metrics in fixture["modes"].items():
            lines.append(
                f"| {mode_name} | {metrics['evidence_hit_rate']:.3f} | {metrics['path_hit_rate']:.3f} | "
                f"{metrics['semantic_hit_rate']:.3f} | {metrics['contradiction_hit_rate']:.3f} | "
                f"{metrics['avg_latency_ms']:.3f} | {metrics['median_latency_ms']:.3f} | "
                f"{metrics['p95_latency_ms']:.3f} |"
            )
        lines.append("")
        lines.append("Ablation deltas:")
        lines.append("")
        lines.append("| Family | primary Δ | evidence Δ | path Δ | semantic Δ | latency Δ ms | direction ok |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for family in fixture["ablation_families"]:
            lines.append(
                f"| {family['family_id']} | {family['delta']['primary_metric']:+.3f} | "
                f"{family['delta']['evidence_hit_rate']:+.3f} | {family['delta']['path_hit_rate']:+.3f} | "
                f"{family['delta']['semantic_hit_rate']:+.3f} | {family['delta']['avg_latency_ms']:+.3f} | "
                f"{'yes' if family['expected_direction_met'] else 'no'} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-format ablation matrix and latency summary report."
    )
    parser.add_argument(
        "--fixtures",
        default=",".join(DEFAULT_ABLATION_FIXTURES),
        help="Comma-separated structured benchmark fixture filenames.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(ABLATION_MODES),
        help="Comma-separated retriever modes to compare.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional markdown output path.",
    )
    args = parser.parse_args()

    report = build_ablation_report(
        dataset_names=_parse_csv(args.fixtures),
        retriever_modes=_parse_csv(args.modes),
        top_k=args.top_k,
    )
    markdown = render_markdown_report(report)

    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else (repo_root() / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output_path = (
            args.markdown_output
            if args.markdown_output.is_absolute()
            else (repo_root() / args.markdown_output).resolve()
        )
        markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_output_path.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
