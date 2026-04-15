from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_engine.benchmarking.application.service import StructuredBenchmarkEvaluationService

DEFAULT_FIXTURES = (
    "spatial_recall_benchmark.json",
    "route_replay_benchmark.json",
    "state_transition_benchmark.json",
    "encoding_recall_benchmark.json",
    "consolidation_gain_benchmark.json",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_layer_b_report(*, dataset_names: tuple[str, ...], retriever_modes: tuple[str, ...]) -> dict:
    service = StructuredBenchmarkEvaluationService()
    report_rows: list[dict] = []
    aggregate: dict[str, dict[str, list[float]]] = {
        mode: {
            "path_hit_rate": [],
            "route_hit_rate": [],
            "space_hit_rate": [],
            "lifecycle_hit_rate": [],
            "activation_snapshot_hit_rate": [],
        }
        for mode in retriever_modes
    }

    for dataset_name in dataset_names:
        dataset_path = repo_root() / "benchmarks" / "structured_memory" / dataset_name
        suite = service.run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=retriever_modes,
            top_k=3,
        )
        row = {
            "dataset_id": suite.dataset_id,
            "datasets": dataset_name,
            "modes": {},
        }
        for mode_name, mode_summary in suite.comparison.mode_summary.items():
            mode_metrics = {
                "path_hit_rate": mode_summary.path_hit_rate,
                "route_hit_rate": mode_summary.route_hit_rate,
                "space_hit_rate": mode_summary.space_hit_rate,
                "lifecycle_hit_rate": mode_summary.lifecycle_hit_rate,
                "activation_snapshot_hit_rate": mode_summary.activation_snapshot_hit_rate,
            }
            row["modes"][mode_name] = mode_metrics
            for metric_name, metric_value in mode_metrics.items():
                aggregate[mode_name][metric_name].append(metric_value)
        report_rows.append(row)

    overall = {
        mode_name: {
            metric_name: round(sum(values) / len(values), 6) if values else 0.0
            for metric_name, values in metric_groups.items()
        }
        for mode_name, metric_groups in aggregate.items()
    }
    return {
        "fixtures": list(dataset_names),
        "modes": list(retriever_modes),
        "overall": overall,
        "per_fixture": report_rows,
    }


def render_markdown_report(report: dict) -> str:
    lines = [
        "# Layer B Report",
        "",
        f"Fixtures: {', '.join(report['fixtures'])}",
        f"Modes: {', '.join(report['modes'])}",
        "",
        "## Overall",
        "",
        "| Mode | path_hit_rate | route_hit_rate | space_hit_rate | lifecycle_hit_rate | activation_snapshot_hit_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode_name, metrics in report["overall"].items():
        lines.append(
            f"| {mode_name} | {metrics['path_hit_rate']:.3f} | {metrics['route_hit_rate']:.3f} | "
            f"{metrics['space_hit_rate']:.3f} | {metrics['lifecycle_hit_rate']:.3f} | "
            f"{metrics['activation_snapshot_hit_rate']:.3f} |"
        )

    lines.extend(["", "## Per Fixture", ""])
    for fixture in report["per_fixture"]:
        lines.append(f"### {fixture['datasets']}")
        lines.append("")
        lines.append(
            "| Mode | path_hit_rate | route_hit_rate | space_hit_rate | lifecycle_hit_rate | activation_snapshot_hit_rate |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for mode_name, metrics in fixture["modes"].items():
            lines.append(
                f"| {mode_name} | {metrics['path_hit_rate']:.3f} | {metrics['route_hit_rate']:.3f} | "
                f"{metrics['space_hit_rate']:.3f} | {metrics['lifecycle_hit_rate']:.3f} | "
                f"{metrics['activation_snapshot_hit_rate']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed-format Layer B benchmark report.")
    parser.add_argument(
        "--fixtures",
        default=",".join(DEFAULT_FIXTURES),
        help="Comma-separated structured benchmark fixture filenames.",
    )
    parser.add_argument(
        "--modes",
        default="weighted_graph,activation_spreading_v1",
        help="Comma-separated retriever modes to report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional markdown output path.",
    )
    args = parser.parse_args()

    report = build_layer_b_report(
        dataset_names=_parse_csv(args.fixtures),
        retriever_modes=_parse_csv(args.modes),
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
