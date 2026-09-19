from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_engine.benchmarking.application.service import StructuredBenchmarkEvaluationService

DEFAULT_LAYER_C_FIXTURES: tuple[str, ...] = (
    "layer_c_contract_benchmark.json",
    "layer_c_runbook_benchmark.json",
)

DEFAULT_MODES: tuple[str, ...] = (
    "lexical_baseline",
    "embedding_baseline",
    "structure_only",
    "weighted_graph",
    "activation_spreading_v1",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def layer_c_root() -> Path:
    return repo_root() / "benchmarks" / "layer_c_minimal"


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_layer_c_report(
    *,
    fixtures: tuple[str, ...],
    modes: tuple[str, ...],
    top_k: int = 3,
) -> dict:
    service = StructuredBenchmarkEvaluationService()
    per_fixture: list[dict] = []
    for fixture_name in fixtures:
        dataset_path = layer_c_root() / fixture_name
        suite = service.run_suite_from_dataset_path(
            dataset_path=dataset_path,
            retriever_modes=modes,
            top_k=top_k,
        )
        mode_rows = {}
        for mode_name, mode_report in suite.modes.items():
            summary = suite.comparison.mode_summary[mode_name]
            mode_rows[mode_name] = {
                "evidence_hit_rate": mode_report.evidence_hit_rate,
                "evidence_recall": mode_report.evidence_recall,
                "path_hit_rate": summary.path_hit_rate,
                "semantic_hit_rate": summary.semantic_hit_rate,
                "contradiction_hit_rate": summary.contradiction_hit_rate,
                "avg_latency_ms": mode_report.avg_latency_ms,
                "questions": mode_report.questions,
            }
        per_fixture.append(
            {
                "fixture": fixture_name,
                "dataset_id": suite.dataset_id,
                "modes": mode_rows,
            }
        )
    return {
        "report_kind": "layer_c_transfer",
        "metric_layer": "C",
        "metric_scope": "real_world_transfer",
        "disclaimer": (
            "Layer C uses noise-realistic public stand-in documents. "
            "Replace documents/ and private annotations with internal golden sets "
            "before making production transfer claims."
        ),
        "fixtures": list(fixtures),
        "modes": list(modes),
        "per_fixture": per_fixture,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Layer C Transfer Report",
        "",
        f"Fixtures: {', '.join(report['fixtures'])}",
        f"Modes: {', '.join(report['modes'])}",
        "",
        f"> {report['disclaimer']}",
        "",
    ]
    for fixture in report["per_fixture"]:
        lines.append(f"## {fixture['fixture']}")
        lines.append("")
        lines.append(
            "| Mode | evidence_hit_rate | path_hit_rate | semantic_hit_rate | contradiction_hit_rate | avg_latency_ms |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for mode_name, metrics in fixture["modes"].items():
            lines.append(
                f"| {mode_name} | {metrics['evidence_hit_rate']:.3f} | {metrics['path_hit_rate']:.3f} | "
                f"{metrics['semantic_hit_rate']:.3f} | {metrics['contradiction_hit_rate']:.3f} | "
                f"{metrics['avg_latency_ms']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Layer C minimal benchmarks with the same structured runner contract as Layer B."
    )
    parser.add_argument(
        "--fixtures",
        default=",".join(DEFAULT_LAYER_C_FIXTURES),
        help="Comma-separated Layer C fixture filenames under benchmarks/layer_c_minimal.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    report = build_layer_c_report(
        fixtures=_parse_csv(args.fixtures),
        modes=_parse_csv(args.modes),
        top_k=args.top_k,
    )
    markdown = render_markdown(report)

    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else (repo_root() / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output is not None:
        markdown_path = (
            args.markdown_output
            if args.markdown_output.is_absolute()
            else (repo_root() / args.markdown_output).resolve()
        )
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
