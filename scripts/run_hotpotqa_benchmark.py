from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_engine.benchmarking.adapters.hotpotqa import (
    load_hotpotqa_json_array,
    run_hotpotqa_benchmark,
    summarize_hotpotqa_suite,
)
from memory_engine.benchmarking.application.layer_a_report import annotate_external_summary


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dataset_path() -> Path:
    return repo_root() / "benchmarks" / "external" / "hotpotqa" / "hotpot_tiny_fixture.json"


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_hotpot_summary_payload(samples, suite, summary, *, dataset_path: Path) -> dict:
    payload = {
        "dataset": str(dataset_path),
        "samples": len(samples),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the HotpotQA benchmark adapter on a local JSON file."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_default_dataset_path(),
        help="Path to a HotpotQA-style JSON array file. Defaults to the checked-in tiny fixture.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of samples to evaluate. 0 means all samples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retriever top_k passed to each mode.",
    )
    parser.add_argument(
        "--modes",
        default="lexical_baseline,embedding_baseline,weighted_graph,activation_spreading_v1",
        help="Comma-separated retriever modes to compare.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the full suite report JSON instead of a compact summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full suite report JSON.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to write a compact HotpotQA summary JSON.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = (repo_root() / dataset_path).resolve()

    samples = load_hotpotqa_json_array(dataset_path)
    if args.limit > 0:
        samples = samples[: args.limit]

    suite = run_hotpotqa_benchmark(
        samples,
        retriever_modes=_parse_modes(args.modes),
        top_k=args.top_k,
        dataset_id=f"hotpotqa::{dataset_path.stem}",
    )
    summary = summarize_hotpotqa_suite(samples, suite)
    summary_payload = build_hotpot_summary_payload(
        samples,
        suite,
        summary,
        dataset_path=dataset_path,
    )

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = (repo_root() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(suite.model_dump_json(indent=2), encoding="utf-8")

    if args.summary_output is not None:
        summary_output_path = args.summary_output
        if not summary_output_path.is_absolute():
            summary_output_path = (repo_root() / summary_output_path).resolve()
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.pretty:
        print(suite.model_dump_json(indent=2))
        return

    print(f"dataset: {dataset_path}")
    print(f"samples: {len(samples)}")
    print(f"metric_scope: {summary_payload['metric_scope']}")
    print(f"modes: {', '.join(suite.modes)}")
    print()
    for mode_name, report in suite.modes.items():
        by_type = {
            case_type: bucket.model_dump()
            for case_type, bucket in summary.modes[mode_name].breakdown_by_type.items()
        }
        print(
            json.dumps(
                {
                    "mode": mode_name,
                    "questions": report.questions,
                    "evidence_hit_rate": report.evidence_hit_rate,
                    "evidence_recall": report.evidence_recall,
                    "avg_latency_ms": report.avg_latency_ms,
                    "breakdown_by_type": by_type,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
