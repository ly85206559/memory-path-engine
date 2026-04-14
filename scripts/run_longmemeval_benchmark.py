from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_engine.benchmarking.adapters.longmemeval import (
    load_longmemeval_json,
    run_longmemeval_benchmark,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dataset_path() -> Path:
    return repo_root() / "benchmarks" / "external" / "longmemeval" / "longmemeval_tiny_fixture.json"


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LongMemEval benchmark adapter on a local JSON file."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_default_dataset_path(),
        help="Path to a LongMemEval-style JSON array file. Defaults to the checked-in tiny fixture.",
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
        default="embedding_baseline,weighted_graph",
        help="Comma-separated retriever modes to compare.",
    )
    parser.add_argument(
        "--granularity",
        default="session",
        choices=("session",),
        help="Retrieval granularity. LongMemEval v0 currently supports session only.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the full suite report JSON instead of a compact summary.",
    )
    parser.add_argument(
        "--v1-recall-summary",
        action="store_true",
        help="Print per-case v1 palace metadata (spaces, routes, memory kinds) from the adapter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full suite report JSON.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = (repo_root() / dataset_path).resolve()

    samples = load_longmemeval_json(dataset_path)
    if args.limit > 0:
        samples = samples[: args.limit]

    suite = run_longmemeval_benchmark(
        samples,
        retriever_modes=_parse_modes(args.modes),
        top_k=args.top_k,
        granularity=args.granularity,
        dataset_id=f"longmemeval::{dataset_path.stem}",
    )

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = (repo_root() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(suite.model_dump_json(indent=2), encoding="utf-8")

    if args.pretty:
        print(suite.model_dump_json(indent=2))
        return

    if args.v1_recall_summary:
        print(f"dataset: {dataset_path}")
        print(f"samples: {len(samples)}")
        print("v1 recall summary (from LongMemEval adapter metadata)")
        for mode_name, mode_report in suite.modes.items():
            for case in mode_report.case_reports:
                meta = case.metadata
                print(
                    json.dumps(
                        {
                            "mode": mode_name,
                            "case_id": case.case_id,
                            "space_count": meta.get("space_count"),
                            "route_count": meta.get("route_count"),
                            "memory_kind_distribution": meta.get("memory_kind_distribution"),
                            "retrieved_items_top": case.retrieved_items[:5],
                        },
                        ensure_ascii=False,
                    )
                )
        return

    print(f"dataset: {dataset_path}")
    print(f"samples: {len(samples)}")
    print(f"granularity: {args.granularity}")
    print(f"modes: {', '.join(suite.modes)}")
    print()
    for mode_name, report in suite.modes.items():
        print(
            json.dumps(
                {
                    "mode": mode_name,
                    "questions": report.questions,
                    "recall_at_5": report.recall_at_5,
                    "recall_at_10": report.recall_at_10,
                    "ndcg_at_10": report.ndcg_at_10,
                    "avg_latency_ms": report.avg_latency_ms,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
