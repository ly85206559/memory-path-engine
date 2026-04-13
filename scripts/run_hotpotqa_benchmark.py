from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_engine.benchmarking.adapters.hotpotqa import (
    load_hotpotqa_json_array,
    run_hotpotqa_benchmark,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dataset_path() -> Path:
    return repo_root() / "benchmarks" / "external" / "hotpotqa" / "hotpot_tiny_fixture.json"


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


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
        default="lexical_baseline,embedding_baseline",
        help="Comma-separated retriever modes to compare.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the full suite report JSON instead of a compact summary.",
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

    if args.pretty:
        print(suite.model_dump_json(indent=2))
        return

    print(f"dataset: {dataset_path}")
    print(f"samples: {len(samples)}")
    print(f"modes: {', '.join(suite.modes)}")
    print()
    for mode_name, report in suite.modes.items():
        print(
            json.dumps(
                {
                    "mode": mode_name,
                    "questions": report.questions,
                    "evidence_hit_rate": report.evidence_hit_rate,
                    "evidence_recall": report.evidence_recall,
                    "avg_latency_ms": report.avg_latency_ms,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
