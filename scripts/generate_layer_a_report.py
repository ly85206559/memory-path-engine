from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_engine.benchmarking.application.layer_a_report import (
    DEFAULT_HOTPOT_MODES,
    DEFAULT_LONGMEM_MODES,
    SLICE_PROFILES,
    build_layer_a_report,
    render_layer_a_markdown,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Layer A positioning report that keeps external metrics "
            "explicitly separated from Layer B architecture claims."
        )
    )
    parser.add_argument(
        "--slice-profile",
        choices=tuple(SLICE_PROFILES),
        default="tiny",
        help="tiny=checked-in fixtures; medium/full apply sample limits for downloaded files.",
    )
    parser.add_argument(
        "--hotpot-dataset",
        type=Path,
        default=repo_root() / "benchmarks/external/hotpotqa/hotpot_tiny_fixture.json",
    )
    parser.add_argument(
        "--longmem-dataset",
        type=Path,
        default=repo_root()
        / "benchmarks/external/longmemeval/longmemeval_tiny_fixture.json",
    )
    parser.add_argument("--hotpot-limit", type=int, default=None)
    parser.add_argument("--longmem-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--hotpot-modes",
        default=",".join(DEFAULT_HOTPOT_MODES),
    )
    parser.add_argument(
        "--longmem-modes",
        default=",".join(DEFAULT_LONGMEM_MODES),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    hotpot_dataset = args.hotpot_dataset
    if not hotpot_dataset.is_absolute():
        hotpot_dataset = (repo_root() / hotpot_dataset).resolve()
    longmem_dataset = args.longmem_dataset
    if not longmem_dataset.is_absolute():
        longmem_dataset = (repo_root() / longmem_dataset).resolve()

    report = build_layer_a_report(
        hotpot_dataset=hotpot_dataset,
        longmem_dataset=longmem_dataset,
        slice_profile=args.slice_profile,
        hotpot_modes=_parse_csv(args.hotpot_modes),
        longmem_modes=_parse_csv(args.longmem_modes),
        top_k=args.top_k,
        hotpot_limit=args.hotpot_limit,
        longmem_limit=args.longmem_limit,
    )
    markdown = render_layer_a_markdown(report)

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
