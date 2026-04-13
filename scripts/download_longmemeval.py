from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen

DEFAULT_LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_path() -> Path:
    return (
        repo_root()
        / "benchmarks"
        / "external"
        / "longmemeval"
        / "data"
        / "longmemeval_s_cleaned.json"
    )


def download_file(*, url: str, output_path: Path, force: bool, timeout: float) -> Path:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Use --force to overwrite or choose --output."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response, output_path.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the LongMemEval-S cleaned JSON for local retrieval benchmarking."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_LONGMEMEVAL_URL,
        help="Download URL. Defaults to the cleaned LongMemEval-S release on Hugging Face.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Where to save the dataset file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds.",
    )
    args = parser.parse_args()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = (repo_root() / output_path).resolve()

    path = download_file(
        url=args.url,
        output_path=output_path,
        force=args.force,
        timeout=args.timeout,
    )
    print(f"downloaded: {path}")
    print("next step:")
    print(
        f'python scripts/run_longmemeval_benchmark.py --dataset "{path}" --limit 50 '
        '--top-k 10 --modes "embedding_baseline,weighted_graph"'
    )


if __name__ == "__main__":
    main()
