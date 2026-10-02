from __future__ import annotations

import argparse
import json
import shutil
import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DEFAULT_HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
DEFAULT_HF_DATASET = "hotpotqa/hotpot_qa"
DEFAULT_HF_CONFIG = "distractor"
DEFAULT_HF_SPLIT = "validation"
DEFAULT_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_path() -> Path:
    return (
        repo_root()
        / "benchmarks"
        / "external"
        / "hotpotqa"
        / "data"
        / "hotpot_dev_distractor_v1.json"
    )


def _should_retry_download(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return 500 <= exc.code < 600
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, socket.gaierror):
            return True
        if isinstance(reason, TimeoutError):
            return True
        if isinstance(reason, OSError):
            return True
        return "temporary failure" in str(reason).lower()
    return isinstance(exc, (TimeoutError, socket.gaierror, OSError))


def download_file(
    *,
    url: str,
    output_path: Path,
    force: bool,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> Path:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Use --force to overwrite or choose --output."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with urlopen(url, timeout=timeout) as response, temp_output_path.open("wb") as fh:
                shutil.copyfileobj(response, fh)
            temp_output_path.replace(output_path)
            return output_path
        except Exception as exc:
            if temp_output_path.exists():
                temp_output_path.unlink()
            if attempt >= retries or not _should_retry_download(exc):
                raise
            wait_seconds = retry_backoff_seconds * (2**attempt)
            print(
                f"download attempt {attempt + 1} failed ({exc}); retrying in {wait_seconds:.1f}s..."
            )
            time.sleep(wait_seconds)
    return output_path


def hf_row_to_hotpot_sample(row: dict) -> dict:
    """Convert a HuggingFace ``hotpotqa/hotpot_qa`` row to official HotpotQA JSON shape."""
    context = [
        [title, list(sentences)]
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
    ]
    supporting_facts = [
        [title, int(sent_id)]
        for title, sent_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
        )
    ]
    return {
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "context": context,
        "supporting_facts": supporting_facts,
    }


def export_from_huggingface(
    *,
    output_path: Path,
    force: bool,
    dataset_name: str = DEFAULT_HF_DATASET,
    config_name: str = DEFAULT_HF_CONFIG,
    split: str = DEFAULT_HF_SPLIT,
) -> Path:
    """Export HotpotQA distractor validation via HuggingFace Hub (CMU mirror fallback)."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Use --force to overwrite or choose --output."
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "HuggingFace fallback requires the 'datasets' package. "
            "Install with: pip install datasets"
        ) from exc

    print(f"exporting HotpotQA from HuggingFace ({dataset_name}/{config_name}/{split})...")
    ds = load_dataset(dataset_name, config_name, split=split)
    samples = [hf_row_to_hotpot_sample(row) for row in ds]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".part")
    temp_output_path.write_text(json.dumps(samples), encoding="utf-8")
    temp_output_path.replace(output_path)
    print(f"wrote {len(samples)} samples")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download HotpotQA dev distractor JSON for local benchmarking. "
            "Tries the official CMU URL first, then HuggingFace Hub fallback."
        )
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_HOTPOTQA_URL,
        help="Primary download URL (official HotpotQA / CMU).",
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
        help="Network timeout in seconds for the primary URL.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="How many times to retry transient primary-URL failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Initial backoff between retries. Doubles on each attempt.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "url", "huggingface"),
        default="auto",
        help="Download source. auto = URL then HuggingFace fallback.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help="HuggingFace dataset id used for fallback export.",
    )
    parser.add_argument(
        "--hf-config",
        default=DEFAULT_HF_CONFIG,
        help="HuggingFace config name (distractor).",
    )
    parser.add_argument(
        "--hf-split",
        default=DEFAULT_HF_SPLIT,
        help="HuggingFace split name (validation ≈ official dev distractor).",
    )
    args = parser.parse_args()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = (repo_root() / output_path).resolve()

    path: Path | None = None
    if args.source in {"auto", "url"}:
        try:
            path = download_file(
                url=args.url,
                output_path=output_path,
                force=args.force,
                timeout=args.timeout,
                retries=args.retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
            print(f"downloaded (url): {path}")
        except Exception as exc:
            if args.source == "url":
                raise
            print(f"primary URL failed ({exc}); falling back to HuggingFace…")

    if path is None:
        path = export_from_huggingface(
            output_path=output_path,
            force=args.force or True,
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            split=args.hf_split,
        )
        print(f"downloaded (huggingface): {path}")

    print("next step:")
    print(
        f'python scripts/run_hotpotqa_benchmark.py --dataset "{path}" --limit 64 --top-k 10 '
        "--modes lexical_baseline,embedding_baseline,weighted_graph,hybrid,activation_spreading_v1"
    )


if __name__ == "__main__":
    main()
