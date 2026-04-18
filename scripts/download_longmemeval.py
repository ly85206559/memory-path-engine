from __future__ import annotations

import argparse
import shutil
import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DEFAULT_LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
)
DEFAULT_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0


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
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="How many times to retry transient download failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Initial backoff between retries. Doubles on each attempt.",
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
        retries=args.retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    print(f"downloaded: {path}")
    print("next step:")
    print(
        f'python scripts/run_longmemeval_benchmark.py --dataset "{path}" --limit 50 '
        '--top-k 10 --modes "embedding_baseline,weighted_graph"'
    )


if __name__ == "__main__":
    main()
