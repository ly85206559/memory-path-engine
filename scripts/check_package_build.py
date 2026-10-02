#!/usr/bin/env python3
"""Build sdist/wheel and run twine check (Product M8 dry-run)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    dist_dir = Path(tempfile.mkdtemp(prefix="mpe-dist-"))
    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(dist_dir)],
            cwd=ROOT,
            check=True,
        )
        artifacts = sorted(dist_dir.glob("*"))
        if len(artifacts) < 2:
            print("expected sdist + wheel", file=sys.stderr)
            return 1
        subprocess.run(
            [sys.executable, "-m", "twine", "check", *map(str, artifacts)],
            check=True,
        )
        for path in artifacts:
            print(f"ok: {path.name} ({path.stat().st_size} bytes)")
        return 0
    finally:
        shutil.rmtree(dist_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
