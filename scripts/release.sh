#!/usr/bin/env bash
# Automate a versioned release: verify package → tag → push → watch publish.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="$(python3 - <<'PY'
import tomllib
from pathlib import Path
print(tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])
PY
)"
TAG="v${VERSION}"

echo "==> version ${VERSION} (tag ${TAG})"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree not clean; commit or stash first" >&2
  exit 1
fi

BRANCH="$(git branch --show-current)"
if [[ "${BRANCH}" != "master" && "${ALLOW_NON_MASTER:-}" != "1" ]]; then
  echo "refusing to release from branch '${BRANCH}' (set ALLOW_NON_MASTER=1 to override)" >&2
  exit 1
fi

echo "==> package build + twine check"
python3 -m pip install -q build twine
python3 scripts/check_package_build.py

if git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "tag ${TAG} already exists locally" >&2
  exit 1
fi
if git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  echo "tag ${TAG} already exists on origin" >&2
  exit 1
fi

echo "==> create annotated tag ${TAG}"
git tag -a "${TAG}" -m "Release ${TAG}"

echo "==> push tag to origin"
git push origin "${TAG}"

echo "==> waiting for publish workflow (if configured)"
if command -v gh >/dev/null 2>&1; then
  # Best-effort: watch the run created by this tag.
  sleep 3
  RUN_ID="$(gh run list --workflow=publish.yml --branch "${TAG}" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)"
  if [[ -n "${RUN_ID}" && "${RUN_ID}" != "null" ]]; then
    echo "watching run ${RUN_ID}"
    gh run watch "${RUN_ID}" --exit-status || {
      echo "publish workflow failed; check Trusted Publisher setup in docs/publish.md" >&2
      exit 1
    }
  else
    echo "no publish run visible yet; check Actions tab for tag ${TAG}"
  fi
fi

echo "==> done. Verify with: pip index versions memory-path-engine"
