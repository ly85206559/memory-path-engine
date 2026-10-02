#!/usr/bin/env bash
# Print the exact PyPI Trusted Publisher fields for this repo (copy/paste).
set -euo pipefail

OWNER="$(gh repo view --json owner --jq .owner.login 2>/dev/null || echo ly85206559)"
REPO="$(gh repo view --json name --jq .name 2>/dev/null || echo memory-path-engine)"

cat <<EOF
Add a pending Trusted Publisher on https://pypi.org/manage/account/publishing/

  PyPI project name : memory-path-engine
  Owner             : ${OWNER}
  Repository        : ${REPO}
  Workflow name     : publish.yml
  Environment name  : (leave blank)

Then run:

  bash scripts/release.sh

EOF
