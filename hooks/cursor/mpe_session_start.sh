#!/usr/bin/env bash
# Cursor / Claude session-start recall hint for Memory Path Engine.
set -euo pipefail
PALACE="${MPE_PALACE:-.mpe}"
QUERY="${1:-What were we working on last?}"

if ! command -v mpe >/dev/null 2>&1; then
  python3 -m memory_engine.cli --palace "$PALACE" status 2>/dev/null || true
  echo "[mpe] tip: install with pip install -e . so 'mpe' is on PATH"
  exit 0
fi

mpe --palace "$PALACE" status || true
echo
echo "[mpe] session recall"
mpe --palace "$PALACE" search "$QUERY" --mode hybrid --top-k 2 2>/dev/null || \
  mpe --palace "$PALACE" search "$QUERY" --top-k 2 2>/dev/null || \
  echo "[mpe] palace empty or search failed — ingest docs first"
