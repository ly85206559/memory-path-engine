#!/usr/bin/env bash
# Save a session memo into the local palace (stdin text or file argument).
set -euo pipefail
PALACE="${MPE_PALACE:-.mpe}"
TITLE="${MPE_MEMO_TITLE:-cursor-session}"
SOURCE="${MPE_MEMO_SOURCE:-cursor-hook}"

if [[ $# -gt 0 ]]; then
  TEXT="$(cat -- "$1")"
else
  TEXT="$(cat)"
fi

if [[ -z "${TEXT//[[:space:]]/}" ]]; then
  echo "[mpe] empty memo — nothing to save"
  exit 0
fi

# Keep hook payloads bounded so accidental huge transcripts do not inflate the palace.
TEXT="$(printf '%s' "$TEXT" | head -c 12000)"

if command -v mpe >/dev/null 2>&1; then
  printf '%s' "$TEXT" | mpe --palace "$PALACE" memo --title "$TITLE" --source "$SOURCE"
else
  printf '%s' "$TEXT" | python3 -m memory_engine.cli --palace "$PALACE" memo --title "$TITLE" --source "$SOURCE"
fi
