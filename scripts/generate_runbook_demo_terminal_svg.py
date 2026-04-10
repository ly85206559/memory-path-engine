#!/usr/bin/env python3
"""Write docs/assets/runbook-demo-terminal.svg from real `demo --scenario runbook` stdout."""

from __future__ import annotations

import subprocess
import sys
import xml.sax.saxutils as xml_esc
from pathlib import Path

OUT_NAME = "runbook-demo-terminal.svg"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _capture_runbook_demo(root: Path) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "memory_engine.demo", "--scenario", "runbook"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(proc.returncode)
    text = proc.stdout.replace("\r\n", "\n").rstrip("\n")
    return text.split("\n")


def _build_svg(lines: list[str]) -> str:
    pad = 16
    fs = 12
    line_h = 14
    # Approximate monospace advance for Latin / ASCII demo output (slightly loose for SVG viewers).
    char_w = fs * 0.72
    max_chars = max((len(s) for s in lines), default=0)
    w = int(max_chars * char_w + 2 * pad)
    h = int(len(lines) * line_h + 2 * pad)

    tspans = []
    y0 = pad + fs
    for i, line in enumerate(lines):
        esc = xml_esc.escape(line)
        dy = 0 if i == 0 else line_h
        tspans.append(f'<tspan x="{pad}" dy="{dy}">{esc}</tspan>')

    body = "\n    ".join(tspans)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <title>memory_engine.demo --scenario runbook (captured stdout)</title>
  <desc>Terminal-style rendering of actual demo output; regenerate with scripts/generate_runbook_demo_terminal_svg.py</desc>
  <rect width="100%" height="100%" fill="#0d0d0d"/>
  <text xml:space="preserve" x="0" y="{y0}" font-family="ui-monospace, 'Cascadia Mono', 'Cascadia Code', Consolas, monospace" font-size="{fs}px" fill="#e8e8e8">{body}</text>
</svg>
'''


def main() -> None:
    root = _repo_root()
    out = root / "docs" / "assets" / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = _capture_runbook_demo(root)
    out.write_text(_build_svg(lines), encoding="utf-8")
    print(f"Wrote {out.relative_to(root)}")


if __name__ == "__main__":
    main()
