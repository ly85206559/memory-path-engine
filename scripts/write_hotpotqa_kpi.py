#!/usr/bin/env python3
"""Write committed HotpotQA mid-slice KPI tables from a summary JSON."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def build_kpi_payload(
    summary: dict,
    *,
    label: str,
    embedding: str,
    notes: list[str],
) -> dict:
    modes: dict[str, dict] = {}
    for mode_name, report in summary.get("modes", {}).items():
        by_type = {
            case_type: {
                "questions": bucket["questions"],
                "evidence_hit_rate": bucket["evidence_hit_rate"],
                "evidence_recall": bucket["evidence_recall"],
                "avg_latency_ms": bucket["avg_latency_ms"],
            }
            for case_type, bucket in report.get("breakdown_by_type", {}).items()
        }
        modes[mode_name] = {
            "questions": report["questions"],
            "evidence_hit_rate": report["evidence_hit_rate"],
            "evidence_recall": report["evidence_recall"],
            "avg_latency_ms": report["avg_latency_ms"],
            "breakdown_by_type": by_type,
        }
    return {
        "label": label,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "source": "hotpot_dev_distractor_v1.json",
        "embedding": embedding,
        "samples": summary.get("samples"),
        "metric_scope": summary.get("metric_scope", "external_positioning"),
        "notes": notes,
        "modes": modes,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        f"# HotpotQA KPI ({payload['label']})",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- source: `{payload['source']}`",
        f"- embedding: `{payload['embedding']}`",
        f"- samples: `{payload['samples']}`",
        f"- metric_scope: `{payload['metric_scope']}`",
        "",
        "| Mode | evidence_hit | evidence_recall | avg_ms | questions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode_name, report in payload["modes"].items():
        lines.append(
            "| {mode} | {hit} | {recall} | {ms} | {n} |".format(
                mode=mode_name,
                hit=_fmt(report["evidence_hit_rate"]),
                recall=_fmt(report["evidence_recall"]),
                ms=_fmt(report["avg_latency_ms"]),
                n=report["questions"],
            )
        )
    lines.extend(["", "## By type", ""])
    for mode_name, report in payload["modes"].items():
        lines.append(f"### {mode_name}")
        lines.append("")
        lines.append("| Type | evidence_hit | questions |")
        lines.append("| --- | ---: | ---: |")
        for case_type, bucket in report.get("breakdown_by_type", {}).items():
            lines.append(
                f"| {case_type} | {_fmt(bucket['evidence_hit_rate'])} | {bucket['questions']} |"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--label", required=True, help="e.g. medium64 / medium64_fastembed")
    parser.add_argument("--embedding", default="ngram")
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional note line (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "benchmarks" / "external" / "hotpotqa" / "baselines",
    )
    args = parser.parse_args()

    summary_path = args.summary
    if not summary_path.is_absolute():
        summary_path = (repo_root() / summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    notes = list(args.note) or [
        "Public HotpotQA evidence-hit KPI (retrieval-only; not official EM/F1).",
        "Do not treat as Layer B architecture proof.",
    ]
    payload = build_kpi_payload(
        summary,
        label=args.label,
        embedding=args.embedding,
        notes=notes,
    )
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (repo_root() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hotpotqa_kpi_{args.label}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
