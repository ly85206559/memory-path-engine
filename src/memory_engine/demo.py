from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from memory_engine.ingest import ingest_contract_markdown, ingest_document
from memory_engine.retrieve import BaselineTopKRetriever, WeightedGraphRetriever
from memory_engine.schema import MemoryPath
from memory_engine.store import MemoryStore

_LINE_WIDTH = 72


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_runbook_store() -> MemoryStore:
    store = MemoryStore()
    runbooks_dir = repo_root() / "examples" / "runbook_pack" / "runbooks"
    for path in runbooks_dir.glob("*.md"):
        ingest_document(path, store, domain_pack="example_runbook_pack")
    return store


def build_contract_store() -> MemoryStore:
    store = MemoryStore()
    contracts_dir = repo_root() / "examples" / "contract_pack" / "contracts"
    for path in contracts_dir.glob("*.md"):
        ingest_contract_markdown(path, store)
    return store


def _rule(char: str = "=", width: int = _LINE_WIDTH) -> str:
    return char * width


def _section_label(title: str, char: str = "-", width: int = _LINE_WIDTH) -> str:
    """Single scan line: underline style with title left-aligned."""
    inner = f" {title} "
    if len(inner) >= width:
        return f"{char * width}\n{title}\n{char * width}"
    pad = width - len(inner)
    left = pad // 2
    right = pad - left
    return f"{char * left}{inner}{char * right}"


def _blank() -> str:
    return ""


def _indented_paragraph(text: str, margin: str = "  ") -> str:
    wrapped = textwrap.fill(
        text,
        width=max(20, _LINE_WIDTH - len(margin)),
        initial_indent=margin,
        subsequent_indent=margin,
    )
    return wrapped


def format_baseline_paths(paths: list[MemoryPath]) -> str:
    lines: list[str] = [
        _section_label("BASELINE  flat top-k (lexical overlap)", char="-"),
        _blank(),
    ]
    if not paths:
        lines.append("  (no hits above threshold)")
        return "\n".join(lines)
    for index, path in enumerate(paths, start=1):
        lines.append(f"  [{index}]")
        lines.append(_indented_paragraph(path.final_answer, margin="      "))
        lines.append(_blank())
    if lines[-1] == _blank():
        lines.pop()
    return "\n".join(lines)


def format_weighted_path(path: MemoryPath) -> str:
    lines: list[str] = [
        _section_label("PATH-AWARE  weighted graph retrieval", char="-"),
        _blank(),
        "  BEST ANSWER",
        _indented_paragraph(path.final_answer, margin="    "),
        _blank(),
        "  REPLAY PATH",
    ]
    for index, step in enumerate(path.steps, start=1):
        edge = step.via_edge_type or "seed"
        step_line = (
            f"    {index}. {step.node_id}  |  score={step.score:.3f}  |  via={edge}"
        )
        reason_wrapped = textwrap.fill(
            step.reason,
            width=max(24, _LINE_WIDTH - 6),
            initial_indent="       ",
            subsequent_indent="       ",
        )
        lines.append(step_line)
        lines.append(reason_wrapped)
        if index < len(path.steps):
            lines.append(_blank())
    return "\n".join(lines)


def _header_banner(scenario: str) -> str:
    title = "Memory Path Engine  |  demo"
    sub = f"scenario: {scenario}"
    return "\n".join(
        [
            _rule("="),
            f"  {title}",
            f"  {sub}",
            _rule("="),
        ]
    )


def _query_block(query: str) -> str:
    return "\n".join(
        [
            _section_label("QUERY", char="-"),
            _blank(),
            _indented_paragraph(query, margin="  "),
            _blank(),
        ]
    )


def run_runbook_demo() -> str:
    query = "What should we do if rollback does not recover the API after a deployment incident?"
    result = WeightedGraphRetriever(build_runbook_store()).search(query, top_k=3)
    return "\n".join(
        [
            _header_banner("runbook"),
            _query_block(query),
            format_weighted_path(result.best_path()),
            _blank(),
            _rule("="),
        ]
    )


def run_contract_demo() -> str:
    query = "What happens if delivery is late and the supplier does not cure in time?"
    store = build_contract_store()
    baseline = BaselineTopKRetriever(store).search(query, top_k=2)
    weighted = WeightedGraphRetriever(store).search(query, top_k=2)

    return "\n".join(
        [
            _header_banner("contract"),
            _query_block(query),
            format_baseline_paths(baseline.paths),
            _blank(),
            format_weighted_path(weighted.best_path()),
            _blank(),
            _rule("="),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a repository demo for Memory Path Engine."
    )
    parser.add_argument(
        "--scenario",
        choices=("runbook", "contract"),
        default="runbook",
        help="Which bundled demo scenario to run.",
    )
    args = parser.parse_args()

    if args.scenario == "contract":
        print(run_contract_demo())
        return

    print(run_runbook_demo())


if __name__ == "__main__":
    main()
