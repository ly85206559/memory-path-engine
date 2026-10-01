from __future__ import annotations

"""
``mpe`` command-line entry (Product M1).

Commands:
  init / ingest / search / path / status / bench longmemeval
"""

import argparse
import json
import sys
from pathlib import Path

from memory_engine.api import recall_from_store, reason_from_recall
from memory_engine.palace_workspace import (
    init_palace,
    open_palace,
    resolve_palace_root,
)
from memory_engine.product_benchmarks import (
    format_longmemeval_baseline_markdown,
    run_longmemeval_baseline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mpe",
        description=(
            "Memory Path Engine CLI — local palace with replayable evidence paths."
        ),
    )
    parser.add_argument(
        "--palace",
        type=Path,
        default=None,
        help="Palace directory (default: ./.mpe or $MPE_PALACE).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create a local palace directory.")
    init_p.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Palace path (default: ./.mpe).",
    )
    init_p.add_argument("--name", default=None, help="Human-readable palace name.")
    init_p.add_argument(
        "--pack",
        default="example_runbook_pack",
        help="Default domain pack for ingest.",
    )
    init_p.add_argument(
        "--mode",
        default="weighted_graph",
        help="Default retriever mode for search/path.",
    )
    init_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate config/store even if the palace already exists.",
    )

    ingest_p = sub.add_parser("ingest", help="Ingest files or a directory into the palace.")
    ingest_p.add_argument("targets", nargs="+", type=Path, help="Files or directories.")
    ingest_p.add_argument(
        "--pack",
        default=None,
        help="Domain pack name (default: palace config).",
    )
    ingest_p.add_argument(
        "--glob",
        default="*.md",
        dest="glob_pattern",
        help="Glob used when a target is a directory (default: *.md).",
    )

    search_p = sub.add_parser(
        "search",
        help="Retrieve with answer + path hops (default product search).",
    )
    _add_query_args(search_p)

    path_p = sub.add_parser(
        "path",
        help="Same as search, emphasizing hop citations.",
    )
    _add_query_args(path_p)

    sub.add_parser("status", help="Show palace location and graph counts.")

    bench_p = sub.add_parser("bench", help="Product benchmark entry points.")
    bench_sub = bench_p.add_subparsers(dest="bench_command", required=True)
    lme = bench_sub.add_parser(
        "longmemeval",
        help="Run LongMemEval retrieval baseline and write JSON/Markdown reports.",
    )
    lme.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="LongMemEval JSON file (default: checked-in tiny fixture).",
    )
    lme.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples (0 = all in file).",
    )
    lme.add_argument(
        "--modes",
        default="lexical_baseline,embedding_baseline,weighted_graph,activation_spreading_v1",
        help="Comma-separated retriever modes.",
    )
    lme.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retriever top_k.",
    )
    lme.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for baseline JSON/MD (default: benchmarks/external/longmemeval/baselines).",
    )
    lme.add_argument(
        "--label",
        default="tiny",
        help="Baseline label used in output filenames (e.g. tiny, medium, full).",
    )
    return parser


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", help="Natural-language query.")
    parser.add_argument(
        "--mode",
        default=None,
        help="Retriever mode (default: palace config).",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of paths.")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON.",
    )


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "ingest":
            return _cmd_ingest(args)
        if args.command in {"search", "path"}:
            return _cmd_search(args, emphasize_path=args.command == "path")
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "bench" and args.bench_command == "longmemeval":
            return _cmd_bench_longmemeval(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command}")
    return 2


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


def _cmd_init(args: argparse.Namespace) -> int:
    root = resolve_palace_root(args.path or args.palace)
    workspace = init_palace(
        root,
        name=args.name,
        default_domain_pack=args.pack,
        default_retriever_mode=args.mode,
        overwrite=args.overwrite,
    )
    print(f"palace ready: {workspace.root}")
    print(f"config: {workspace.config_path}")
    print(f"store:  {workspace.store_path}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    workspace = open_palace(args.palace)
    result = workspace.ingest_paths(
        list(args.targets),
        domain_pack=args.pack,
        glob_pattern=args.glob_pattern,
    )
    print(f"palace: {workspace.root}")
    print(f"domain_pack: {result['domain_pack']}")
    print(f"ingested_files: {len(result['ingested_files'])}")
    for path in result["ingested_files"]:
        print(f"  - {path}")
    print(f"nodes: {result['nodes']}")
    print(f"edges: {result['edges']}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    workspace = open_palace(args.palace)
    status = workspace.status()
    for key, value in status.items():
        print(f"{key}: {value}")
    return 0


def _cmd_search(args: argparse.Namespace, *, emphasize_path: bool) -> int:
    workspace = open_palace(args.palace)
    store = workspace.load_store()
    if not store.nodes():
        print("error: palace is empty — run mpe ingest first", file=sys.stderr)
        return 2
    mode = args.mode or workspace.config.default_retriever_mode
    unified = recall_from_store(
        store,
        args.query,
        retriever_mode=mode,
        top_k=args.top_k,
        project_palace=False,
    )
    reasoned = reason_from_recall(args.query, unified, store=store)
    workspace.save_store(store)
    payload = {
        "query": args.query,
        "retriever_mode": mode,
        "answer": reasoned.answer or unified.best_answer,
        "confidence": reasoned.confidence,
        "cited_node_ids": list(reasoned.cited_node_ids),
        "path_edge_types": list(reasoned.path_edge_types),
        "hop_explanations": list(reasoned.hop_explanations),
        "paths": [
            {
                "final_score": path.final_score,
                "final_answer": path.final_answer,
                "steps": [
                    {
                        "node_id": step.node_id,
                        "score": step.score,
                        "via_edge_type": step.via_edge_type,
                        "reason": step.reason,
                    }
                    for step in path.steps
                ],
            }
            for path in (unified.legacy.paths if unified.legacy else [])
        ],
    }
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"palace: {workspace.root}")
    print(f"mode: {mode}")
    print(f"query: {args.query}")
    print()
    print("ANSWER")
    print(payload["answer"] or "(empty)")
    print()
    print("PATH" if emphasize_path else "BEST PATH")
    if reasoned.hop_explanations:
        for line in reasoned.hop_explanations:
            print(f"  {line}")
    elif payload["paths"]:
        for index, step in enumerate(payload["paths"][0]["steps"]):
            via = step["via_edge_type"] or "seed"
            print(f"  hop {index}: {step['node_id']} via={via} score={step['score']:.3f}")
    else:
        print("  (no path)")
    if reasoned.path_edge_types:
        print(f"edges: {', '.join(reasoned.path_edge_types)}")
    print(f"confidence: {reasoned.confidence:.3f}")
    return 0


def _cmd_bench_longmemeval(args: argparse.Namespace) -> int:
    result = run_longmemeval_baseline(
        dataset=args.dataset,
        limit=args.limit,
        modes=tuple(part.strip() for part in args.modes.split(",") if part.strip()),
        top_k=args.top_k,
        output_dir=args.output_dir,
        label=args.label,
    )
    print(f"dataset: {result['dataset']}")
    print(f"samples: {result['samples']}")
    print(f"json: {result['json_path']}")
    print(f"markdown: {result['markdown_path']}")
    print()
    print(format_longmemeval_baseline_markdown(result["summary"]))
    return 0


if __name__ == "__main__":
    main()
