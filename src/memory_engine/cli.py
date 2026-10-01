from __future__ import annotations

"""
``mpe`` command-line entry (Product M1 + M2).

Commands:
  init / ingest / memo / search / path / reinforce / status
  backup / repair / doctor
  mcp / hooks install / bench longmemeval
"""

import argparse
import json
import sys
from pathlib import Path

from memory_engine.hooks_install import install_hooks
from memory_engine.palace_workspace import init_palace, resolve_palace_root
from memory_engine.product_benchmarks import (
    format_longmemeval_baseline_markdown,
    run_longmemeval_baseline,
)
from memory_engine.product_service import (
    palace_ingest,
    palace_ingest_memo,
    palace_reinforce,
    palace_search,
    palace_status,
)
from memory_engine.palace_ops import backup_palace, doctor_report, repair_palace


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
        help="Default retriever mode for search/path (try: hybrid).",
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

    memo_p = sub.add_parser("memo", help="Append a freeform memo (stdin or --text).")
    memo_p.add_argument("--text", default=None, help="Memo body (default: read stdin).")
    memo_p.add_argument("--title", default=None, help="Optional title prefix.")
    memo_p.add_argument("--source", default="memo", help="Source label stored on the node.")

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

    reinforce_p = sub.add_parser(
        "reinforce",
        help="Search then apply online reinforce/forget (default policy: mild).",
    )
    _add_query_args(reinforce_p)
    reinforce_p.add_argument(
        "--policy",
        default="mild",
        help="Forgetting policy: mild | aggressive | default.",
    )

    sub.add_parser("status", help="Show palace location and graph counts.")
    sub.add_parser("mcp", help="Run the stdio MCP server for agent clients.")
    sub.add_parser(
        "doctor",
        help="Check install + palace health (Product M4).",
    )

    backup_p = sub.add_parser("backup", help="Archive palace config + SQLite store.")
    backup_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Archive path or directory (default: ./mpe-backups/<name>-<ts>.tar.gz).",
    )

    repair_p = sub.add_parser(
        "repair",
        help="Validate/repair palace SQLite (quarantine corrupt stores).",
    )
    repair_p.add_argument(
        "--rebuild-empty",
        action="store_true",
        help="Force replace store with an empty graph (keeps a local backup copy).",
    )

    hooks_p = sub.add_parser("hooks", help="Install Cursor/Claude hook templates.")
    hooks_sub = hooks_p.add_subparsers(dest="hooks_command", required=True)
    hooks_install = hooks_sub.add_parser(
        "install",
        help="Copy hook scripts + MCP snippet into .cursor/mpe-hooks/.",
    )
    hooks_install.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project root (default: cwd).",
    )
    hooks_install.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing hook files.",
    )

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
        default="lexical_baseline,embedding_baseline,weighted_graph,hybrid,activation_spreading_v1",
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
    lme.add_argument(
        "--granularity",
        default="session",
        choices=("session", "turn"),
        help="Memory unit granularity: session (default) or turn.",
    )
    return parser


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", help="Natural-language query.")
    parser.add_argument(
        "--mode",
        default=None,
        help="Retriever mode (default: palace config; product tip: hybrid).",
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
        if args.command == "memo":
            return _cmd_memo(args)
        if args.command in {"search", "path"}:
            return _cmd_search(args, emphasize_path=args.command == "path")
        if args.command == "reinforce":
            return _cmd_reinforce(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "backup":
            return _cmd_backup(args)
        if args.command == "repair":
            return _cmd_repair(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "mcp":
            from memory_engine.mcp_server import serve_stdio

            return serve_stdio()
        if args.command == "hooks" and args.hooks_command == "install":
            return _cmd_hooks_install(args)
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
    result = palace_ingest(
        list(args.targets),
        palace=args.palace,
        domain_pack=args.pack,
        glob_pattern=args.glob_pattern,
    )
    print(f"palace: {result['palace']}")
    print(f"domain_pack: {result['domain_pack']}")
    print(f"ingested_files: {len(result['ingested_files'])}")
    for path in result["ingested_files"]:
        print(f"  - {path}")
    print(f"nodes: {result['nodes']}")
    print(f"edges: {result['edges']}")
    return 0


def _cmd_memo(args: argparse.Namespace) -> int:
    text = args.text
    if text is None:
        text = sys.stdin.read()
    result = palace_ingest_memo(
        text,
        palace=args.palace,
        title=args.title,
        source=args.source,
    )
    print(f"palace: {result['palace']}")
    print(f"node_id: {result['node_id']}")
    print(f"nodes: {result['nodes']}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    status = palace_status(args.palace)
    for key, value in status.items():
        print(f"{key}: {value}")
    return 0


def _cmd_backup(args: argparse.Namespace) -> int:
    result = backup_palace(args.palace, output=args.output)
    print(f"palace: {result['palace']}")
    print(f"archive: {result['archive']}")
    print(f"files: {', '.join(result['files'])}")
    print(f"bytes: {result['bytes']}")
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    result = repair_palace(args.palace, rebuild_empty=args.rebuild_empty)
    for key in ("palace", "ok", "integrity", "nodes", "edges"):
        if key in result:
            print(f"{key}: {result[key]}")
    if result.get("issues"):
        print("issues:")
        for item in result["issues"]:
            print(f"  - {item}")
    if result.get("actions"):
        print("actions:")
        for item in result["actions"]:
            print(f"  - {item}")
    return 0 if result.get("ok") else 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(args.palace)
    print(f"ok: {report['ok']}")
    for check in report["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"[{mark}] {check['name']}: {check['detail']}")
    print("install tips:")
    for tip in report["install_tips"]:
        print(f"  - {tip}")
    return 0 if report["ok"] else 1


def _cmd_search(args: argparse.Namespace, *, emphasize_path: bool) -> int:
    payload = palace_search(
        args.query,
        palace=args.palace,
        mode=args.mode,
        top_k=args.top_k,
    )
    return _print_search_payload(payload, emphasize_path=emphasize_path, as_json=args.as_json)


def _cmd_reinforce(args: argparse.Namespace) -> int:
    payload = palace_reinforce(
        args.query,
        palace=args.palace,
        mode=args.mode,
        top_k=args.top_k,
        policy=args.policy,
    )
    return _print_search_payload(payload, emphasize_path=True, as_json=args.as_json)


def _print_search_payload(
    payload: dict,
    *,
    emphasize_path: bool,
    as_json: bool,
) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"palace: {payload['palace']}")
    print(f"mode: {payload['retriever_mode']}")
    print(f"query: {payload['query']}")
    print()
    print("ANSWER")
    print(payload["answer"] or "(empty)")
    print()
    print("PATH" if emphasize_path else "BEST PATH")
    hops = payload.get("hop_explanations") or []
    if hops:
        for line in hops:
            print(f"  {line}")
    elif payload.get("paths"):
        for index, step in enumerate(payload["paths"][0]["steps"]):
            via = step["via_edge_type"] or "seed"
            print(f"  hop {index}: {step['node_id']} via={via} score={step['score']:.3f}")
    else:
        print("  (no path)")
    edges = payload.get("path_edge_types") or []
    if edges:
        print(f"edges: {', '.join(edges)}")
    print(f"confidence: {float(payload.get('confidence') or 0.0):.3f}")
    return 0


def _cmd_hooks_install(args: argparse.Namespace) -> int:
    written = install_hooks(args.project, force=args.force)
    print("installed hook templates:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print()
    print("Next: merge .cursor/mpe-hooks/mcp.local.json into Cursor MCP settings.")
    return 0


def _cmd_bench_longmemeval(args: argparse.Namespace) -> int:
    label = args.label
    if args.granularity == "turn" and "turn" not in label:
        label = f"{label}-turn"
    result = run_longmemeval_baseline(
        dataset=args.dataset,
        limit=args.limit,
        modes=tuple(part.strip() for part in args.modes.split(",") if part.strip()),
        top_k=args.top_k,
        output_dir=args.output_dir,
        label=label,
        granularity=args.granularity,
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
