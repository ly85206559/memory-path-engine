from __future__ import annotations

"""
Minimal stdio MCP server for Memory Path Engine (Product M2).

Implements enough JSON-RPC to work with Cursor / Claude MCP clients without
pulling an external MCP SDK dependency.
"""

import json
import sys
from typing import Any

from memory_engine.product_service import (
    ensure_palace,
    palace_ingest,
    palace_ingest_memo,
    palace_reinforce,
    palace_search,
    palace_status,
)

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "memory-path-engine",
    "version": "0.3.0",
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "mpe_status",
        "description": "Show local palace path, node/edge counts, and defaults.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "palace": {"type": "string", "description": "Optional palace directory."},
            },
        },
    },
    {
        "name": "mpe_ingest",
        "description": "Ingest markdown files or a directory into the palace via a domain pack.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files or directories to ingest.",
                },
                "pack": {"type": "string", "description": "Domain pack name."},
                "palace": {"type": "string"},
                "glob": {"type": "string", "default": "*.md"},
            },
            "required": ["targets"],
        },
    },
    {
        "name": "mpe_ingest_memo",
        "description": "Append a freeform session memo into the palace (hook-friendly).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "title": {"type": "string"},
                "palace": {"type": "string"},
                "source": {"type": "string", "default": "memo"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "mpe_search",
        "description": "Search the palace and return answer + replayable path hops.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {
                    "type": "string",
                    "description": "Retriever mode, e.g. weighted_graph or hybrid.",
                },
                "top_k": {"type": "integer", "default": 3},
                "palace": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "mpe_get_path",
        "description": "Alias of mpe_search emphasizing hop citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
                "palace": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "mpe_reinforce",
        "description": "Search then apply an online reinforce/forget step (default mild).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "policy": {"type": "string", "default": "mild"},
                "mode": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
                "palace": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "mpe_init",
        "description": "Initialize a local palace directory if missing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "palace": {"type": "string"},
                "name": {"type": "string"},
            },
        },
    },
]


def _tool_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


def call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
    try:
        if name == "mpe_status":
            return _tool_result(palace_status(args.get("palace")))
        if name == "mpe_init":
            workspace = ensure_palace(args.get("palace"), create=True)
            if args.get("name"):
                workspace.config.name = str(args["name"])
                workspace._write_config()
            return _tool_result(workspace.status())
        if name == "mpe_ingest":
            targets = args.get("targets") or []
            if isinstance(targets, str):
                targets = [targets]
            return _tool_result(
                palace_ingest(
                    targets,
                    palace=args.get("palace"),
                    domain_pack=args.get("pack"),
                    glob_pattern=str(args.get("glob") or "*.md"),
                )
            )
        if name == "mpe_ingest_memo":
            return _tool_result(
                palace_ingest_memo(
                    str(args.get("text") or ""),
                    palace=args.get("palace"),
                    title=args.get("title"),
                    source=str(args.get("source") or "memo"),
                )
            )
        if name in {"mpe_search", "mpe_get_path"}:
            return _tool_result(
                palace_search(
                    str(args.get("query") or ""),
                    palace=args.get("palace"),
                    mode=args.get("mode"),
                    top_k=int(args.get("top_k") or 3),
                )
            )
        if name == "mpe_reinforce":
            return _tool_result(
                palace_reinforce(
                    str(args.get("query") or ""),
                    palace=args.get("palace"),
                    mode=args.get("mode"),
                    top_k=int(args.get("top_k") or 3),
                    policy=str(args.get("policy") or "mild"),
                )
            )
        return _tool_result(f"Unknown tool: {name}", is_error=True)
    except Exception as exc:  # noqa: BLE001 - MCP tools must surface errors to client
        return _tool_result(f"{type(exc).__name__}: {exc}", is_error=True)


def handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    req_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": call_tool(name, arguments),
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if req_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _read_message(stdin) -> dict[str, Any] | None:
    """Read one MCP message (Content-Length framing or newline JSON)."""
    # Prefer Content-Length framing when present.
    header_lines: list[str] = []
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line in ("\n", "\r\n"):
            break
        header_lines.append(line)
        # Newline-delimited JSON fallback: first non-empty line is the body.
        if not header_lines[0].lower().startswith("content-length:") and line.strip().startswith("{"):
            return json.loads(line)

    headers = {}
    for raw in header_lines:
        if ":" in raw:
            key, value = raw.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = stdin.read(length)
    if not body:
        return None
    return json.loads(body)


def _write_message(stdout, message: dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False)
    encoded = payload.encode("utf-8")
    stdout.write(f"Content-Length: {len(encoded)}\r\n\r\n")
    stdout.write(payload)
    stdout.flush()


def serve_stdio() -> int:
    stdin = sys.stdin
    stdout = sys.stdout
    # Ensure palace exists for agent sessions.
    ensure_palace(None, create=True)
    while True:
        try:
            message = _read_message(stdin)
        except json.JSONDecodeError as exc:
            _write_message(
                stdout,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                },
            )
            continue
        if message is None:
            return 0
        response = handle_request(message)
        if response is not None:
            _write_message(stdout, response)


def main() -> None:
    raise SystemExit(serve_stdio())


if __name__ == "__main__":
    main()
