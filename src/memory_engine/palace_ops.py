from __future__ import annotations

"""Palace backup / repair / doctor helpers (Product M4)."""

import json
import shutil
import sqlite3
import sys
import tarfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from memory_engine.palace_workspace import (
    CONFIG_FILENAME,
    STORE_FILENAME,
    open_palace,
    resolve_palace_root,
)
from memory_engine.persistence.sqlite_store import load_store, save_store
from memory_engine.store import MemoryStore


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_palace(
    path: Path | str | None = None,
    *,
    output: Path | str | None = None,
) -> dict[str, Any]:
    """Create a ``.tar.gz`` archive of ``config.json`` + ``store.sqlite``."""
    workspace = open_palace(path)
    root = workspace.root
    if output is None:
        out_dir = Path.cwd() / "mpe-backups"
        out_dir.mkdir(parents=True, exist_ok=True)
        archive = out_dir / f"{root.name}-{_utc_stamp()}.tar.gz"
    else:
        target = Path(output).expanduser()
        looks_like_archive = str(target).endswith((".tar.gz", ".tgz"))
        if looks_like_archive:
            archive = target.resolve()
            archive.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = target.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            archive = out_dir / f"{root.name}-{_utc_stamp()}.tar.gz"

    members: list[Path] = []
    for name in (CONFIG_FILENAME, STORE_FILENAME):
        candidate = root / name
        if candidate.exists():
            members.append(candidate)
    if not members:
        raise FileNotFoundError(f"Nothing to back up under {root}")

    with tarfile.open(archive, "w:gz") as tar:
        for member in members:
            tar.add(member, arcname=member.name)

    return {
        "palace": str(root.resolve()),
        "archive": str(archive),
        "files": [member.name for member in members],
        "bytes": archive.stat().st_size,
    }


def repair_palace(
    path: Path | str | None = None,
    *,
    rebuild_empty: bool = False,
) -> dict[str, Any]:
    """Validate palace config + SQLite; quarantine corrupt stores and rebuild."""
    root = resolve_palace_root(path)
    actions: list[str] = []
    issues: list[str] = []

    config_path = root / CONFIG_FILENAME
    store_path = root / STORE_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"No palace found at {root}. Run: mpe init {root}")

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("config.json must be an object")
        actions.append("config_ok")
    except (json.JSONDecodeError, ValueError) as exc:
        issues.append(f"config_invalid:{exc}")
        quarantine = config_path.with_name(f"config.corrupt-{_utc_stamp()}.json")
        shutil.move(str(config_path), str(quarantine))
        from memory_engine.palace_workspace import init_palace

        init_palace(root, overwrite=True)
        actions.append(f"config_rebuilt:{quarantine.name}")
        return {
            "palace": str(root.resolve()),
            "ok": False,
            "issues": issues,
            "actions": actions,
            "nodes": 0,
            "edges": 0,
        }

    integrity = "missing"
    if store_path.exists():
        integrity = _sqlite_integrity(store_path)
        if integrity != "ok":
            issues.append(f"sqlite_integrity:{integrity}")
            quarantine = store_path.with_name(f"store.corrupt-{_utc_stamp()}.sqlite")
            shutil.move(str(store_path), str(quarantine))
            save_store(MemoryStore(), store_path)
            actions.append(f"store_quarantined:{quarantine.name}")
            actions.append("store_rebuilt_empty")
        elif rebuild_empty:
            quarantine = store_path.with_name(f"store.backup-{_utc_stamp()}.sqlite")
            shutil.copy2(store_path, quarantine)
            save_store(MemoryStore(), store_path)
            actions.append(f"store_backup:{quarantine.name}")
            actions.append("store_forced_empty")
        else:
            removed = _drop_dangling_edges(store_path)
            if removed:
                actions.append(f"dangling_edges_removed:{removed}")
            else:
                actions.append("store_ok")
            # Round-trip through loader to ensure schema is present.
            store = load_store(store_path)
            save_store(store, store_path)
            actions.append("schema_ensured")
    else:
        issues.append("store_missing")
        save_store(MemoryStore(), store_path)
        actions.append("store_created_empty")

    workspace = open_palace(root)
    store = workspace.load_store()
    final_integrity = _sqlite_integrity(workspace.store_path) if workspace.store_path.exists() else "missing"
    return {
        "palace": str(root.resolve()),
        "ok": final_integrity == "ok" and not any(item.startswith("config_invalid") for item in issues),
        "integrity": final_integrity,
        "issues": issues,
        "actions": actions,
        "nodes": len(store.nodes()),
        "edges": len(store.edges()),
    }


def doctor_report(path: Path | str | None = None) -> dict[str, Any]:
    """Environment + palace health summary for install debugging."""
    try:
        pkg_version = version("memory-path-engine")
    except PackageNotFoundError:
        pkg_version = "editable/unknown"

    checks: list[dict[str, Any]] = [
        {"name": "python", "ok": sys.version_info >= (3, 11), "detail": sys.version.split()[0]},
        {"name": "package", "ok": True, "detail": pkg_version},
    ]
    try:
        import pydantic  # noqa: F401

        checks.append({"name": "pydantic", "ok": True, "detail": getattr(pydantic, "__version__", "present")})
    except Exception as exc:  # pragma: no cover - dependency always present in tests
        checks.append({"name": "pydantic", "ok": False, "detail": str(exc)})

    root = resolve_palace_root(path)
    palace_info: dict[str, Any] = {"palace": str(root), "exists": (root / CONFIG_FILENAME).exists()}
    if palace_info["exists"]:
        try:
            status = open_palace(root).status()
            palace_info.update(
                {
                    "ok": True,
                    "nodes": status["nodes"],
                    "edges": status["edges"],
                    "store_exists": status["exists"],
                }
            )
            if Path(status["store_path"]).exists():
                integrity = _sqlite_integrity(Path(status["store_path"]))
                palace_info["integrity"] = integrity
                palace_info["ok"] = integrity == "ok"
        except Exception as exc:
            palace_info.update({"ok": False, "error": str(exc)})
    else:
        palace_info["ok"] = False
        palace_info["hint"] = f"Run: mpe init {root}"

    checks.append(
        {
            "name": "palace",
            "ok": bool(palace_info.get("ok")),
            "detail": palace_info.get("error")
            or palace_info.get("hint")
            or f"nodes={palace_info.get('nodes', 0)} integrity={palace_info.get('integrity', 'n/a')}",
        }
    )
    return {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "palace": palace_info,
        "install_tips": [
            "pip install memory-path-engine",
            "pipx install memory-path-engine",
            "uv tool install memory-path-engine",
            "pip install 'memory-path-engine[embed]'  # optional dense embeddings",
            "pip install -e .  # from a clone",
            "docker build -t mpe-mcp . && docker run -i --rm -v \"$PWD/.mpe:/data/palace\" -e MPE_PALACE=/data/palace mpe-mcp",
        ],
    }


def _sqlite_integrity(store_path: Path) -> str:
    try:
        with sqlite3.connect(store_path) as conn:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        if not row:
            return "empty"
        return str(row[0])
    except sqlite3.Error as exc:
        return f"error:{exc}"


def _drop_dangling_edges(store_path: Path) -> int:
    store = load_store(store_path)
    node_ids = {node.id for node in store.nodes()}
    kept = MemoryStore()
    for node in store.nodes():
        kept.add_node(node)
    removed = 0
    for edge in store.edges():
        if edge.from_id not in node_ids or edge.to_id not in node_ids:
            removed += 1
            continue
        kept.add_edge(edge)
    if removed:
        save_store(kept, store_path)
    return removed
