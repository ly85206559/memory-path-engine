from __future__ import annotations

"""Install Cursor/Claude hook templates into a project."""

import json
import shutil
from pathlib import Path

HOOK_FILES = (
    "mpe_session_start.sh",
    "mpe_stop_save.sh",
    "mcp.example.json",
    "hooks.example.json",
)


def package_hooks_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "hooks"


def install_hooks(
    project_root: Path | None = None,
    *,
    force: bool = False,
) -> dict[str, str]:
    root = (project_root or Path.cwd()).resolve()
    dest = root / ".cursor" / "mpe-hooks"
    dest.mkdir(parents=True, exist_ok=True)
    source = package_hooks_dir()
    if not source.exists():
        raise FileNotFoundError(f"Hook templates missing at {source}")
    written: dict[str, str] = {}
    for name in HOOK_FILES:
        src = source / name
        target = dest / name
        if target.exists() and not force:
            written[name] = f"exists:{target}"
            continue
        shutil.copy2(src, target)
        if target.suffix == ".sh":
            target.chmod(target.stat().st_mode | 0o111)
        written[name] = str(target)

    mcp_snippet = {
        "mcpServers": {
            "memory-path-engine": {
                "command": "mpe",
                "args": ["mcp"],
                "env": {"MPE_PALACE": str(root / ".mpe")},
            }
        }
    }
    mcp_path = dest / "mcp.local.json"
    if force or not mcp_path.exists():
        mcp_path.write_text(json.dumps(mcp_snippet, indent=2) + "\n", encoding="utf-8")
        written["mcp.local.json"] = str(mcp_path)
    else:
        written["mcp.local.json"] = f"exists:{mcp_path}"

    readme = dest / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Memory Path Engine hooks",
                "",
                "Installed by `mpe hooks install`.",
                "",
                "- Wire `mcp.local.json` into Cursor MCP settings (or merge `mcpServers`).",
                "- Point session-start / stop hooks at the `.sh` scripts in this folder.",
                "- See `docs/getting-started.md` for the 5-minute closed-loop setup.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    written["README.md"] = str(readme)
    return written
