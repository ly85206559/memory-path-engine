from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_engine.ingest import ingest_document
from memory_engine.persistence.sqlite_store import load_store, save_store
from memory_engine.store import MemoryStore

DEFAULT_PALACE_DIRNAME = ".mpe"
CONFIG_FILENAME = "config.json"
STORE_FILENAME = "store.sqlite"
PALACE_ENV_VAR = "MPE_PALACE"


@dataclass(slots=True)
class PalaceConfig:
    name: str
    created_at: str
    schema_version: int = 1
    default_domain_pack: str = "example_runbook_pack"
    default_retriever_mode: str = "weighted_graph"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> PalaceConfig:
        known = {
            "name": str(payload.get("name", "palace")),
            "created_at": str(payload.get("created_at", "")),
            "schema_version": int(payload.get("schema_version", 1)),
            "default_domain_pack": str(
                payload.get("default_domain_pack", "example_runbook_pack")
            ),
            "default_retriever_mode": str(
                payload.get("default_retriever_mode", "weighted_graph")
            ),
        }
        extra = {
            key: value
            for key, value in payload.items()
            if key not in known
        }
        return cls(**known, extra=extra)


@dataclass(slots=True)
class PalaceWorkspace:
    """On-disk palace directory: config.json + store.sqlite."""

    root: Path
    config: PalaceConfig

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def store_path(self) -> Path:
        return self.root / STORE_FILENAME

    def load_store(self) -> MemoryStore:
        return load_store(self.store_path)

    def save_store(self, store: MemoryStore) -> None:
        save_store(store, self.store_path)
        self._touch_config()

    def status(self) -> dict[str, Any]:
        store = self.load_store()
        return {
            "palace": str(self.root.resolve()),
            "name": self.config.name,
            "created_at": self.config.created_at,
            "default_domain_pack": self.config.default_domain_pack,
            "default_retriever_mode": self.config.default_retriever_mode,
            "nodes": len(store.nodes()),
            "edges": len(store.edges()),
            "store_path": str(self.store_path.resolve()),
            "exists": self.store_path.exists(),
        }

    def ingest_paths(
        self,
        paths: list[Path],
        *,
        domain_pack: str | None = None,
        glob_pattern: str = "*.md",
    ) -> dict[str, Any]:
        pack = domain_pack or self.config.default_domain_pack
        store = self.load_store()
        ingested: list[str] = []
        for path in paths:
            resolved = path.expanduser().resolve()
            if resolved.is_dir():
                files = sorted(resolved.glob(glob_pattern))
            else:
                files = [resolved]
            for file_path in files:
                if not file_path.is_file():
                    continue
                ingest_document(file_path, store, domain_pack=pack)
                ingested.append(str(file_path))
        self.save_store(store)
        if domain_pack:
            self.config.default_domain_pack = domain_pack
            self._write_config()
        return {
            "ingested_files": ingested,
            "domain_pack": pack,
            "nodes": len(store.nodes()),
            "edges": len(store.edges()),
        }

    def _touch_config(self) -> None:
        self.config.extra["updated_at"] = _utc_now()
        self._write_config()

    def _write_config(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(self.config.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def resolve_palace_root(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    env = os.environ.get(PALACE_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / DEFAULT_PALACE_DIRNAME).resolve()


def init_palace(
    path: Path | str | None = None,
    *,
    name: str | None = None,
    default_domain_pack: str = "example_runbook_pack",
    default_retriever_mode: str = "weighted_graph",
    overwrite: bool = False,
) -> PalaceWorkspace:
    root = resolve_palace_root(path)
    config_path = root / CONFIG_FILENAME
    store_path = root / STORE_FILENAME
    if (config_path.exists() or store_path.exists()) and not overwrite:
        return open_palace(root)
    root.mkdir(parents=True, exist_ok=True)
    config = PalaceConfig(
        name=name or root.name,
        created_at=_utc_now(),
        default_domain_pack=default_domain_pack,
        default_retriever_mode=default_retriever_mode,
    )
    workspace = PalaceWorkspace(root=root, config=config)
    workspace._write_config()
    if overwrite or not store_path.exists():
        save_store(MemoryStore(), store_path)
    return workspace


def open_palace(path: Path | str | None = None) -> PalaceWorkspace:
    root = resolve_palace_root(path)
    config_path = root / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"No palace found at {root}. Run: mpe init {root}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return PalaceWorkspace(root=root, config=PalaceConfig.from_json(payload))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
