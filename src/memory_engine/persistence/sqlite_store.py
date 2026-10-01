from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from memory_engine.schema import EvidenceRef, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    embedding_json TEXT,
    importance REAL NOT NULL DEFAULT 0,
    risk REAL NOT NULL DEFAULT 0,
    novelty REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 1,
    usage_count INTEGER NOT NULL DEFAULT 0,
    decay_factor REAL NOT NULL DEFAULT 1,
    source_path TEXT,
    section_id TEXT,
    line_start INTEGER,
    line_end INTEGER,
    source_metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 1,
    bidirectional INTEGER NOT NULL DEFAULT 0,
    source_path TEXT,
    section_id TEXT,
    line_start INTEGER,
    line_end INTEGER,
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(from_id, to_id, edge_type, bidirectional)
);

CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
"""


def save_store(store: MemoryStore, db_path: Path | str) -> None:
    """Persist a ``MemoryStore`` into a SQLite file (overwrite contents)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM nodes")
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        for node in store.nodes():
            _insert_node(conn, node)
        # Persist adjacency exactly as stored (bidirectional already expanded).
        for edge in store.edges():
            _insert_edge(conn, edge)
        conn.commit()


def load_store(db_path: Path | str) -> MemoryStore:
    """Load a ``MemoryStore`` from SQLite. Missing file yields an empty store."""
    path = Path(db_path)
    store = MemoryStore()
    if not path.exists():
        return store
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA_SQL)
        for row in conn.execute("SELECT * FROM nodes"):
            store.add_node(_row_to_node(row))
        for row in conn.execute("SELECT * FROM edges"):
            # Edges are stored in expanded adjacency form; never auto-mirror again.
            store.add_edge(_row_to_edge(row, force_unidirectional=True))
    return store


def replace_store(store: MemoryStore, db_path: Path | str) -> None:
    """Alias for ``save_store`` — full replace of on-disk graph."""
    save_store(store, db_path)


def read_meta(db_path: Path | str) -> dict[str, str]:
    path = Path(db_path)
    if not path.exists():
        return {}
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA_SQL)
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {str(key): str(value) for key, value in rows}


def write_meta(db_path: Path | str, values: dict[str, Any]) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA_SQL)
        for key, value in values.items():
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(key), json.dumps(value) if not isinstance(value, str) else value),
            )
        conn.commit()


def _insert_node(conn: sqlite3.Connection, node: MemoryNode) -> None:
    source = node.source_ref
    conn.execute(
        """
        INSERT INTO nodes(
            id, type, content, attributes_json, embedding_json,
            importance, risk, novelty, confidence, usage_count, decay_factor,
            source_path, section_id, line_start, line_end, source_metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node.id,
            node.type,
            node.content,
            json.dumps(node.attributes, ensure_ascii=False),
            json.dumps(node.embedding) if node.embedding is not None else None,
            node.weights.importance,
            node.weights.risk,
            node.weights.novelty,
            node.weights.confidence,
            node.weights.usage_count,
            node.weights.decay_factor,
            None if source is None else source.source_path,
            None if source is None else source.section_id,
            None if source is None else source.line_start,
            None if source is None else source.line_end,
            "{}" if source is None else json.dumps(source.metadata, ensure_ascii=False),
        ),
    )


def _insert_edge(conn: sqlite3.Connection, edge: MemoryEdge) -> None:
    source = edge.source_ref
    conn.execute(
        """
        INSERT OR IGNORE INTO edges(
            from_id, to_id, edge_type, weight, confidence, bidirectional,
            source_path, section_id, line_start, line_end, source_metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            edge.from_id,
            edge.to_id,
            edge.edge_type,
            edge.weight,
            edge.confidence,
            1 if edge.bidirectional else 0,
            None if source is None else source.source_path,
            None if source is None else source.section_id,
            None if source is None else source.line_start,
            None if source is None else source.line_end,
            "{}" if source is None else json.dumps(source.metadata, ensure_ascii=False),
        ),
    )


def _row_to_node(row: sqlite3.Row) -> MemoryNode:
    attributes = json.loads(row["attributes_json"] or "{}")
    embedding_raw = row["embedding_json"]
    embedding = json.loads(embedding_raw) if embedding_raw else None
    source_ref = None
    if row["source_path"]:
        source_ref = EvidenceRef(
            source_path=row["source_path"],
            section_id=row["section_id"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            metadata=json.loads(row["source_metadata_json"] or "{}"),
        )
    return MemoryNode(
        id=row["id"],
        type=row["type"],
        content=row["content"],
        attributes=attributes,
        embedding=embedding,
        weights=MemoryWeight(
            importance=float(row["importance"]),
            risk=float(row["risk"]),
            novelty=float(row["novelty"]),
            confidence=float(row["confidence"]),
            usage_count=int(row["usage_count"]),
            decay_factor=float(row["decay_factor"]),
        ),
        source_ref=source_ref,
    )


def _row_to_edge(row: sqlite3.Row, *, force_unidirectional: bool = False) -> MemoryEdge:
    source_ref = None
    if row["source_path"]:
        source_ref = EvidenceRef(
            source_path=row["source_path"],
            section_id=row["section_id"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            metadata=json.loads(row["source_metadata_json"] or "{}"),
        )
    bidirectional = False if force_unidirectional else bool(row["bidirectional"])
    return MemoryEdge(
        from_id=row["from_id"],
        to_id=row["to_id"],
        edge_type=row["edge_type"],
        weight=float(row["weight"]),
        confidence=float(row["confidence"]),
        bidirectional=bidirectional,
        source_ref=source_ref,
    )
