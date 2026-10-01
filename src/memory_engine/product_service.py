from __future__ import annotations

"""Shared product operations used by CLI and MCP (Product M2)."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_engine.api import (
    apply_online_memory_step,
    recall_from_store,
    reason_from_recall,
)
from memory_engine.palace_workspace import (
    PalaceWorkspace,
    init_palace,
    open_palace,
    resolve_palace_root,
)
from memory_engine.schema import EvidenceRef, MemoryNode, MemoryWeight


def ensure_palace(
    path: Path | str | None = None,
    *,
    create: bool = True,
) -> PalaceWorkspace:
    root = resolve_palace_root(path)
    try:
        return open_palace(root)
    except FileNotFoundError:
        if not create:
            raise
        return init_palace(root)


def palace_status(path: Path | str | None = None) -> dict[str, Any]:
    return ensure_palace(path, create=False).status()


def palace_ingest(
    targets: list[str | Path],
    *,
    palace: Path | str | None = None,
    domain_pack: str | None = None,
    glob_pattern: str = "*.md",
) -> dict[str, Any]:
    workspace = ensure_palace(palace, create=True)
    result = workspace.ingest_paths(
        [Path(item) for item in targets],
        domain_pack=domain_pack,
        glob_pattern=glob_pattern,
    )
    return {"palace": str(workspace.root), **result}


def palace_ingest_memo(
    text: str,
    *,
    palace: Path | str | None = None,
    title: str | None = None,
    source: str = "memo",
) -> dict[str, Any]:
    """Append a freeform memo node (used by session hooks)."""
    workspace = ensure_palace(palace, create=True)
    store = workspace.load_store()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    node_id = f"memo:{stamp}:{digest}"
    body = text.strip()
    if title:
        body = f"{title.strip()}\n\n{body}"
    store.add_node(
        MemoryNode(
            id=node_id,
            type="memo",
            content=body,
            attributes={
                "memory_kind": "episodic",
                "source_kind": source,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
            weights=MemoryWeight(importance=0.55, novelty=0.4, confidence=0.9),
            source_ref=EvidenceRef(source_path=source, section_id=node_id),
        )
    )
    workspace.save_store(store)
    return {
        "palace": str(workspace.root),
        "node_id": node_id,
        "nodes": len(store.nodes()),
        "edges": len(store.edges()),
    }


def palace_search(
    query: str,
    *,
    palace: Path | str | None = None,
    mode: str | None = None,
    top_k: int = 3,
    reinforce: bool = False,
    forgetting_policy: str | None = None,
) -> dict[str, Any]:
    workspace = ensure_palace(palace, create=False)
    store = workspace.load_store()
    if not store.nodes():
        raise ValueError("palace is empty — ingest documents or memos first")
    retriever_mode = mode or workspace.config.default_retriever_mode
    unified = recall_from_store(
        store,
        query,
        retriever_mode=retriever_mode,
        top_k=top_k,
        project_palace=False,
    )
    snapshots = None
    if reinforce or forgetting_policy:
        snapshots = apply_online_memory_step(
            store,
            unified,
            policy=forgetting_policy or "default",
        )
    reasoned = reason_from_recall(query, unified, store=store)
    workspace.save_store(store)
    return {
        "palace": str(workspace.root),
        "query": query,
        "retriever_mode": retriever_mode,
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
        "lifecycle_snapshots": snapshots,
    }


def palace_reinforce(
    query: str,
    *,
    palace: Path | str | None = None,
    mode: str | None = None,
    top_k: int = 3,
    policy: str = "mild",
) -> dict[str, Any]:
    return palace_search(
        query,
        palace=palace,
        mode=mode,
        top_k=top_k,
        reinforce=True,
        forgetting_policy=policy,
    )
