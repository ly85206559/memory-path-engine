from __future__ import annotations

from dataclasses import dataclass

from memory_engine.schema import MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


@dataclass(frozen=True, slots=True)
class DualRepresentationIds:
    source_node_id: str
    episodic_node_id: str
    semantic_node_id: str


def episodic_id_for(source_node_id: str) -> str:
    return f"{source_node_id}__episodic"


def semantic_id_for(source_node_id: str) -> str:
    return f"{source_node_id}__semantic"


def project_dual_representations(
    store: MemoryStore,
    *,
    source_node_id: str,
    overwrite: bool = False,
) -> DualRepresentationIds:
    """
    Project one source node into linked episodic + semantic representations.

    Episodic keeps event-like wording; semantic keeps a generalized rule form.
    Both stay connected to the source via `recalls` / `summarizes` edges.
    """
    source = store.get_node(source_node_id)
    episodic_id = episodic_id_for(source_node_id)
    semantic_id = semantic_id_for(source_node_id)

    if overwrite:
        for node_id in (episodic_id, semantic_id):
            if node_id in {node.id for node in store.nodes()}:
                # MemoryStore has no delete API; overwrite by re-adding fields via replace pattern.
                pass

    existing_ids = {node.id for node in store.nodes()}
    if episodic_id not in existing_ids or overwrite:
        store.add_node(
            MemoryNode(
                id=episodic_id,
                type="episodic_view",
                content=f"Episode recall: {source.content}",
                attributes={
                    **dict(source.attributes),
                    "memory_kind": "episodic",
                    "representation_of": source_node_id,
                    "representation_role": "episodic",
                    "consolidation_kind": source.attributes.get("consolidation_kind"),
                },
                weights=MemoryWeight(
                    importance=source.weights.importance,
                    risk=source.weights.risk,
                    novelty=min(1.0, source.weights.novelty + 0.1),
                    confidence=source.weights.confidence,
                    usage_count=source.weights.usage_count,
                    decay_factor=source.weights.decay_factor,
                ),
                source_ref=source.source_ref,
            )
        )
    if semantic_id not in existing_ids or overwrite:
        store.add_node(
            MemoryNode(
                id=semantic_id,
                type="semantic_view",
                content=f"Generalized rule: {source.content}",
                attributes={
                    **dict(source.attributes),
                    "memory_kind": "semantic",
                    "representation_of": source_node_id,
                    "representation_role": "semantic",
                    "consolidation_kind": source.attributes.get(
                        "consolidation_kind",
                        "generalized_rule_memory",
                    ),
                },
                weights=MemoryWeight(
                    importance=min(1.0, source.weights.importance + 0.05),
                    risk=source.weights.risk,
                    novelty=max(0.0, source.weights.novelty - 0.05),
                    confidence=min(1.0, source.weights.confidence + 0.02),
                    usage_count=source.weights.usage_count,
                    decay_factor=source.weights.decay_factor,
                ),
                source_ref=source.source_ref,
            )
        )

    _ensure_edge(
        store,
        MemoryEdge(
            from_id=episodic_id,
            to_id=source_node_id,
            edge_type="recalls",
            weight=0.8,
            bidirectional=False,
            source_ref=source.source_ref,
        ),
    )
    _ensure_edge(
        store,
        MemoryEdge(
            from_id=semantic_id,
            to_id=source_node_id,
            edge_type="summarizes",
            weight=0.85,
            bidirectional=False,
            source_ref=source.source_ref,
        ),
    )
    _ensure_edge(
        store,
        MemoryEdge(
            from_id=episodic_id,
            to_id=semantic_id,
            edge_type="summarizes",
            weight=0.7,
            bidirectional=True,
            source_ref=source.source_ref,
        ),
    )
    source.attributes["has_dual_representation"] = True
    source.attributes["episodic_view_id"] = episodic_id
    source.attributes["semantic_view_id"] = semantic_id
    return DualRepresentationIds(
        source_node_id=source_node_id,
        episodic_node_id=episodic_id,
        semantic_node_id=semantic_id,
    )


def project_dual_representations_for_store(
    store: MemoryStore,
    *,
    node_ids: list[str] | None = None,
) -> list[DualRepresentationIds]:
    targets = node_ids or [node.id for node in store.nodes() if "__" not in node.id]
    return [project_dual_representations(store, source_node_id=node_id) for node_id in targets]


def _ensure_edge(store: MemoryStore, edge: MemoryEdge) -> None:
    """Idempotently add an edge. Bidirectional edges rely on ``MemoryStore.add_edge``."""
    existing = {
        (item.from_id, item.to_id, item.edge_type)
        for item in store.edges()
    }
    key = (edge.from_id, edge.to_id, edge.edge_type)
    if key in existing:
        return
    if edge.bidirectional:
        reverse = (edge.to_id, edge.from_id, edge.edge_type)
        if reverse in existing:
            # One direction already present; add the forward side only.
            store.add_edge(
                MemoryEdge(
                    from_id=edge.from_id,
                    to_id=edge.to_id,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    bidirectional=False,
                    source_ref=edge.source_ref,
                )
            )
            return
    store.add_edge(edge)
