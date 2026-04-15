from __future__ import annotations

from memory_engine.memory.domain.enums import MemoryKind, MemoryLifecycleState, MemoryLinkType
from memory_engine.memory.domain.encoding import EncodingProfile, TriggerProfile
from memory_engine.memory.domain.memory_state import DomainMemoryState
from memory_engine.memory.domain.memory_types import EpisodicMemory, Memory, RouteMemory, SemanticMemory
from memory_engine.memory.domain.palace import MemoryLink, MemoryPalace, PalaceSpace
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile
from memory_engine.schema import EvidenceRef, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


def memory_to_node(memory: Memory) -> MemoryNode:
    return MemoryNode(
        id=memory.memory_id,
        type=memory.kind.value,
        content=memory.content,
        attributes={
            **memory.metadata,
            "memory_kind": memory.kind.value,
            "palace_id": memory.palace_id,
            "location": memory.location.as_key(),
            "lifecycle_state": memory.state.state.value,
            "reinforcement_count": memory.state.reinforcement_count,
            "stability_score": memory.state.stability_score,
            "trigger_phrases": list(memory.encoding.trigger_profile.phrases),
            "trigger_situations": list(memory.encoding.trigger_profile.situations),
            "scenario_tags": list(memory.encoding.scenario_tags),
            "symbolic_tags": list(memory.encoding.symbolic_tags),
        },
        weights=MemoryWeight(
            importance=memory.salience.importance,
            risk=memory.salience.risk,
            novelty=memory.salience.novelty,
            confidence=memory.salience.confidence,
            usage_count=memory.state.reinforcement_count,
            decay_factor=memory.state.decay_factor,
        ),
        source_ref=memory.source,
    )


def link_to_edge(link: MemoryLink) -> MemoryEdge:
    return MemoryEdge(
        from_id=link.from_memory_id,
        to_id=link.to_memory_id,
        edge_type=link.link_type.value,
        weight=link.strength,
        confidence=link.confidence,
        bidirectional=link.bidirectional,
        source_ref=link.source,
    )


def palace_to_store(palace: MemoryPalace) -> MemoryStore:
    store = MemoryStore()
    for memory in palace.memories.values():
        store.add_node(memory_to_node(memory))
    for link in palace.links:
        store.add_edge(link_to_edge(link))
    return store


def store_to_palace(store: MemoryStore, *, palace_id: str = "legacy-palace") -> MemoryPalace:
    palace = MemoryPalace(palace_id=palace_id)
    for node in store.nodes():
        location = _location_from_attributes(node.attributes)
        state = DomainMemoryState(
            state=MemoryLifecycleState(
                node.attributes.get("lifecycle_state", MemoryLifecycleState.ENCODED.value)
            ),
            reinforcement_count=int(node.attributes.get("reinforcement_count", node.weights.usage_count)),
            stability_score=float(node.attributes.get("stability_score", 0.0)),
            decay_factor=node.weights.decay_factor,
        )
        metadata = dict(node.attributes)
        kind = MemoryKind(metadata.get("memory_kind", _kind_from_node_type(node.type).value))
        memory_cls = {
            MemoryKind.EPISODIC: EpisodicMemory,
            MemoryKind.SEMANTIC: SemanticMemory,
            MemoryKind.ROUTE: RouteMemory,
        }[kind]
        memory = memory_cls(
            memory_id=node.id,
            palace_id=palace_id,
            location=location,
            content=node.content,
            salience=SalienceProfile(
                importance=node.weights.importance,
                risk=node.weights.risk,
                novelty=node.weights.novelty,
                confidence=node.weights.confidence,
                recency=float(metadata.get("recency", 0.0)),
            ),
            source=node.source_ref,
            state=state,
            encoding=EncodingProfile(
                trigger_profile=TriggerProfile(
                    phrases=tuple(metadata.get("trigger_phrases", ())),
                    situations=tuple(metadata.get("trigger_situations", ())),
                ),
                scenario_tags=tuple(metadata.get("scenario_tags", ())),
                symbolic_tags=tuple(metadata.get("symbolic_tags", ())),
            ),
            metadata=metadata,
        )
        palace.add_memory(memory)
        space_id = str(metadata.get("space_id", location.as_key() or "legacy-space"))
        if space_id not in palace.spaces:
            palace.add_space(
                PalaceSpace(
                    space_id=space_id,
                    name=space_id,
                    location=location,
                )
            )
    for edge in store.edges():
        palace.add_link(
            MemoryLink(
                from_memory_id=edge.from_id,
                to_memory_id=edge.to_id,
                link_type=_link_type_from_edge_type(edge.edge_type),
                strength=edge.weight,
                confidence=edge.confidence,
                source=edge.source_ref,
                bidirectional=edge.bidirectional,
            )
        )
    return palace


def _kind_from_node_type(node_type: str) -> MemoryKind:
    lowered = node_type.lower()
    if "route" in lowered:
        return MemoryKind.ROUTE
    if "session" in lowered or "episode" in lowered or "event" in lowered:
        return MemoryKind.EPISODIC
    return MemoryKind.SEMANTIC


def _link_type_from_edge_type(edge_type: str) -> MemoryLinkType:
    try:
        return MemoryLinkType(edge_type)
    except ValueError:
        if edge_type == "next_session":
            return MemoryLinkType.NEXT
        return MemoryLinkType.RECALLS


def _location_from_attributes(attributes: dict) -> PalaceLocation:
    location = str(attributes.get("location", "legacy"))
    parts = [part for part in location.split("/") if part]
    padded = parts + [None] * max(0, 4 - len(parts))
    return PalaceLocation(
        building=padded[0] or "legacy",
        floor=padded[1],
        room=padded[2],
        locus=padded[3],
    )
