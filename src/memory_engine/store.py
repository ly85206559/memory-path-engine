from __future__ import annotations

from collections import defaultdict

from memory_engine.schema import MemoryEdge, MemoryNode


class MemoryStore:
    def __init__(self) -> None:
        self._nodes: dict[str, MemoryNode] = {}
        self._adjacency: dict[str, list[MemoryEdge]] = defaultdict(list)

    def add_node(self, node: MemoryNode) -> None:
        self._nodes[node.id] = node

    def add_edge(self, edge: MemoryEdge) -> None:
        self._adjacency[edge.from_id].append(edge)
        if edge.bidirectional:
            self._adjacency[edge.to_id].append(
                MemoryEdge(
                    from_id=edge.to_id,
                    to_id=edge.from_id,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    confidence=edge.confidence,
                    bidirectional=True,
                    source_ref=edge.source_ref,
                )
            )

    def get_node(self, node_id: str) -> MemoryNode:
        return self._nodes[node_id]

    def nodes(self) -> list[MemoryNode]:
        return list(self._nodes.values())

    def neighbors(self, node_id: str) -> list[MemoryEdge]:
        return list(self._adjacency.get(node_id, []))
