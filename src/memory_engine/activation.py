from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memory_engine.schema import MemoryEdge


@dataclass(frozen=True, slots=True)
class ActivationSignal:
    node_id: str
    activation: float
    hop: int = 0
    source_node_id: str | None = None
    via_edge_type: str | None = None


@dataclass(frozen=True, slots=True)
class ActivatedNode:
    node_id: str
    activation: float
    score: float
    hop: int
    source_node_id: str | None = None
    via_edge_type: str | None = None


@dataclass(frozen=True, slots=True)
class PropagationStep:
    from_node_id: str
    to_node_id: str
    edge_type: str
    hop: int
    incoming_activation: float
    propagated_activation: float
    stopped_reason: str | None = None


class PropagationPolicy(Protocol):
    def seed_activation(self, *, seed_score: float) -> float:
        """Return the initial activation for a seed node."""

    def propagate(
        self,
        *,
        signal: ActivationSignal,
        edge: MemoryEdge,
    ) -> PropagationStep:
        """Return the propagation result for a single edge traversal."""


class DefaultPropagationPolicy:
    def __init__(
        self,
        *,
        activation_decay: float = 0.75,
        activation_threshold: float = 0.15,
        allowed_edge_types: set[str] | None = None,
    ) -> None:
        self.activation_decay = activation_decay
        self.activation_threshold = activation_threshold
        self.allowed_edge_types = allowed_edge_types

    def seed_activation(self, *, seed_score: float) -> float:
        return max(0.0, seed_score)

    def propagate(
        self,
        *,
        signal: ActivationSignal,
        edge: MemoryEdge,
    ) -> PropagationStep:
        if self.allowed_edge_types is not None and edge.edge_type not in self.allowed_edge_types:
            return PropagationStep(
                from_node_id=signal.node_id,
                to_node_id=edge.to_id,
                edge_type=edge.edge_type,
                hop=signal.hop + 1,
                incoming_activation=signal.activation,
                propagated_activation=0.0,
                stopped_reason="disallowed_edge_type",
            )

        propagated_activation = signal.activation * self.activation_decay * edge.weight
        if propagated_activation < self.activation_threshold:
            return PropagationStep(
                from_node_id=signal.node_id,
                to_node_id=edge.to_id,
                edge_type=edge.edge_type,
                hop=signal.hop + 1,
                incoming_activation=signal.activation,
                propagated_activation=propagated_activation,
                stopped_reason="below_threshold",
            )

        return PropagationStep(
            from_node_id=signal.node_id,
            to_node_id=edge.to_id,
            edge_type=edge.edge_type,
            hop=signal.hop + 1,
            incoming_activation=signal.activation,
            propagated_activation=propagated_activation,
        )
