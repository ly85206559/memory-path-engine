from __future__ import annotations

from dataclasses import dataclass, field

from memory_engine.memory.domain.memory_state_machine import MemoryStateMachine
from memory_engine.memory_state import MemoryStatePolicy
from memory_engine.schema import MemoryNode


@dataclass(slots=True)
class MildForgettingPolicy(MemoryStatePolicy):
    """Slow decay; useful as the conservative online-memory baseline."""

    decay_rate: float = 0.02
    minimum_decay_factor: float = 0.7
    state_machine: MemoryStateMachine = field(
        default_factory=lambda: MemoryStateMachine(
            decay_stability_penalty=0.05,
            decay_factor_step=0.02,
        )
    )


@dataclass(slots=True)
class AggressiveForgettingPolicy(MemoryStatePolicy):
    """Faster decay; unused memories fade quickly across ordered queries."""

    decay_rate: float = 0.18
    minimum_decay_factor: float = 0.15
    state_machine: MemoryStateMachine = field(
        default_factory=lambda: MemoryStateMachine(
            decay_stability_penalty=0.2,
            decay_factor_step=0.12,
        )
    )


def policy_by_name(name: str) -> MemoryStatePolicy:
    normalized = name.strip().lower()
    if normalized in {"mild", "mild_forgetting"}:
        return MildForgettingPolicy()
    if normalized in {"aggressive", "aggressive_forgetting"}:
        return AggressiveForgettingPolicy()
    if normalized in {"default", "standard"}:
        return MemoryStatePolicy()
    raise ValueError(f"Unknown forgetting policy '{name}'")


def snapshot_lifecycle(node: MemoryNode) -> dict[str, float | int | str]:
    return {
        "node_id": node.id,
        "lifecycle_state": str(node.attributes.get("lifecycle_state", "encoded")),
        "reinforcement_count": int(node.attributes.get("reinforcement_count", 0)),
        "stability_score": float(node.attributes.get("stability_score", 0.0)),
        "decay_factor": float(node.weights.decay_factor),
        "usage_count": int(node.weights.usage_count),
    }
