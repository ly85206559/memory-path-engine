from __future__ import annotations

from dataclasses import dataclass

from memory_engine.schema import MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


@dataclass(slots=True)
class MemoryStatePolicy:
    reinforcement_step: int = 1
    usage_bonus_step: float = 0.02
    usage_bonus_cap: float = 0.15
    decay_rate: float = 0.05
    minimum_decay_factor: float = 0.5

    def reinforce_node(self, node: MemoryNode) -> None:
        node.weights.usage_count += self.reinforcement_step
        node.weights.decay_factor = min(1.0, node.weights.decay_factor + 0.02)

    def decay_node(self, node: MemoryNode, *, steps: int = 1) -> None:
        decay_multiplier = max(0.0, 1.0 - self.decay_rate * steps)
        node.weights.decay_factor = max(
            self.minimum_decay_factor,
            node.weights.decay_factor * decay_multiplier,
        )

    def effective_weight_score(self, weight: MemoryWeight) -> float:
        usage_bonus = min(weight.usage_count * self.usage_bonus_step, self.usage_bonus_cap)
        score = (weight.bounded_score() + usage_bonus) * weight.decay_factor
        return max(0.0, min(score, 1.0))


def reinforce_result_paths(store: MemoryStore, *, paths, policy: MemoryStatePolicy | None = None) -> None:
    policy = policy or MemoryStatePolicy()
    seen_node_ids: set[str] = set()
    for path in paths:
        for step in path.steps:
            if step.node_id in seen_node_ids:
                continue
            seen_node_ids.add(step.node_id)
            policy.reinforce_node(store.get_node(step.node_id))


def decay_unvisited_nodes(
    store: MemoryStore,
    *,
    visited_node_ids: set[str],
    policy: MemoryStatePolicy | None = None,
    steps: int = 1,
) -> None:
    policy = policy or MemoryStatePolicy()
    for node in store.nodes():
        if node.id in visited_node_ids:
            continue
        policy.decay_node(node, steps=steps)
