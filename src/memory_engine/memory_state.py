from __future__ import annotations

from dataclasses import dataclass, field

from memory_engine.memory.domain.enums import MemoryLifecycleState
from memory_engine.memory.domain.memory_state import DomainMemoryState
from memory_engine.memory.domain.memory_state_machine import MemoryStateMachine
from memory_engine.schema import MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


@dataclass(slots=True)
class MemoryStatePolicy:
    reinforcement_step: int = 1
    usage_bonus_step: float = 0.02
    usage_bonus_cap: float = 0.15
    decay_rate: float = 0.05
    minimum_decay_factor: float = 0.5
    state_machine: MemoryStateMachine = field(default_factory=MemoryStateMachine)

    def reinforce_node(self, node: MemoryNode) -> None:
        node.weights.usage_count += self.reinforcement_step
        node.weights.decay_factor = min(1.0, node.weights.decay_factor + 0.02)
        self._write_domain_state(
            node,
            self.state_machine.reinforce(self._read_domain_state(node)),
        )

    def decay_node(self, node: MemoryNode, *, steps: int = 1) -> None:
        decay_multiplier = max(0.0, 1.0 - self.decay_rate * steps)
        node.weights.decay_factor = max(
            self.minimum_decay_factor,
            node.weights.decay_factor * decay_multiplier,
        )
        self._write_domain_state(
            node,
            self.state_machine.decay(self._read_domain_state(node), elapsed_steps=steps),
        )

    def effective_weight_score(self, weight: MemoryWeight) -> float:
        usage_bonus = min(weight.usage_count * self.usage_bonus_step, self.usage_bonus_cap)
        score = (weight.bounded_score() + usage_bonus) * weight.decay_factor
        return max(0.0, min(score, 1.0))

    def recall_multiplier(self, node: MemoryNode) -> float:
        return self.recall_multiplier_for_state(self._read_domain_state(node))

    def recall_multiplier_for_state(self, state: DomainMemoryState) -> float:
        base = {
            MemoryLifecycleState.ENCODED: 0.95,
            MemoryLifecycleState.ACTIVE: 1.2,
            MemoryLifecycleState.STABILIZING: 1.08,
            MemoryLifecycleState.CONSOLIDATED: 1.12,
            MemoryLifecycleState.FADING: 0.65,
            MemoryLifecycleState.ARCHIVED: 0.3,
        }[state.state]
        stability_bonus = 0.0
        if state.state in {
            MemoryLifecycleState.ACTIVE,
            MemoryLifecycleState.STABILIZING,
            MemoryLifecycleState.CONSOLIDATED,
        }:
            stability_bonus = min(0.1, state.stability_score * 0.12)
        return max(0.2, min(base + stability_bonus, 1.35))

    def propagation_factor(self, node: MemoryNode) -> float:
        return max(0.0, min(node.weights.decay_factor, 1.0))

    def _read_domain_state(self, node: MemoryNode) -> DomainMemoryState:
        state_value = str(
            node.attributes.get("lifecycle_state", MemoryLifecycleState.ENCODED.value)
        )
        try:
            lifecycle_state = MemoryLifecycleState(state_value)
        except ValueError:
            lifecycle_state = MemoryLifecycleState.ENCODED
        reinforcement_count = (
            int(node.attributes["reinforcement_count"])
            if "reinforcement_count" in node.attributes
            else 0
        )
        return DomainMemoryState(
            state=lifecycle_state,
            reinforcement_count=reinforcement_count,
            stability_score=float(node.attributes.get("stability_score", 0.0)),
            decay_factor=node.weights.decay_factor,
        )

    def _write_domain_state(self, node: MemoryNode, state: DomainMemoryState) -> None:
        node.attributes["lifecycle_state"] = state.state.value
        node.attributes["reinforcement_count"] = state.reinforcement_count
        node.attributes["stability_score"] = round(state.stability_score, 6)
        node.weights.decay_factor = state.decay_factor


@dataclass(slots=True)
class StaticMemoryStatePolicy(MemoryStatePolicy):
    def reinforce_node(self, node: MemoryNode) -> None:
        del node

    def decay_node(self, node: MemoryNode, *, steps: int = 1) -> None:
        del node, steps

    def effective_weight_score(self, weight: MemoryWeight) -> float:
        return weight.bounded_score()

    def propagation_factor(self, node: MemoryNode) -> float:
        del node
        return 1.0

    def recall_multiplier(self, node: MemoryNode) -> float:
        del node
        return 1.0

    def recall_multiplier_for_state(self, state: DomainMemoryState) -> float:
        del state
        return 1.0


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
