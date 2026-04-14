from __future__ import annotations

from dataclasses import dataclass

from memory_engine.memory.domain.enums import MemoryLifecycleState
from memory_engine.memory.domain.memory_state import DomainMemoryState


@dataclass(slots=True)
class MemoryStateMachine:
    reinforcement_stability_gain: float = 0.2
    decay_stability_penalty: float = 0.1
    decay_factor_step: float = 0.05

    def reinforce(
        self,
        state: DomainMemoryState,
        *,
        accessed_at: str | None = None,
    ) -> DomainMemoryState:
        reinforcement_count = state.reinforcement_count + 1
        stability_score = min(1.0, state.stability_score + self.reinforcement_stability_gain)
        next_state = MemoryLifecycleState.ACTIVE
        if reinforcement_count >= 2:
            next_state = MemoryLifecycleState.STABILIZING
        if reinforcement_count >= 4 or stability_score >= 0.8:
            next_state = MemoryLifecycleState.CONSOLIDATED
        return state.with_updates(
            state=next_state,
            reinforcement_count=reinforcement_count,
            last_accessed_at=accessed_at,
            stability_score=stability_score,
            decay_factor=min(1.0, state.decay_factor + 0.02),
        )

    def decay(self, state: DomainMemoryState, *, elapsed_steps: int = 1) -> DomainMemoryState:
        next_decay_factor = max(0.0, state.decay_factor - self.decay_factor_step * elapsed_steps)
        stability_score = max(0.0, state.stability_score - self.decay_stability_penalty * elapsed_steps)
        next_state = state.state
        if state.state == MemoryLifecycleState.CONSOLIDATED and stability_score < 0.55:
            next_state = MemoryLifecycleState.FADING
        elif state.state in (
            MemoryLifecycleState.ACTIVE,
            MemoryLifecycleState.STABILIZING,
        ) and stability_score <= 0.0:
            next_state = MemoryLifecycleState.FADING
        if next_decay_factor <= 0.2:
            next_state = MemoryLifecycleState.ARCHIVED
        return state.with_updates(
            state=next_state,
            stability_score=stability_score,
            decay_factor=next_decay_factor,
        )

    def consolidate(self, state: DomainMemoryState) -> DomainMemoryState:
        return state.with_updates(
            state=MemoryLifecycleState.CONSOLIDATED,
            stability_score=max(0.85, state.stability_score),
            decay_factor=max(0.9, state.decay_factor),
        )
