from __future__ import annotations

from dataclasses import dataclass, replace

from memory_engine.memory.domain.enums import MemoryLifecycleState


@dataclass(frozen=True, slots=True)
class DomainMemoryState:
    state: MemoryLifecycleState = MemoryLifecycleState.ENCODED
    reinforcement_count: int = 0
    last_accessed_at: str | None = None
    stability_score: float = 0.0
    decay_factor: float = 1.0

    def with_updates(
        self,
        *,
        state: MemoryLifecycleState | None = None,
        reinforcement_count: int | None = None,
        last_accessed_at: str | None = None,
        stability_score: float | None = None,
        decay_factor: float | None = None,
    ) -> "DomainMemoryState":
        next_state = replace(
            self,
            state=state if state is not None else self.state,
            reinforcement_count=(
                reinforcement_count
                if reinforcement_count is not None
                else self.reinforcement_count
            ),
            last_accessed_at=last_accessed_at if last_accessed_at is not None else self.last_accessed_at,
            stability_score=stability_score if stability_score is not None else self.stability_score,
            decay_factor=decay_factor if decay_factor is not None else self.decay_factor,
        )
        if not 0.0 <= next_state.stability_score <= 1.0:
            raise ValueError("stability_score must be between 0.0 and 1.0")
        if not 0.0 <= next_state.decay_factor <= 1.0:
            raise ValueError("decay_factor must be between 0.0 and 1.0")
        return next_state
