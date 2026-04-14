from __future__ import annotations

from dataclasses import dataclass, field

from memory_engine.memory.domain.memory_state_machine import MemoryStateMachine
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.retrieval_result import PalaceRecallResult


@dataclass(slots=True)
class ReinforceMemoryService:
    state_machine: MemoryStateMachine = field(default_factory=MemoryStateMachine)

    def reinforce_recall_result(
        self,
        palace: MemoryPalace,
        result: PalaceRecallResult,
    ) -> MemoryPalace:
        touched_ids = {item.memory_id for item in result.retrieved_memories}
        for memory_id, memory in palace.memories.items():
            if memory_id in touched_ids:
                memory.state = self.state_machine.reinforce(memory.state)
            else:
                memory.state = self.state_machine.decay(memory.state)
        return palace
