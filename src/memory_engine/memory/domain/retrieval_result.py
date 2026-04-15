from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_engine.schema import RetrievalResult


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    memory_id: str
    score: float
    reason: str
    retrieval_role: str = "seed"
    source_path: str | None = None
    memory_kind: str | None = None
    lifecycle_state: str | None = None
    consolidation_kind: str | None = None


@dataclass(frozen=True, slots=True)
class RecallRoute:
    route_id: str
    route_kind: str
    step_memory_ids: tuple[str, ...]
    support_memory_ids: tuple[str, ...] = ()
    score: float = 0.0
    route_source: str = "legacy_path"


@dataclass(frozen=True, slots=True)
class ActivationSnapshotEntry:
    memory_id: str
    source_memory_id: str | None = None
    edge_type: str | None = None
    hop: int = 0
    incoming_activation: float = 0.0
    propagated_activation: float = 0.0
    activated_score: float | None = None
    stopped_reason: str | None = None
    is_seed: bool = False


@dataclass(frozen=True, slots=True)
class ActivationSnapshot:
    steps: tuple[ActivationSnapshotEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class PalaceRecallResult:
    query: str
    retrieved_memories: tuple[RetrievedMemory, ...] = ()
    routes: tuple[RecallRoute, ...] = ()
    activation_snapshot: ActivationSnapshot = field(default_factory=ActivationSnapshot)
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def retrieved_ids(self, *, top_k: int | None = None) -> list[str]:
        items = [item.memory_id for item in self.retrieved_memories]
        if top_k is None:
            return items
        return items[:top_k]

    def to_legacy_retrieval_result(self) -> "RetrievalResult":
        from memory_engine.schema import MemoryPath, RetrievalResult

        paths: list[MemoryPath] = []
        for route in self.routes:
            steps = []
            for memory_id in route.step_memory_ids:
                retrieved = next(
                    (item for item in self.retrieved_memories if item.memory_id == memory_id),
                    None,
                )
                if retrieved is None:
                    continue
                from memory_engine.schema import PathStep

                steps.append(
                    PathStep(
                        node_id=memory_id,
                        reason=retrieved.reason,
                        score=retrieved.score,
                    )
                )
            if steps:
                paths.append(
                    MemoryPath(
                        query=self.query,
                        steps=steps,
                        final_answer=self.final_answer or "",
                        final_score=route.score,
                    )
                )
        result = RetrievalResult(query=self.query, paths=paths)
        result.palace_result = self
        return result

    @classmethod
    def from_legacy_result(cls, result: "RetrievalResult") -> "PalaceRecallResult":
        from memory_engine.schema import RetrievalResult

        if not isinstance(result, RetrievalResult):
            raise TypeError("result must be a RetrievalResult")
        if result.palace_result is not None:
            return result.palace_result

        retrieved_memories: list[RetrievedMemory] = []
        seen: set[str] = set()
        routes: list[RecallRoute] = []
        activation_steps: list[ActivationSnapshotEntry] = []
        for path_index, path in enumerate(sorted(result.paths, key=lambda item: item.final_score, reverse=True)):
            step_ids = tuple(step.node_id for step in path.steps)
            support_ids = tuple(
                evidence.section_id or evidence.source_path
                for evidence in path.supporting_evidence
            )
            routes.append(
                RecallRoute(
                    route_id=f"legacy-route-{path_index}",
                    route_kind="legacy_path",
                    step_memory_ids=step_ids,
                    support_memory_ids=support_ids,
                    score=path.final_score,
                    route_source="legacy_path",
                )
            )
            for step_index, step in enumerate(path.steps):
                if step.node_id not in seen:
                    seen.add(step.node_id)
                    retrieved_memories.append(
                        RetrievedMemory(
                            memory_id=step.node_id,
                            score=step.score,
                            reason=step.reason,
                            retrieval_role="seed" if step_index == 0 else "support",
                        )
                    )
            for trace_step in path.activation_trace:
                activation_steps.append(
                    ActivationSnapshotEntry(
                        memory_id=trace_step.node_id,
                        source_memory_id=trace_step.source_node_id,
                        edge_type=trace_step.edge_type,
                        hop=trace_step.hop,
                        incoming_activation=trace_step.incoming_activation,
                        propagated_activation=trace_step.propagated_activation,
                        activated_score=trace_step.activated_score,
                        stopped_reason=trace_step.stopped_reason,
                        is_seed=trace_step.is_seed,
                    )
                )
        return cls(
            query=result.query,
            retrieved_memories=tuple(retrieved_memories),
            routes=tuple(routes),
            activation_snapshot=ActivationSnapshot(tuple(activation_steps)),
            final_answer=result.best_path().final_answer if result.paths else None,
        )
