from __future__ import annotations

"""
Build legacy graph retrievers from a :class:`MemoryStore`.

Kept separate from :mod:`memory_engine.benchmarking.application.service` so
palace-layer code (e.g. :class:`RetrieveMemoryService`) can construct retrievers
without importing the benchmark runner stack.
"""

from dataclasses import dataclass, field

from memory_engine.memory_state import MemoryStatePolicy, StaticMemoryStatePolicy
from memory_engine.retrieve import (
    ActivationSpreadingRetriever,
    BaselineTopKRetriever,
    EmbeddingTopKRetriever,
    StructureAwareRetriever,
    WeightedGraphRetriever,
)
from memory_engine.store import MemoryStore


@dataclass(slots=True)
class LegacyModeRetriever:
    retriever_mode: str
    store: MemoryStore
    memory_state_policy: MemoryStatePolicy
    _delegate: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        builder = _retriever_builders()[self.retriever_mode]
        self._delegate = builder(self.store, memory_state_policy=self.memory_state_policy)

    def search(self, query: str, top_k: int = 3, **kwargs):
        return self._delegate.search(query, top_k=top_k, **kwargs)


def _retriever_builders():
    return {
        "lexical_baseline": BaselineTopKRetriever,
        "embedding_baseline": EmbeddingTopKRetriever,
        "structure_only": StructureAwareRetriever,
        "weighted_graph": WeightedGraphRetriever,
        "activation_spreading_v1": ActivationSpreadingRetriever,
        "weighted_graph_static": WeightedGraphRetriever,
        "weighted_graph_dynamic": WeightedGraphRetriever,
        "activation_spreading_static": ActivationSpreadingRetriever,
        "activation_spreading_dynamic": ActivationSpreadingRetriever,
    }


def build_legacy_retriever(retriever_mode: str, store: MemoryStore):
    memory_policies: dict[str, MemoryStatePolicy] = {
        "lexical_baseline": StaticMemoryStatePolicy(),
        "embedding_baseline": StaticMemoryStatePolicy(),
        "structure_only": StaticMemoryStatePolicy(),
        "weighted_graph": MemoryStatePolicy(),
        "activation_spreading_v1": MemoryStatePolicy(),
        "weighted_graph_static": StaticMemoryStatePolicy(),
        "weighted_graph_dynamic": MemoryStatePolicy(),
        "activation_spreading_static": StaticMemoryStatePolicy(),
        "activation_spreading_dynamic": MemoryStatePolicy(),
    }
    retriever_builders = _retriever_builders()
    try:
        retriever_builders[retriever_mode]
        memory_state_policy = memory_policies[retriever_mode]
    except KeyError as exc:
        available = ", ".join(sorted(retriever_builders))
        raise ValueError(
            f"Unknown retriever mode '{retriever_mode}'. Available: {available}"
        ) from exc
    return LegacyModeRetriever(
        retriever_mode=retriever_mode,
        store=store,
        memory_state_policy=memory_state_policy,
    )
