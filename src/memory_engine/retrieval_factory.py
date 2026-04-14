from __future__ import annotations

"""
Build legacy graph retrievers from a :class:`MemoryStore`.

Kept separate from :mod:`memory_engine.benchmarking.application.service` so
palace-layer code (e.g. :class:`RetrieveMemoryService`) can construct retrievers
without importing the benchmark runner stack.
"""

from memory_engine.memory_state import MemoryStatePolicy, StaticMemoryStatePolicy
from memory_engine.retrieve import (
    ActivationSpreadingRetriever,
    BaselineTopKRetriever,
    EmbeddingTopKRetriever,
    StructureAwareRetriever,
    WeightedGraphRetriever,
)
from memory_engine.store import MemoryStore


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
    retriever_builders = {
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
    try:
        retriever_builder = retriever_builders[retriever_mode]
        memory_state_policy = memory_policies[retriever_mode]
    except KeyError as exc:
        available = ", ".join(sorted(retriever_builders))
        raise ValueError(
            f"Unknown retriever mode '{retriever_mode}'. Available: {available}"
        ) from exc
    return retriever_builder(store, memory_state_policy=memory_state_policy)
