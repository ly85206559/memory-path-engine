from __future__ import annotations

"""
Build legacy graph retrievers from a :class:`MemoryStore`.

Kept separate from :mod:`memory_engine.benchmarking.application.service` so
palace-layer code (e.g. :class:`RetrieveMemoryService`) can construct retrievers
without importing the benchmark runner stack.
"""

from dataclasses import dataclass, field

from memory_engine.embeddings import (
    EmbeddingProvider,
    resolve_embedding_provider,
)
from memory_engine.memory_state import MemoryStatePolicy, StaticMemoryStatePolicy
from memory_engine.retrieve import (
    ActivationSpreadingRetriever,
    BaselineTopKRetriever,
    EmbeddingTopKRetriever,
    HybridRetriever,
    StructureAwareRetriever,
    WeightedGraphRetriever,
)
from memory_engine.store import MemoryStore


@dataclass(slots=True)
class LegacyModeRetriever:
    retriever_mode: str
    store: MemoryStore
    memory_state_policy: MemoryStatePolicy
    embedding_provider: EmbeddingProvider | None = None
    _delegate: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        builder = _retriever_builders()[self.retriever_mode]
        kwargs = {"memory_state_policy": self.memory_state_policy}
        if self.embedding_provider is not None and self.retriever_mode in {
            "embedding_baseline",
            "hybrid",
            "weighted_graph",
            "weighted_graph_static",
            "weighted_graph_dynamic",
            "activation_spreading_v1",
            "activation_spreading_static",
            "activation_spreading_dynamic",
            "structure_only",
        }:
            kwargs["embedding_provider"] = self.embedding_provider
        self._delegate = builder(self.store, **kwargs)

    def search(self, query: str, top_k: int = 3, **kwargs):
        return self._delegate.search(query, top_k=top_k, **kwargs)


def product_embedding_provider(
    name: str | None = None,
) -> EmbeddingProvider:
    """Product modes (hybrid / embedding_baseline) resolve via env/name."""
    return resolve_embedding_provider(name)


def _embedding_baseline_retriever(
    store: MemoryStore,
    *,
    memory_state_policy: MemoryStatePolicy,
    embedding_provider: EmbeddingProvider | None = None,
) -> EmbeddingTopKRetriever:
    return EmbeddingTopKRetriever(
        store,
        embedding_provider=embedding_provider or product_embedding_provider(),
        memory_state_policy=memory_state_policy,
    )


def _hybrid_retriever(
    store: MemoryStore,
    *,
    memory_state_policy: MemoryStatePolicy,
    embedding_provider: EmbeddingProvider | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        store,
        embedding_provider=embedding_provider or product_embedding_provider(),
        memory_state_policy=memory_state_policy,
    )


def _retriever_builders():
    return {
        "lexical_baseline": BaselineTopKRetriever,
        "embedding_baseline": _embedding_baseline_retriever,
        "structure_only": StructureAwareRetriever,
        "weighted_graph": WeightedGraphRetriever,
        "hybrid": _hybrid_retriever,
        "activation_spreading_v1": ActivationSpreadingRetriever,
        "weighted_graph_static": WeightedGraphRetriever,
        "weighted_graph_dynamic": WeightedGraphRetriever,
        "activation_spreading_static": ActivationSpreadingRetriever,
        "activation_spreading_dynamic": ActivationSpreadingRetriever,
    }


def build_legacy_retriever(
    retriever_mode: str,
    store: MemoryStore,
    *,
    embedding: str | None = None,
):
    memory_policies: dict[str, MemoryStatePolicy] = {
        "lexical_baseline": StaticMemoryStatePolicy(),
        "embedding_baseline": StaticMemoryStatePolicy(),
        "structure_only": StaticMemoryStatePolicy(),
        "weighted_graph": MemoryStatePolicy(),
        "hybrid": MemoryStatePolicy(),
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

    provider: EmbeddingProvider | None = None
    # Product semantic modes always resolve via MPE_EMBEDDING / --embedding.
    # Other modes keep Layer-B-stable hashing defaults unless embedding is explicit.
    if retriever_mode in {"embedding_baseline", "hybrid"}:
        provider = product_embedding_provider(embedding)
    elif embedding is not None:
        provider = product_embedding_provider(embedding)

    return LegacyModeRetriever(
        retriever_mode=retriever_mode,
        store=store,
        memory_state_policy=memory_state_policy,
        embedding_provider=provider,
    )