from __future__ import annotations

"""
Unified recall entry points for the dual-track architecture.

Callers should prefer these helpers over reaching into both stacks ad hoc:

- ``recall_from_documents`` / ``recall_from_store``: legacy graph path with
  optional palace projection on the result
- ``recall_from_palace``: palace-first orchestration (`RetrieveMemoryService`)

Both tracks share retriever modes via ``build_legacy_retriever`` and the same
``MemoryStore`` bridge (`palace_to_store` / `store_to_palace`).
"""

from dataclasses import dataclass
from pathlib import Path

from memory_engine.domain_pack import DomainPack, get_domain_pack
from memory_engine.ingest import ingest_document
from memory_engine.memory.application.bridge import store_to_palace
from memory_engine.memory.application.query_models import RecallPolicy, RecallQuery
from memory_engine.memory.application.retrieve_memory_service import RetrieveMemoryService
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.retrieval_result import PalaceRecallResult
from memory_engine.retrieval_factory import build_legacy_retriever
from memory_engine.schema import RetrievalResult
from memory_engine.store import MemoryStore


@dataclass(frozen=True, slots=True)
class UnifiedRecallResult:
    """Normalized dual-track recall payload."""

    track: str
    retriever_mode: str
    legacy: RetrievalResult | None = None
    palace: PalaceRecallResult | None = None

    @property
    def best_answer(self) -> str:
        if self.legacy is not None and self.legacy.paths:
            return self.legacy.best_path().final_answer
        if self.palace is not None:
            if self.palace.final_answer:
                return self.palace.final_answer
            if self.palace.retrieved_memories:
                return self.palace.retrieved_memories[0].memory_id
        return ""


def ingest_paths(
    paths: list[Path] | tuple[Path, ...],
    *,
    domain_pack: str | DomainPack,
    store: MemoryStore | None = None,
) -> MemoryStore:
    resolved_store = store or MemoryStore()
    for path in paths:
        ingest_document(path, resolved_store, domain_pack=domain_pack)
    return resolved_store


def recall_from_store(
    store: MemoryStore,
    query: str,
    *,
    retriever_mode: str = "weighted_graph",
    top_k: int = 3,
    project_palace: bool = True,
) -> UnifiedRecallResult:
    """Recall on the legacy graph track; optionally attach a palace projection."""
    retriever = build_legacy_retriever(retriever_mode, store)
    legacy = retriever.search(query, top_k=top_k)
    palace_result = legacy.palace_result if project_palace else None
    if project_palace and palace_result is None:
        # Ensure callers always see a palace-shaped projection when requested.
        palace = store_to_palace(store, palace_id="unified-recall")
        palace_result = RetrieveMemoryService().recall(
            palace,
            RecallQuery(
                palace_id=palace.palace_id,
                text=query,
                policy=RecallPolicy(
                    retriever_mode=retriever_mode,
                    top_k=top_k,
                    allow_legacy_fallback=True,
                ),
            ),
        )
    return UnifiedRecallResult(
        track="legacy",
        retriever_mode=retriever_mode,
        legacy=legacy,
        palace=palace_result,
    )


def recall_from_palace(
    palace: MemoryPalace,
    query: str,
    *,
    retriever_mode: str = "weighted_graph",
    top_k: int = 3,
    max_hops: int = 3,
    max_spaces: int = 3,
    max_seeds: int = 5,
    allow_legacy_fallback: bool = True,
    service: RetrieveMemoryService | None = None,
) -> UnifiedRecallResult:
    """Recall on the palace-first track."""
    resolved_service = service or RetrieveMemoryService()
    palace_result = resolved_service.recall(
        palace,
        RecallQuery(
            palace_id=palace.palace_id,
            text=query,
            policy=RecallPolicy(
                retriever_mode=retriever_mode,
                top_k=top_k,
                max_hops=max_hops,
                max_spaces=max_spaces,
                max_seeds=max_seeds,
                allow_legacy_fallback=allow_legacy_fallback,
            ),
        ),
    )
    return UnifiedRecallResult(
        track="palace",
        retriever_mode=retriever_mode,
        legacy=palace_result.to_legacy_retrieval_result(),
        palace=palace_result,
    )


def recall_from_documents(
    document_dir: Path,
    query: str,
    *,
    domain_pack: str | DomainPack,
    retriever_mode: str = "weighted_graph",
    top_k: int = 3,
    track: str = "legacy",
    glob_pattern: str = "*.md",
) -> UnifiedRecallResult:
    """
    Ingest a document directory and recall with a chosen track.

    ``track='legacy'`` uses graph retrievers directly.
    ``track='palace'`` bridges the store into a palace then runs palace recall.
    """
    if track not in {"legacy", "palace"}:
        raise ValueError("track must be 'legacy' or 'palace'")
    pack_name = domain_pack if isinstance(domain_pack, str) else domain_pack.name
    # Validate pack early for clearer errors.
    get_domain_pack(pack_name if isinstance(domain_pack, str) else domain_pack.name)
    paths = sorted(Path(document_dir).glob(glob_pattern))
    store = ingest_paths(paths, domain_pack=domain_pack)
    if track == "legacy":
        return recall_from_store(
            store,
            query,
            retriever_mode=retriever_mode,
            top_k=top_k,
            project_palace=True,
        )
    palace = store_to_palace(store, palace_id=f"docs:{Path(document_dir).name}")
    return recall_from_palace(
        palace,
        query,
        retriever_mode=retriever_mode,
        top_k=top_k,
    )
