from __future__ import annotations

from dataclasses import dataclass

from memory_engine.memory.application.bridge import palace_to_store
from memory_engine.retrieval_factory import build_legacy_retriever
from memory_engine.memory.application.query_models import RecallQuery
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.retrieval_result import PalaceRecallResult
from memory_engine.schema import ActivationContext


@dataclass(slots=True)
class RetrieveMemoryService:
    def recall(self, palace: MemoryPalace, query: RecallQuery) -> PalaceRecallResult:
        store = palace_to_store(palace)
        retriever = build_legacy_retriever(query.policy.retriever_mode, store)
        try:
            result = retriever.search(
                query.text,
                top_k=query.policy.top_k,
                context=ActivationContext(
                    query=query.text,
                    max_hops=query.policy.max_hops,
                ),
            )
        except TypeError:
            result = retriever.search(
                query.text,
                top_k=query.policy.top_k,
            )
        return PalaceRecallResult.from_legacy_result(result)
