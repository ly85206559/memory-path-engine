from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from memory_engine.embeddings import HashingEmbeddingProvider, cosine_similarity, lexical_overlap
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.ranking import RankedRecallBundle
from memory_engine.memory.domain.retrieval_result import RecallRoute, RetrievedMemory
from memory_engine.memory.domain.seed_selection import SeedActivation
from memory_engine.memory_state import MemoryStatePolicy


class RecallResultRanker(Protocol):
    def rank(
        self,
        palace: MemoryPalace,
        query_text: str,
        seeds: tuple[SeedActivation, ...],
        activated_memories: tuple[RetrievedMemory, ...],
        routes: tuple[RecallRoute, ...],
        *,
        top_k: int,
    ) -> RankedRecallBundle:
        ...


@dataclass(slots=True)
class DefaultRecallResultRanker:
    """Fuse seeds and route steps into a deduplicated ranked memory list."""

    state_policy: MemoryStatePolicy = field(default_factory=MemoryStatePolicy)
    embedding_provider: HashingEmbeddingProvider = field(default_factory=HashingEmbeddingProvider)

    def rank(
        self,
        palace: MemoryPalace,
        query_text: str,
        seeds: tuple[SeedActivation, ...],
        activated_memories: tuple[RetrievedMemory, ...],
        routes: tuple[RecallRoute, ...],
        *,
        top_k: int,
    ) -> RankedRecallBundle:
        query_embedding = self.embedding_provider.embed(query_text)
        by_id: dict[str, RetrievedMemory] = {}
        route_alignments: dict[str, float] = {}

        for seed in seeds:
            mem = palace.memories.get(seed.memory_id)
            kind = mem.kind.value if mem is not None else None
            lifecycle_state = mem.state.state.value if mem is not None else None
            consolidation_kind = (
                str(mem.metadata.get("consolidation_kind"))
                if mem is not None and mem.metadata.get("consolidation_kind") is not None
                else None
            )
            score = (
                min(1.0, seed.score * self.state_policy.recall_multiplier_for_state(mem.state))
                if mem is not None
                else seed.score
            )
            if mem is not None:
                score = self._reranked_score(
                    query_text,
                    query_embedding,
                    content=mem.content,
                    base_score=score,
                )
            prev = by_id.get(seed.memory_id)
            candidate = RetrievedMemory(
                memory_id=seed.memory_id,
                score=score,
                reason=seed.reason,
                retrieval_role="seed",
                memory_kind=kind,
                lifecycle_state=lifecycle_state,
                consolidation_kind=consolidation_kind,
            )
            if prev is None or candidate.score > prev.score:
                by_id[seed.memory_id] = candidate

        for activated in activated_memories:
            mem = palace.memories.get(activated.memory_id)
            score = activated.score
            if mem is not None:
                score = self._reranked_score(
                    query_text,
                    query_embedding,
                    content=mem.content,
                    base_score=activated.score,
                )
                activated = replace(activated, score=score)
            prev = by_id.get(activated.memory_id)
            if prev is None or activated.score > prev.score:
                by_id[activated.memory_id] = activated

        reranked_routes = []
        for route in routes:
            route_alignment = self._route_alignment(palace, query_text, query_embedding, route)
            route_alignments[route.route_id] = route_alignment
            reranked_routes.append(
                replace(
                    route,
                    score=self._reranked_route_score(route.score, route_alignment),
                )
            )

        for route in reranked_routes:
            for idx, mid in enumerate(route.step_memory_ids):
                mem = palace.memories.get(mid)
                kind = mem.kind.value if mem is not None else None
                lifecycle_state = mem.state.state.value if mem is not None else None
                consolidation_kind = (
                    str(mem.metadata.get("consolidation_kind"))
                    if mem is not None and mem.metadata.get("consolidation_kind") is not None
                    else None
                )
                step_score = route.score * (1.0 / (1 + idx))
                if mem is not None:
                    step_score = min(
                        1.0,
                        step_score * self.state_policy.recall_multiplier_for_state(mem.state),
                    )
                    step_score = self._reranked_score(
                        query_text,
                        query_embedding,
                        content=mem.content,
                        base_score=step_score,
                    )
                candidate = RetrievedMemory(
                    memory_id=mid,
                    score=step_score,
                    reason=f"route:{route.route_id}",
                    retrieval_role="support" if idx > 0 else "seed",
                    memory_kind=kind,
                    lifecycle_state=lifecycle_state,
                    consolidation_kind=consolidation_kind,
                )
                prev = by_id.get(mid)
                if prev is None or candidate.score > prev.score:
                    by_id[mid] = candidate

        ranked = sorted(
            by_id.values(),
            key=lambda item: (
                item.score,
                self._memory_alignment(
                    query_text,
                    query_embedding,
                    palace.memories.get(item.memory_id).content
                    if palace.memories.get(item.memory_id) is not None
                    else "",
                ),
            ),
            reverse=True,
        )
        retrieved = tuple(ranked[: max(1, top_k)])
        ordered_routes = tuple(
            sorted(
                reranked_routes,
                key=lambda route: (
                    route.score,
                    route_alignments.get(route.route_id, 0.0),
                    -len(route.step_memory_ids),
                ),
                reverse=True,
            )
        )
        return RankedRecallBundle(retrieved_memories=retrieved, routes=ordered_routes)

    def _reranked_score(
        self,
        query_text: str,
        query_embedding: list[float],
        *,
        content: str,
        base_score: float,
    ) -> float:
        alignment = self._memory_alignment(query_text, query_embedding, content)
        return base_score + 0.35 * alignment

    def _reranked_route_score(self, base_score: float, alignment: float) -> float:
        return base_score + 0.25 * alignment

    def _memory_alignment(
        self,
        query_text: str,
        query_embedding: list[float],
        content: str,
    ) -> float:
        if not content:
            return 0.0
        lexical = lexical_overlap(query_text, content)
        semantic = cosine_similarity(query_embedding, self.embedding_provider.embed(content))
        return max(lexical, semantic)

    def _route_alignment(
        self,
        palace: MemoryPalace,
        query_text: str,
        query_embedding: list[float],
        route: RecallRoute,
    ) -> float:
        alignments = [
            self._memory_alignment(query_text, query_embedding, palace.memories[mid].content)
            for mid in route.step_memory_ids
            if mid in palace.memories
        ]
        if not alignments:
            return 0.0
        return max(alignments)
