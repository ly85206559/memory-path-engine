from __future__ import annotations

from collections import deque
import time

from memory_engine.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity,
    lexical_overlap,
)
from memory_engine.replay import path_answer
from memory_engine.schema import ActivationContext, RetrievalResult
from memory_engine.scoring import ScoringStrategy, StructureOnlyScoringStrategy, WeightedSumScoringStrategy
from memory_engine.store import MemoryStore


class BaselineTopKRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        ranked = sorted(
            self.store.nodes(),
            key=lambda node: lexical_overlap(query, node.content),
            reverse=True,
        )[:top_k]
        paths = [
            path_answer(query, [(node, lexical_overlap(query, node.content), "baseline lexical hit", None)])
            for node in ranked
            if lexical_overlap(query, node.content) > 0
        ]
        return RetrievalResult(query=query, paths=paths)


class EmbeddingTopKRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self._embedding_cache: dict[str, list[float]] = {}

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        ranked = self.rank_candidates(query, top_k=top_k)
        paths = [
            path_answer(query, [(node, score, "embedding semantic hit", None)])
            for node, score in ranked
            if score > 0
        ]
        return RetrievalResult(query=query, paths=paths)

    def rank_candidates(self, query: str, top_k: int = 3) -> list[tuple]:
        query_embedding = self._embed(query)
        ranked = sorted(
            (
                (node, cosine_similarity(query_embedding, self._embed(node.content)))
                for node in self.store.nodes()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]

    def semantic_similarity(self, query: str, text: str) -> float:
        return cosine_similarity(self._embed(query), self._embed(text))

    def _embed(self, text: str) -> list[float]:
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.embedding_provider.embed(text)
        return self._embedding_cache[text]


class WeightedGraphRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
        scoring_strategy: ScoringStrategy | None = None,
    ) -> None:
        self.store = store
        self.embedding_retriever = EmbeddingTopKRetriever(
            store=store,
            embedding_provider=embedding_provider,
        )
        self.scoring_strategy = scoring_strategy or WeightedSumScoringStrategy()

    def search(self, query: str, top_k: int = 3, context: ActivationContext | None = None) -> RetrievalResult:
        context = context or ActivationContext(query=query)
        start = time.perf_counter()
        seeds = self._rank_seed_candidates(query, top_k=top_k)

        paths = []
        for seed, seed_similarity in seeds:
            path = self._expand_from_seed(
                query=query,
                seed_id=seed.id,
                seed_similarity=seed_similarity,
                context=context,
            )
            if path is not None:
                paths.append(path)

        elapsed_ms = (time.perf_counter() - start) * 1000
        for path in paths:
            path.final_answer = f"{path.final_answer} [latency_ms={elapsed_ms:.2f}]"
        return RetrievalResult(query=query, paths=paths)

    def _rank_seed_candidates(self, query: str, top_k: int) -> list[tuple]:
        combined: dict[str, tuple] = {}

        for node, score in self.embedding_retriever.rank_candidates(query, top_k=top_k):
            combined[node.id] = (node, score)

        for node in self._lexical_rank_candidates(query, top_k=top_k):
            score = self._semantic_similarity(query, node.content)
            existing = combined.get(node.id)
            if existing is None or score > existing[1]:
                combined[node.id] = (node, score)

        ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
        return ranked[: max(top_k, min(len(ranked), top_k * 2))]

    def _lexical_rank_candidates(self, query: str, top_k: int) -> list:
        return sorted(
            self.store.nodes(),
            key=lambda node: lexical_overlap(query, node.content),
            reverse=True,
        )[:top_k]

    def _semantic_similarity(self, query: str, text: str) -> float:
        embedding_score = self.embedding_retriever.semantic_similarity(query, text)
        lexical_score = lexical_overlap(query, text)
        return max(embedding_score, lexical_score)

    def _expand_from_seed(
        self,
        query: str,
        seed_id: str,
        seed_similarity: float,
        context: ActivationContext,
    ):
        visited = {seed_id}
        queue = deque([(seed_id, 0, None)])
        chain: list[tuple] = []

        while queue:
            node_id, depth, via_edge = queue.popleft()
            node = self.store.get_node(node_id)
            semantic_score = (
                seed_similarity
                if depth == 0
                else self._semantic_similarity(query, node.content)
            )
            breakdown = self.scoring_strategy.score_node(
                query=query,
                node=node,
                semantic_score=semantic_score,
                context=context,
                depth=depth,
            )
            reason = (
                f"seed hit semantic={breakdown.semantic_score:.3f}"
                if depth == 0
                else f"expanded at hop {depth} total={breakdown.total_score:.3f}"
            )
            chain.append((node, breakdown.total_score, reason, via_edge))

            if depth >= context.max_hops:
                continue

            for edge in self.store.neighbors(node_id):
                if edge.to_id in visited:
                    continue
                visited.add(edge.to_id)
                queue.append((edge.to_id, depth + 1, edge.edge_type))

        chain.sort(key=lambda item: item[1], reverse=True)
        if not chain:
            return None
        return path_answer(query, chain[:3])


class StructureAwareRetriever(WeightedGraphRetriever):
    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        super().__init__(
            store=store,
            embedding_provider=embedding_provider,
            scoring_strategy=StructureOnlyScoringStrategy(),
        )
