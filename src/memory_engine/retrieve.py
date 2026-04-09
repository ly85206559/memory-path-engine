from __future__ import annotations

from collections import deque
import re
import time

from memory_engine.replay import path_answer
from memory_engine.schema import ActivationContext, RetrievalResult
from memory_engine.store import MemoryStore


def tokenize(text: str) -> set[str]:
    return {normalize_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())}


def normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def lexical_overlap(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


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


class WeightedGraphRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def search(self, query: str, top_k: int = 3, context: ActivationContext | None = None) -> RetrievalResult:
        context = context or ActivationContext(query=query)
        start = time.perf_counter()

        ranked = sorted(
            self.store.nodes(),
            key=lambda node: self._score_node(query, node.content, node.weights.bounded_score(), False, context),
            reverse=True,
        )
        seeds = ranked[:top_k]

        paths = []
        for seed in seeds:
            path = self._expand_from_seed(query, seed.id, context)
            if path is not None:
                paths.append(path)

        elapsed_ms = (time.perf_counter() - start) * 1000
        for path in paths:
            path.final_answer = f"{path.final_answer} [latency_ms={elapsed_ms:.2f}]"
        return RetrievalResult(query=query, paths=paths)

    def _expand_from_seed(self, query: str, seed_id: str, context: ActivationContext):
        visited = {seed_id}
        queue = deque([(seed_id, 0, None)])
        chain: list[tuple] = []

        while queue:
            node_id, depth, via_edge = queue.popleft()
            node = self.store.get_node(node_id)
            anomaly = node.weights.risk > 0.8 or node.weights.novelty > 0.8
            score = self._score_node(query, node.content, node.weights.bounded_score(), anomaly, context, depth)
            reason = "seed hit" if depth == 0 else f"expanded at hop {depth}"
            chain.append((node, score, reason, via_edge))

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

    def _score_node(
        self,
        query: str,
        text: str,
        importance_score: float,
        anomaly: bool,
        context: ActivationContext,
        depth: int = 0,
    ) -> float:
        semantic_score = lexical_overlap(query, text)
        structural_score = max(0.0, 1.0 - depth * 0.25)
        anomaly_score = 1.0 if anomaly else 0.0
        return (
            semantic_score * context.semantic_weight
            + structural_score * context.structural_weight
            + anomaly_score * context.anomaly_weight
            + importance_score * context.importance_weight
        )
