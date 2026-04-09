from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


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


def tokenize(text: str) -> list[str]:
    return [normalize_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def lexical_overlap(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return a deterministic embedding vector for the given text."""


class HashingEmbeddingProvider:
    """A dependency-free local embedder for experiments and tests."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimension
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(lv * rv for lv, rv in zip(left, right))
