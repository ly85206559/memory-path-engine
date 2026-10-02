from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "from",
        "with",
        "as",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
    }
)


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


def content_tokens(text: str) -> list[str]:
    """Tokenize while dropping ultra-common function words for BM25 ranking."""
    return [token for token in tokenize(text) if token and token not in _STOPWORDS]


def lexical_overlap(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def bm25_score(
    query: str,
    text: str,
    *,
    avgdl: float,
    doc_freq: dict[str, int],
    doc_count: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Corpus-aware lexical score used for seed ranking on large haystacks."""
    q_tokens = content_tokens(query) or tokenize(query)
    if not q_tokens or doc_count <= 0:
        return 0.0
    t_counts = Counter(content_tokens(text) or tokenize(text))
    if not t_counts:
        return 0.0
    dl = float(sum(t_counts.values()))
    score = 0.0
    for token in dict.fromkeys(q_tokens):
        tf = t_counts.get(token, 0)
        if tf <= 0:
            continue
        df = max(doc_freq.get(token, 0), 1)
        idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
        score += idf * (tf * (k1 + 1.0) / max(denom, 1e-9))
    return max(0.0, score)


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return a deterministic embedding vector for the given text."""

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for many texts (may batch for speed)."""


def embed_many(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    """Batch-embed texts, falling back to per-text ``embed`` when needed."""
    if not texts:
        return []
    embed_many_fn = getattr(provider, "embed_many", None)
    if callable(embed_many_fn):
        return list(embed_many_fn(texts))
    return [provider.embed(text) for text in texts]


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

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class NgramHashingEmbeddingProvider:
    """Richer dependency-free embedder for public-recall product modes.

    Uses signed feature hashing over unigrams, bigrams, and character trigrams
    so paraphrases share more mass than a plain bag-of-hashed-tokens baseline.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = content_tokens(text) or tokenize(text)
        features: list[tuple[str, float]] = [(f"u:{token}", 1.0) for token in tokens]
        for left, right in zip(tokens, tokens[1:]):
            features.append((f"b:{left}_{right}", 0.75))
        compact = re.sub(r"[^a-z0-9]+", " ", text.lower())
        for gram in self._char_ngrams(compact, 3):
            features.append((f"c:{gram}", 0.35))

        for feature, weight in features:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimension
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    @staticmethod
    def _char_ngrams(text: str, size: int) -> list[str]:
        compact = text.replace(" ", "")
        if len(compact) < size:
            return [compact] if compact else []
        return [compact[idx : idx + size] for idx in range(len(compact) - size + 1)]


class FastEmbedEmbeddingProvider:
    """Optional ONNX embedding backend via ``fastembed`` (Product M5/M6).

    Install with ``pip install 'memory-path-engine[embed]'`` or ``pip install fastembed``.
    Selected by ``MPE_EMBEDDING=fastembed`` (default model: ``BAAI/bge-small-en-v1.5``).

    Long session texts are head+tail truncated to ``max_chars`` (BGE context is
    ~512 tokens). Batch size auto-falls back to 1 for very long corpora where
    padding would dominate.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 32,
        max_chars: int = 4000,
        long_text_chars: int = 1500,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised when extra missing
            raise ImportError(
                "FastEmbedEmbeddingProvider requires the optional 'fastembed' package. "
                "Install with: pip install 'memory-path-engine[embed]'"
            ) from exc
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.max_chars = max(256, max_chars)
        self.long_text_chars = max(256, long_text_chars)
        self._model = TextEmbedding(model_name=model_name)
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vectors = self.embed_many([text])
        return vectors[0] if vectors else []

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        pending_idx: list[int] = []
        pending_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[index] = cached
            else:
                pending_idx.append(index)
                pending_texts.append(text)
        if pending_texts:
            # Deduplicate within the batch so identical contents share one encode.
            unique: list[str] = []
            unique_index: dict[str, int] = {}
            for text in pending_texts:
                if text not in unique_index:
                    unique_index[text] = len(unique)
                    unique.append(text)
            prepared = [_truncate_for_dense_embed(text, self.max_chars) for text in unique]
            avg_len = sum(len(text) for text in prepared) / max(len(prepared), 1)
            chunk_size = 1 if avg_len >= self.long_text_chars else self.batch_size
            encoded: list[list[float]] = []
            for start in range(0, len(prepared), chunk_size):
                chunk = prepared[start : start + chunk_size]
                encoded.extend(
                    [[float(value) for value in vector] for vector in self._model.embed(chunk)]
                )
            for index, text in zip(pending_idx, pending_texts):
                vector = encoded[unique_index[text]]
                self._cache[text] = vector
                results[index] = vector
        return [vector or [] for vector in results]


class SentenceTransformerEmbeddingProvider:
    """Optional dense embedder via ``sentence-transformers`` (Product M5/M6)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        batch_size: int = 32,
        max_chars: int = 4000,
        long_text_chars: int = 1500,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SentenceTransformerEmbeddingProvider requires 'sentence-transformers'. "
                "Install with: pip install 'memory-path-engine[embed-st]'"
            ) from exc
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.max_chars = max(256, max_chars)
        self.long_text_chars = max(256, long_text_chars)
        self._model = SentenceTransformer(model_name)
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vectors = self.embed_many([text])
        return vectors[0] if vectors else []

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        pending_idx: list[int] = []
        pending_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[index] = cached
            else:
                pending_idx.append(index)
                pending_texts.append(text)
        if pending_texts:
            unique: list[str] = []
            unique_index: dict[str, int] = {}
            for text in pending_texts:
                if text not in unique_index:
                    unique_index[text] = len(unique)
                    unique.append(text)
            prepared = [_truncate_for_dense_embed(text, self.max_chars) for text in unique]
            avg_len = sum(len(text) for text in prepared) / max(len(prepared), 1)
            chunk_size = 1 if avg_len >= self.long_text_chars else self.batch_size
            encoded: list[list[float]] = []
            for start in range(0, len(prepared), chunk_size):
                chunk = prepared[start : start + chunk_size]
                matrix = self._model.encode(
                    chunk,
                    normalize_embeddings=True,
                    batch_size=min(chunk_size, len(chunk)),
                )
                encoded.extend([[float(value) for value in row] for row in matrix])
            for index, text in zip(pending_idx, pending_texts):
                vector = encoded[unique_index[text]]
                self._cache[text] = vector
                results[index] = vector
        return [vector or [] for vector in results]


def _truncate_for_dense_embed(text: str, max_chars: int) -> str:
    """Keep head+tail for long memories so dense models stay within context."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...\n{text[-tail:]}"


_PROVIDER_CACHE: dict[tuple[str, str | None], EmbeddingProvider] = {}


def clear_embedding_provider_cache() -> None:
    """Drop cached providers (tests / model hot-swap)."""
    _PROVIDER_CACHE.clear()


def resolve_embedding_provider(
    name: str | None = None,
    *,
    model_name: str | None = None,
) -> EmbeddingProvider:
    """Resolve an embedding backend by name or ``MPE_EMBEDDING`` env var.

    Supported names: ``ngram`` (default), ``hash``, ``fastembed``, ``sentence``.
    Optional model override via ``model_name`` or ``MPE_EMBEDDING_MODEL``.

    Heavy backends (fastembed / sentence-transformers) are cached per
    ``(name, model)`` so LongMemEval loops do not reload ONNX/weights per sample.
    """
    import os

    key = (name or os.environ.get("MPE_EMBEDDING") or "ngram").strip().lower()
    model = model_name or os.environ.get("MPE_EMBEDDING_MODEL")
    cache_key = (key, model)
    cached = _PROVIDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if key in {"ngram", "default", "product"}:
        provider: EmbeddingProvider = NgramHashingEmbeddingProvider()
    elif key in {"hash", "hashing"}:
        provider = HashingEmbeddingProvider()
    elif key in {"fastembed", "bge", "bge-small"}:
        max_chars = int(os.environ.get("MPE_EMBEDDING_MAX_CHARS") or "4000")
        provider = FastEmbedEmbeddingProvider(
            model_name=model or "BAAI/bge-small-en-v1.5",
            max_chars=max_chars,
        )
    elif key in {"sentence", "minilm", "sentence-transformers"}:
        max_chars = int(os.environ.get("MPE_EMBEDDING_MAX_CHARS") or "4000")
        provider = SentenceTransformerEmbeddingProvider(
            model_name=model or "sentence-transformers/all-MiniLM-L6-v2",
            max_chars=max_chars,
        )
    else:
        raise ValueError(
            f"Unknown embedding provider '{key}'. "
            "Supported: ngram, hash, fastembed, sentence."
        )
    _PROVIDER_CACHE[cache_key] = provider
    return provider


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(lv * rv for lv, rv in zip(left, right))
