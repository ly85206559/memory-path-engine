from __future__ import annotations

from dataclasses import dataclass, replace

from memory_engine.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity,
    lexical_overlap,
)
from memory_engine.memory.domain.enums import MemoryKind
from memory_engine.memory.domain.memory_types import Memory, RouteMemory
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.seed_selection import (
    SeedActivation,
    SeedSelectionInput,
    SeedSelector,
)


def _memory_space_id(memory: Memory) -> str:
    return str(memory.metadata.get("space_id", memory.location.as_key() or ""))


def _iter_seed_memories(
    palace: MemoryPalace,
    selection: SeedSelectionInput,
) -> list[Memory]:
    allowed_spaces = set(selection.allowed_space_ids)
    kinds_filter = set(selection.allowed_memory_kinds) if selection.allowed_memory_kinds else None
    out: list[Memory] = []
    for memory in palace.memories.values():
        if isinstance(memory, RouteMemory):
            continue
        if kinds_filter is not None and memory.kind not in kinds_filter:
            continue
        sid = _memory_space_id(memory)
        if allowed_spaces and sid not in allowed_spaces:
            continue
        out.append(memory)
    return out


@dataclass(slots=True)
class EmbeddingSeedSelector:
    embedder: EmbeddingProvider

    def select_seeds(
        self,
        palace: MemoryPalace,
        selection: SeedSelectionInput,
    ) -> tuple[SeedActivation, ...]:
        memories = _iter_seed_memories(palace, selection)
        if not memories:
            return ()

        q_vec = self.embedder.embed(selection.text)
        scored: list[SeedActivation] = []
        for memory in memories:
            vec = self.embedder.embed(memory.content)
            score = cosine_similarity(q_vec, vec)
            scored.append(
                SeedActivation(
                    memory_id=memory.memory_id,
                    score=score,
                    reason="embedding_cosine",
                    space_id=_memory_space_id(memory) or None,
                ),
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        positive = [s for s in scored if s.score > 0.0]
        if not positive:
            return ()
        return tuple(positive[: max(1, selection.max_seeds)])


@dataclass(slots=True)
class LexicalSeedSelector:
    def select_seeds(
        self,
        palace: MemoryPalace,
        selection: SeedSelectionInput,
    ) -> tuple[SeedActivation, ...]:
        memories = _iter_seed_memories(palace, selection)
        if not memories:
            return ()

        scored: list[SeedActivation] = []
        for memory in memories:
            score = lexical_overlap(selection.text, memory.content)
            scored.append(
                SeedActivation(
                    memory_id=memory.memory_id,
                    score=score,
                    reason="lexical_overlap",
                    space_id=_memory_space_id(memory) or None,
                ),
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        positive = [s for s in scored if s.score > 0.0]
        if not positive:
            return ()
        return tuple(positive[: max(1, selection.max_seeds)])


@dataclass(slots=True)
class HybridSeedSelector:
    embedding_selector: EmbeddingSeedSelector
    lexical_selector: LexicalSeedSelector

    def select_seeds(
        self,
        palace: MemoryPalace,
        selection: SeedSelectionInput,
    ) -> tuple[SeedActivation, ...]:
        wide = replace(selection, max_seeds=max(selection.max_seeds * 4, 16))
        emb = {s.memory_id: s for s in self.embedding_selector.select_seeds(palace, wide)}
        lex = {s.memory_id: s for s in self.lexical_selector.select_seeds(palace, wide)}
        ids = set(emb) | set(lex)
        fused: list[SeedActivation] = []
        for mid in ids:
            e = emb.get(mid)
            l = lex.get(mid)
            if e and l:
                score = 0.6 * e.score + 0.4 * l.score
                reason = "hybrid_embed_lex"
                space_id = e.space_id or l.space_id
            elif e:
                score = e.score
                reason = e.reason
                space_id = e.space_id
            else:
                score = l.score if l else 0.0
                reason = l.reason if l else "hybrid_missing"
                space_id = l.space_id if l else None
            fused.append(
                SeedActivation(
                    memory_id=mid,
                    score=score,
                    reason=reason,
                    space_id=space_id,
                ),
            )
        fused.sort(key=lambda s: s.score, reverse=True)
        positive = [s for s in fused if s.score > 0.0]
        if not positive:
            return ()
        return tuple(positive[: max(1, selection.max_seeds)])


def default_hybrid_seed_selector() -> HybridSeedSelector:
    embedder = HashingEmbeddingProvider()
    return HybridSeedSelector(
        embedding_selector=EmbeddingSeedSelector(embedder=embedder),
        lexical_selector=LexicalSeedSelector(),
    )
