from __future__ import annotations

from dataclasses import dataclass, field, replace

from memory_engine.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity,
    lexical_overlap,
)
from memory_engine.memory.application.encoding_service import (
    build_encoding_profile,
    trigger_match_score,
)
from memory_engine.memory.domain.encoding import EncodingProfile
from memory_engine.memory.domain.enums import MemoryKind
from memory_engine.memory.domain.memory_types import Memory, RouteMemory
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.seed_selection import (
    SeedActivation,
    SeedSelectionInput,
    SeedSelector,
)
from memory_engine.memory_state import MemoryStatePolicy

_MIN_SEED_SCORE = 0.1


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


def _encoding_for_memory(memory: Memory) -> EncodingProfile:
    if (
        memory.encoding.trigger_profile.phrases
        or memory.encoding.trigger_profile.situations
        or memory.encoding.scenario_tags
        or memory.encoding.symbolic_tags
    ):
        return memory.encoding
    return build_encoding_profile(
        memory.content,
        semantic_role=str(memory.metadata.get("semantic_role") or ""),
        existing_scenario_tags=tuple(memory.metadata.get("scenario_tags", ())),
        existing_symbolic_tags=tuple(memory.metadata.get("symbolic_tags", ())),
    )


def _semantic_role_alignment_score(query_text: str, semantic_role: str | None) -> float:
    lowered = query_text.lower()
    if semantic_role == "escalation" and any(
        token in lowered for token in ("escalate", "escalation", "page", "who should")
    ):
        return 1.0
    if semantic_role == "exception" and any(
        token in lowered for token in ("unless", "except", "override", "exception", "notwithstanding")
    ):
        return 1.0
    if semantic_role in {"remedy", "action"} and any(
        token in lowered for token in ("recover", "restart", "resolve", "what should", "what do")
    ):
        return 0.8
    if semantic_role == "condition" and any(
        token in lowered for token in ("if", "when", "after", "once", "under what")
    ):
        return 0.6
    return 0.0


def _intent_alignment_score(query_text: str, query_encoding: EncodingProfile, memory: Memory) -> float:
    memory_encoding = _encoding_for_memory(memory)
    semantic_role = str(memory.metadata.get("semantic_role") or "")
    shared_scenarios = set(query_encoding.scenario_tags) & set(memory_encoding.scenario_tags)
    shared_symbolic = set(query_encoding.symbolic_tags) & set(memory_encoding.symbolic_tags)
    score = 0.0
    score += 0.22 * min(len(shared_scenarios), 1)
    score += 0.08 * min(len(shared_symbolic), 2)
    score += 0.18 * _semantic_role_alignment_score(query_text, semantic_role)
    return min(score, 0.45)


def _seed_reason(*, trigger_score: float, intent_score: float, semantic_source: str) -> str:
    if trigger_score > 0.0 and intent_score > 0.0:
        return f"{semantic_source}+trigger+intent"
    if trigger_score > 0.0:
        return f"{semantic_source}+trigger"
    if intent_score > 0.0:
        return f"{semantic_source}+intent"
    return semantic_source


@dataclass(slots=True)
class EmbeddingSeedSelector:
    embedder: EmbeddingProvider
    state_policy: MemoryStatePolicy = field(default_factory=MemoryStatePolicy)

    def select_seeds(
        self,
        palace: MemoryPalace,
        selection: SeedSelectionInput,
    ) -> tuple[SeedActivation, ...]:
        memories = _iter_seed_memories(palace, selection)
        if not memories:
            return ()

        q_vec = self.embedder.embed(selection.text)
        query_encoding = build_encoding_profile(selection.text)
        scored: list[SeedActivation] = []
        for memory in memories:
            vec = self.embedder.embed(memory.content)
            trigger_score = trigger_match_score(selection.text, _encoding_for_memory(memory))
            intent_score = _intent_alignment_score(selection.text, query_encoding, memory)
            base_score = min(1.0, cosine_similarity(q_vec, vec) + 0.2 * trigger_score + 0.25 * intent_score)
            score = min(1.0, base_score * self.state_policy.recall_multiplier_for_state(memory.state))
            scored.append(
                SeedActivation(
                    memory_id=memory.memory_id,
                    score=score,
                    reason=_seed_reason(
                        trigger_score=trigger_score,
                        intent_score=intent_score,
                        semantic_source="embedding_cosine",
                    ),
                    space_id=_memory_space_id(memory) or None,
                ),
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        positive = [s for s in scored if s.score >= _MIN_SEED_SCORE]
        if not positive:
            return ()
        return tuple(positive[: max(1, selection.max_seeds)])


@dataclass(slots=True)
class LexicalSeedSelector:
    state_policy: MemoryStatePolicy = field(default_factory=MemoryStatePolicy)

    def select_seeds(
        self,
        palace: MemoryPalace,
        selection: SeedSelectionInput,
    ) -> tuple[SeedActivation, ...]:
        memories = _iter_seed_memories(palace, selection)
        if not memories:
            return ()

        query_encoding = build_encoding_profile(selection.text)
        scored: list[SeedActivation] = []
        for memory in memories:
            lexical_score = lexical_overlap(selection.text, memory.content)
            trigger_score = trigger_match_score(selection.text, _encoding_for_memory(memory))
            intent_score = _intent_alignment_score(selection.text, query_encoding, memory)
            base_score = min(1.0, lexical_score + 0.35 * trigger_score + 0.3 * intent_score)
            score = min(1.0, base_score * self.state_policy.recall_multiplier_for_state(memory.state))
            scored.append(
                SeedActivation(
                    memory_id=memory.memory_id,
                    score=score,
                    reason=_seed_reason(
                        trigger_score=trigger_score,
                        intent_score=intent_score,
                        semantic_source="lexical_overlap",
                    ),
                    space_id=_memory_space_id(memory) or None,
                ),
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        positive = [s for s in scored if s.score >= _MIN_SEED_SCORE]
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
                if "intent" in e.reason or "intent" in l.reason:
                    reason = "hybrid_embed_lex_trigger_intent"
                elif "trigger" in e.reason or "trigger" in l.reason:
                    reason = "hybrid_embed_lex_trigger"
                else:
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
        positive = [s for s in fused if s.score >= _MIN_SEED_SCORE]
        if not positive:
            return ()
        return tuple(positive[: max(1, selection.max_seeds)])


def default_hybrid_seed_selector(
    *, state_policy: MemoryStatePolicy | None = None
) -> HybridSeedSelector:
    embedder = HashingEmbeddingProvider()
    resolved_state_policy = state_policy or MemoryStatePolicy()
    return HybridSeedSelector(
        embedding_selector=EmbeddingSeedSelector(
            embedder=embedder,
            state_policy=resolved_state_policy,
        ),
        lexical_selector=LexicalSeedSelector(state_policy=resolved_state_policy),
    )
