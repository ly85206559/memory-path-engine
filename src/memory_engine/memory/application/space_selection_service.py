from __future__ import annotations

from dataclasses import dataclass

from memory_engine.embeddings import lexical_overlap
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.space_selection import (
    SpaceCandidate,
    SpaceSelectionInput,
    SpaceSelector,
)


def _space_text(palace: MemoryPalace, space_id: str) -> str:
    space = palace.spaces[space_id]
    parts = [space.name, space.location.as_key(), *space.tags]
    return " ".join(parts)


def _memories_in_space(palace: MemoryPalace, space_id: str) -> list[str]:
    return [
        m.content
        for m in palace.memories.values()
        if str(m.metadata.get("space_id", "")) == space_id
    ]


def _sort_candidates(candidates: list[SpaceCandidate]) -> list[SpaceCandidate]:
    return sorted(candidates, key=lambda c: (-c.score, c.space_id))


def _fallback_all_spaces(candidates: list[SpaceCandidate]) -> tuple[SpaceCandidate, ...]:
    return tuple(
        SpaceCandidate(space_id=c.space_id, score=0.01, reason="fallback_all_spaces")
        for c in _sort_candidates(candidates)
    )


@dataclass(slots=True)
class KeywordSpaceSelector:
    """Rank spaces by lexical overlap of query with space name, tags, and location key."""

    def select_spaces(
        self,
        palace: MemoryPalace,
        selection: SpaceSelectionInput,
    ) -> tuple[SpaceCandidate, ...]:
        if not palace.spaces:
            return ()

        candidates: list[SpaceCandidate] = []
        for space_id, _space in palace.spaces.items():
            text = _space_text(palace, space_id)
            score = lexical_overlap(selection.text, text)
            if selection.preferred_space_ids and space_id in selection.preferred_space_ids:
                score = min(1.0, score + 0.35)
            candidates.append(
                SpaceCandidate(space_id=space_id, score=score, reason="keyword_space_lexical"),
            )

        ranked = _sort_candidates(candidates)
        top = ranked[: max(1, selection.max_spaces)]

        if ranked and all(c.score <= 0.0 for c in ranked):
            return _fallback_all_spaces(ranked)
        return tuple(top)


@dataclass(slots=True)
class MetadataSpaceSelector:
    """Rank spaces by lexical overlap of query with aggregated memory content in each space."""

    def select_spaces(
        self,
        palace: MemoryPalace,
        selection: SpaceSelectionInput,
    ) -> tuple[SpaceCandidate, ...]:
        if not palace.spaces:
            return ()

        candidates: list[SpaceCandidate] = []
        for space_id in palace.spaces:
            contents = _memories_in_space(palace, space_id)
            blob = "\n".join(contents) if contents else ""
            score = lexical_overlap(selection.text, blob) if blob else 0.0
            if selection.preferred_space_ids and space_id in selection.preferred_space_ids:
                score = min(1.0, score + 0.25)
            candidates.append(
                SpaceCandidate(
                    space_id=space_id,
                    score=score,
                    reason="metadata_space_memory_blob",
                ),
            )

        ranked = _sort_candidates(candidates)
        top = ranked[: max(1, selection.max_spaces)]

        if ranked and all(c.score <= 0.0 for c in ranked):
            return _fallback_all_spaces(ranked)
        return tuple(top)


@dataclass(slots=True)
class HybridSpaceSelector:
    """Combine keyword and metadata signals with a simple max fusion."""

    keyword_selector: KeywordSpaceSelector
    metadata_selector: MetadataSpaceSelector

    def select_spaces(
        self,
        palace: MemoryPalace,
        selection: SpaceSelectionInput,
    ) -> tuple[SpaceCandidate, ...]:
        by_kw = {c.space_id: c for c in self.keyword_selector.select_spaces(palace, selection)}
        by_md = {c.space_id: c for c in self.metadata_selector.select_spaces(palace, selection)}
        space_ids = set(by_kw) | set(by_md)
        fused: list[SpaceCandidate] = []
        for sid in space_ids:
            k = by_kw.get(sid)
            m = by_md.get(sid)
            if k and m:
                score = max(k.score, m.score)
                reason = "hybrid_max(keyword,metadata)"
            elif k:
                score = k.score
                reason = k.reason
            else:
                score = m.score if m else 0.0
                reason = m.reason if m else "hybrid_missing"
            fused.append(SpaceCandidate(space_id=sid, score=score, reason=reason))

        ranked = _sort_candidates(fused)
        top = ranked[: max(1, selection.max_spaces)]
        if ranked and all(c.score <= 0.0 for c in ranked):
            return _fallback_all_spaces(ranked)
        return tuple(top)
