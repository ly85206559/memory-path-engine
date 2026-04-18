from __future__ import annotations

from dataclasses import dataclass

from memory_engine.embeddings import lexical_overlap, tokenize
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


def _location_prior_score(palace: MemoryPalace, space_id: str, query_text: str) -> float:
    space = palace.spaces[space_id]
    query_tokens = set(tokenize(query_text))
    if not query_tokens:
        return 0.0

    building_tokens = set(tokenize(space.location.building))
    floor_tokens = set(tokenize(space.location.floor or ""))
    room_tokens = set(tokenize(space.location.room or ""))
    locus_tokens = set(tokenize(space.location.locus or ""))
    tag_tokens = {token for tag in space.tags for token in tokenize(tag)}
    name_tokens = set(tokenize(space.name))

    score = 0.0
    if building_tokens & query_tokens:
        score += 0.35
    if floor_tokens & query_tokens:
        score += 0.2
    if room_tokens & query_tokens:
        score += 0.35
    if locus_tokens & query_tokens:
        score += 0.15
    if tag_tokens & query_tokens:
        score += 0.15
    if name_tokens & query_tokens:
        score += 0.2
    return min(score, 1.0)


def _sort_candidates(candidates: list[SpaceCandidate]) -> list[SpaceCandidate]:
    return sorted(candidates, key=lambda c: (-c.score, c.space_id))


def _fallback_scope_subset(
    palace: MemoryPalace,
    selection: SpaceSelectionInput,
) -> tuple[SpaceCandidate, ...]:
    fallback_ids: list[str] = []
    seen: set[str] = set()

    for space_id in selection.preferred_space_ids:
        if space_id in palace.spaces and space_id not in seen:
            fallback_ids.append(space_id)
            seen.add(space_id)

    for space_id in palace.spaces:
        if space_id not in seen:
            fallback_ids.append(space_id)
            seen.add(space_id)

    limited_ids = fallback_ids[: max(1, selection.max_spaces)]
    reason = "fallback_preferred_spaces" if selection.preferred_space_ids else "fallback_low_confidence_scope"
    return tuple(
        SpaceCandidate(space_id=space_id, score=0.01, reason=reason)
        for space_id in limited_ids
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
        has_signal = False
        for space_id, _space in palace.spaces.items():
            text = _space_text(palace, space_id)
            lexical_score = lexical_overlap(selection.text, text)
            location_score = _location_prior_score(palace, space_id, selection.text)
            signal_score = max(lexical_score, location_score)
            has_signal = has_signal or signal_score > 0.0
            score = signal_score
            if selection.preferred_space_ids and space_id in selection.preferred_space_ids:
                score = min(1.0, score + 0.35)
            candidates.append(
                SpaceCandidate(
                    space_id=space_id,
                    score=score,
                    reason="keyword_space_location_prior" if location_score > lexical_score else "keyword_space_lexical",
                ),
            )

        ranked = _sort_candidates(candidates)
        top = ranked[: max(1, selection.max_spaces)]

        if ranked and not has_signal:
            return _fallback_scope_subset(palace, selection)
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
        has_signal = False
        for space_id in palace.spaces:
            contents = _memories_in_space(palace, space_id)
            blob = "\n".join(contents) if contents else ""
            lexical_score = lexical_overlap(selection.text, blob) if blob else 0.0
            location_score = _location_prior_score(palace, space_id, selection.text)
            signal_score = max(lexical_score, location_score)
            has_signal = has_signal or signal_score > 0.0
            score = signal_score
            if selection.preferred_space_ids and space_id in selection.preferred_space_ids:
                score = min(1.0, score + 0.25)
            candidates.append(
                SpaceCandidate(
                    space_id=space_id,
                    score=score,
                    reason=(
                        "metadata_space_location_prior"
                        if location_score > lexical_score
                        else "metadata_space_memory_blob"
                    ),
                ),
            )

        ranked = _sort_candidates(candidates)
        top = ranked[: max(1, selection.max_spaces)]

        if ranked and not has_signal:
            return _fallback_scope_subset(palace, selection)
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
                if k.reason.startswith("fallback_") and m.reason.startswith("fallback_"):
                    reason = k.reason
                elif "location_prior" in k.reason or "location_prior" in m.reason:
                    reason = "hybrid_max(location_prior)"
                else:
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
            return _fallback_scope_subset(palace, selection)
        return tuple(top)
