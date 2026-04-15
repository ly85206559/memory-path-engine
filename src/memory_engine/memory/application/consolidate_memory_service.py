from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from memory_engine.embeddings import tokenize
from memory_engine.memory.application.encoding_service import build_encoding_profile
from memory_engine.memory.domain.consolidation import (
    ConsolidationArtifact,
    ConsolidationGroup,
    ConsolidationPolicy,
)
from memory_engine.memory.domain.enums import MemoryLifecycleState, MemoryLinkType
from memory_engine.memory.domain.memory_state_machine import MemoryStateMachine
from memory_engine.memory.domain.memory_types import EpisodicMemory, SemanticMemory
from memory_engine.memory.domain.palace import MemoryLink, MemoryPalace
from memory_engine.memory.domain.value_objects import SalienceProfile

_STOP_TOKENS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "if",
    "when",
    "after",
    "once",
    "then",
    "must",
    "should",
    "does",
    "not",
    "within",
    "for",
    "with",
    "into",
    "from",
    "that",
    "this",
    "what",
    "which",
    "where",
}


def _normalize_key(value: str) -> str:
    return "_".join(tokenize(value)) or "memory"


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    groups: tuple[ConsolidationGroup, ...] = ()
    created_memory_ids: tuple[str, ...] = ()
    created_artifacts: tuple[ConsolidationArtifact, ...] = ()


@dataclass(slots=True)
class ConsolidateMemoryService:
    policy: ConsolidationPolicy = field(default_factory=ConsolidationPolicy)
    state_machine: MemoryStateMachine = field(default_factory=MemoryStateMachine)

    def consolidate(self, palace: MemoryPalace) -> ConsolidationResult:
        groups = self._candidate_groups(palace)
        scenario_group_counts = Counter(group.scenario_tag for group in groups)
        created_memory_ids: list[str] = []
        artifacts: list[ConsolidationArtifact] = []

        for group in groups:
            source_memories = [palace.memories[memory_id] for memory_id in group.source_memory_ids]
            location = source_memories[0].location
            summary = self._build_semantic_memory(
                palace=palace,
                group=group,
                scenario_group_counts=scenario_group_counts,
                concept_type="summary_memory",
                content=self._summary_content(group, source_memories),
                canonical_form=f"{group.scenario_tag} summary",
                aliases=("summary memory", group.scenario_tag),
            )
            generalized = self._build_semantic_memory(
                palace=palace,
                group=group,
                scenario_group_counts=scenario_group_counts,
                concept_type="generalized_rule_memory",
                content=self._generalized_rule_content(group, source_memories),
                canonical_form=f"{group.scenario_tag} generalized rule",
                aliases=("generalized rule", group.scenario_tag),
            )
            abstraction = self._build_semantic_memory(
                palace=palace,
                group=group,
                scenario_group_counts=scenario_group_counts,
                concept_type="cross_episode_abstraction",
                content=self._cross_episode_content(group, source_memories),
                canonical_form=f"{group.scenario_tag} cross episode abstraction",
                aliases=("cross episode abstraction", group.scenario_tag),
            )

            for memory in (summary, generalized, abstraction):
                if memory.memory_id in palace.memories:
                    continue
                palace.add_memory(memory)
                created_memory_ids.append(memory.memory_id)
                artifacts.append(
                    ConsolidationArtifact(
                        memory_id=memory.memory_id,
                        concept_type=memory.concept_type,
                        source_memory_ids=group.source_memory_ids,
                    )
                )

            for source_memory in source_memories:
                source_memory.state = self.state_machine.consolidate(source_memory.state)
                self._link_source_to_summary(palace, source_memory.memory_id, summary.memory_id)
                self._link_abstraction_to_source(palace, generalized.memory_id, source_memory.memory_id)
                self._link_abstraction_to_source(palace, abstraction.memory_id, source_memory.memory_id)

            self._link_summary_to_abstraction(palace, summary.memory_id, generalized.memory_id)
            self._link_summary_to_abstraction(palace, summary.memory_id, abstraction.memory_id)

        return ConsolidationResult(
            groups=tuple(groups),
            created_memory_ids=tuple(created_memory_ids),
            created_artifacts=tuple(artifacts),
        )

    def _candidate_groups(self, palace: MemoryPalace) -> list[ConsolidationGroup]:
        by_scenario_and_space: dict[tuple[str, str], list[EpisodicMemory]] = defaultdict(list)
        for memory in palace.memories.values():
            if not isinstance(memory, EpisodicMemory):
                continue
            if memory.state.state not in self.policy.allowed_source_states:
                continue
            scenarios = memory.encoding.scenario_tags or tuple(memory.metadata.get("scenario_tags", ()))
            if not scenarios:
                inferred = build_encoding_profile(
                    memory.content,
                    semantic_role=str(memory.metadata.get("semantic_role") or ""),
                )
                scenarios = inferred.scenario_tags
            space_id = str(memory.metadata.get("space_id", memory.location.as_key()))
            for scenario in scenarios:
                by_scenario_and_space[(scenario, space_id)].append(memory)

        groups: list[ConsolidationGroup] = []
        for (scenario_tag, space_id), memories in by_scenario_and_space.items():
            if len(memories) < self.policy.minimum_group_size:
                continue
            ordered = sorted(
                memories,
                key=lambda memory: (
                    -memory.state.stability_score,
                    -memory.salience.base_score(),
                    memory.memory_id,
                ),
            )[: self.policy.max_source_memories_per_group]
            shared_symbolic_tags = self._shared_symbolic_tags(tuple(ordered))
            groups.append(
                ConsolidationGroup(
                    palace_id=palace.palace_id,
                    scenario_tag=scenario_tag,
                    space_id=space_id,
                    source_memory_ids=tuple(memory.memory_id for memory in ordered),
                    shared_symbolic_tags=shared_symbolic_tags,
                )
            )
        groups.sort(key=lambda group: (group.scenario_tag, group.space_id, group.source_memory_ids))
        return groups

    def _shared_symbolic_tags(self, memories: tuple[EpisodicMemory, ...]) -> tuple[str, ...]:
        if not memories:
            return ()
        shared = set(memories[0].encoding.symbolic_tags or memories[0].metadata.get("symbolic_tags", ()))
        for memory in memories[1:]:
            shared &= set(memory.encoding.symbolic_tags or memory.metadata.get("symbolic_tags", ()))
        return tuple(sorted(shared))

    def _build_semantic_memory(
        self,
        *,
        palace: MemoryPalace,
        group: ConsolidationGroup,
        scenario_group_counts: Counter[str],
        concept_type: str,
        content: str,
        canonical_form: str,
        aliases: tuple[str, ...],
    ) -> SemanticMemory:
        source_memories = [palace.memories[memory_id] for memory_id in group.source_memory_ids]
        location = source_memories[0].location
        salience = self._aggregate_salience(tuple(source_memories))
        encoding = build_encoding_profile(
            content,
            existing_scenario_tags=(group.scenario_tag,),
            existing_symbolic_tags=group.shared_symbolic_tags,
        )
        memory_id = self._semantic_memory_id(
            group=group,
            concept_type=concept_type,
            disambiguate_with_space=scenario_group_counts[group.scenario_tag] > 1,
        )
        source_refs = tuple(
            memory.source.source_path
            for memory in source_memories
            if memory.source is not None and memory.source.source_path
        )
        metadata = {
            "space_id": group.space_id,
            "scenario_tags": list(encoding.scenario_tags),
            "symbolic_tags": list(encoding.symbolic_tags),
            "consolidation_kind": concept_type,
            "consolidated_from": list(group.source_memory_ids),
            "source_document_paths": list(sorted(set(source_refs))),
        }
        return SemanticMemory(
            memory_id=memory_id,
            palace_id=palace.palace_id,
            location=location,
            content=content,
            salience=salience,
            state=self.state_machine.consolidate(source_memories[0].state),
            encoding=encoding,
            metadata=metadata,
            concept_id=memory_id,
            concept_type=concept_type,
            canonical_form=canonical_form,
            aliases=aliases,
        )

    def _semantic_memory_id(
        self,
        *,
        group: ConsolidationGroup,
        concept_type: str,
        disambiguate_with_space: bool,
    ) -> str:
        scenario_key = _normalize_key(group.scenario_tag)
        if not disambiguate_with_space:
            return f"semantic:{scenario_key}:{concept_type}"
        return f"semantic:{scenario_key}:{_normalize_key(group.space_id)}:{concept_type}"

    def _aggregate_salience(self, memories: tuple[EpisodicMemory, ...]) -> SalienceProfile:
        count = max(1, len(memories))
        return SalienceProfile(
            importance=min(1.0, sum(memory.salience.importance for memory in memories) / count + 0.08),
            risk=min(1.0, sum(memory.salience.risk for memory in memories) / count),
            novelty=min(1.0, sum(memory.salience.novelty for memory in memories) / count * 0.9),
            confidence=min(1.0, sum(memory.salience.confidence for memory in memories) / count + 0.05),
            emotional_intensity=min(
                1.0,
                sum(memory.salience.emotional_intensity for memory in memories) / count,
            ),
            recency=min(1.0, sum(memory.salience.recency for memory in memories) / count),
        )

    def _summary_content(self, group: ConsolidationGroup, memories: list[EpisodicMemory]) -> str:
        examples = " | ".join(memory.content for memory in memories[:2])
        return (
            f"Summary memory for {group.scenario_tag}: "
            f"{len(memories)} related episodes recur around this scenario. "
            f"Representative episodes: {examples}"
        )

    def _generalized_rule_content(self, group: ConsolidationGroup, memories: list[EpisodicMemory]) -> str:
        recurring_terms = ", ".join(self._top_terms(memories))
        symbolic = ", ".join(group.shared_symbolic_tags) if group.shared_symbolic_tags else "none"
        return (
            f"Generalized rule for {group.scenario_tag}: "
            f"When this scenario appears, prioritize recurring signals [{recurring_terms}] "
            f"and account for symbolic tags [{symbolic}]."
        )

    def _cross_episode_content(self, group: ConsolidationGroup, memories: list[EpisodicMemory]) -> str:
        event_types = sorted({memory.event_type for memory in memories if memory.event_type})
        participants = sorted(
            {
                participant
                for memory in memories
                for participant in memory.participants
                if participant
            }
        )
        return (
            f"Cross-episode abstraction for {group.scenario_tag}: "
            f"Across {len(memories)} episodes, shared event types are {', '.join(event_types) or 'event'} "
            f"and repeated actors are {', '.join(participants) or 'unspecified'}. "
            f"This abstraction links multiple episodes into one stable semantic memory."
        )

    def _top_terms(self, memories: list[EpisodicMemory]) -> tuple[str, ...]:
        counter: Counter[str] = Counter()
        for memory in memories:
            counter.update(
                token
                for token in tokenize(memory.content)
                if len(token) > 2 and token not in _STOP_TOKENS
            )
        return tuple(token for token, _count in counter.most_common(5))

    def _link_source_to_summary(self, palace: MemoryPalace, source_id: str, summary_id: str) -> None:
        self._add_link_if_missing(
            palace,
            MemoryLink(
                from_memory_id=source_id,
                to_memory_id=summary_id,
                link_type=MemoryLinkType.PART_OF,
                strength=0.8,
            ),
        )

    def _link_abstraction_to_source(self, palace: MemoryPalace, abstraction_id: str, source_id: str) -> None:
        self._add_link_if_missing(
            palace,
            MemoryLink(
                from_memory_id=abstraction_id,
                to_memory_id=source_id,
                link_type=MemoryLinkType.RECALLS,
                strength=0.72,
            ),
        )

    def _link_summary_to_abstraction(self, palace: MemoryPalace, summary_id: str, abstraction_id: str) -> None:
        self._add_link_if_missing(
            palace,
            MemoryLink(
                from_memory_id=summary_id,
                to_memory_id=abstraction_id,
                link_type=MemoryLinkType.SUMMARIZES,
                strength=0.84,
            ),
        )

    def _add_link_if_missing(self, palace: MemoryPalace, link: MemoryLink) -> None:
        for existing in palace.links:
            if (
                existing.from_memory_id == link.from_memory_id
                and existing.to_memory_id == link.to_memory_id
                and existing.link_type == link.link_type
            ):
                return
        palace.add_link(link)
