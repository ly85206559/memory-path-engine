import unittest

from memory_engine.memory.application.consolidate_memory_service import ConsolidateMemoryService
from memory_engine.memory.application.query_models import RecallPolicy, RecallQuery
from memory_engine.memory.application.retrieve_memory_service import RetrieveMemoryService
from memory_engine.memory.domain.encoding import EncodingProfile, TriggerProfile
from memory_engine.memory.domain.enums import MemoryLifecycleState, MemoryLinkType
from memory_engine.memory.domain.memory_state import DomainMemoryState
from memory_engine.memory.domain.memory_types import EpisodicMemory
from memory_engine.memory.domain.palace import MemoryPalace, PalaceSpace
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile


class ConsolidateMemoryServiceTests(unittest.TestCase):
    def _build_palace(self) -> MemoryPalace:
        palace = MemoryPalace(palace_id="palace-c")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="ep-1",
                palace_id="palace-c",
                location=loc,
                content="Rollback failure required worker restart after queue drain stalled.",
                salience=SalienceProfile(0.9, 0.3, 0.4, 0.95),
                state=DomainMemoryState(state=MemoryLifecycleState.STABILIZING, reinforcement_count=2, stability_score=0.5),
                encoding=EncodingProfile(
                    trigger_profile=TriggerProfile(phrases=("rollback failure",)),
                    scenario_tags=("rollback failure",),
                    symbolic_tags=("causal",),
                ),
                metadata={"space_id": "ops", "semantic_role": "action"},
                episode_id="episode-1",
                event_type="incident",
                participants=("worker-service",),
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="ep-2",
                palace_id="palace-c",
                location=loc,
                content="Rollback failure recovery depended on queue drain confirmation and restart validation.",
                salience=SalienceProfile(0.86, 0.25, 0.35, 0.93),
                state=DomainMemoryState(state=MemoryLifecycleState.STABILIZING, reinforcement_count=3, stability_score=0.62),
                encoding=EncodingProfile(
                    trigger_profile=TriggerProfile(phrases=("rollback failure",)),
                    scenario_tags=("rollback failure",),
                    symbolic_tags=("causal", "urgency"),
                ),
                metadata={"space_id": "ops", "semantic_role": "action"},
                episode_id="episode-2",
                event_type="incident",
                participants=("worker-service", "incident-commander"),
            )
        )
        return palace

    def test_consolidation_creates_three_semantic_memories(self) -> None:
        palace = self._build_palace()
        result = ConsolidateMemoryService().consolidate(palace)

        self.assertEqual(len(result.groups), 1)
        created = {artifact.concept_type for artifact in result.created_artifacts}
        self.assertEqual(
            created,
            {"summary_memory", "generalized_rule_memory", "cross_episode_abstraction"},
        )
        self.assertIn("semantic:rollback_failure:summary_memory", palace.memories)
        self.assertIn("semantic:rollback_failure:generalized_rule_memory", palace.memories)
        self.assertIn("semantic:rollback_failure:cross_episode_abstraction", palace.memories)
        self.assertEqual(palace.memories["ep-1"].state.state, MemoryLifecycleState.CONSOLIDATED)

    def test_consolidation_adds_links_between_sources_and_artifacts(self) -> None:
        palace = self._build_palace()
        ConsolidateMemoryService().consolidate(palace)

        link_types = {
            (link.from_memory_id, link.to_memory_id, link.link_type)
            for link in palace.links
        }
        self.assertIn(
            ("ep-1", "semantic:rollback_failure:summary_memory", MemoryLinkType.PART_OF),
            link_types,
        )
        self.assertIn(
            (
                "semantic:rollback_failure:generalized_rule_memory",
                "ep-1",
                MemoryLinkType.RECALLS,
            ),
            link_types,
        )
        self.assertIn(
            (
                "semantic:rollback_failure:summary_memory",
                "semantic:rollback_failure:cross_episode_abstraction",
                MemoryLinkType.SUMMARIZES,
            ),
            link_types,
        )

    def test_consolidated_semantic_memory_can_surface_in_recall(self) -> None:
        palace = self._build_palace()
        ConsolidateMemoryService().consolidate(palace)

        result = RetrieveMemoryService().recall(
            palace,
            RecallQuery(
                palace_id="palace-c",
                text="give me the generalized rule for rollback failure",
                policy=RecallPolicy(top_k=4, max_hops=2),
            ),
        )

        retrieved_ids = {item.memory_id for item in result.retrieved_memories}
        self.assertIn("semantic:rollback_failure:generalized_rule_memory", retrieved_ids)
        generalized = next(
            item
            for item in result.retrieved_memories
            if item.memory_id == "semantic:rollback_failure:generalized_rule_memory"
        )
        self.assertEqual(generalized.consolidation_kind, "generalized_rule_memory")
        self.assertEqual(generalized.lifecycle_state, MemoryLifecycleState.CONSOLIDATED.value)

    def test_consolidation_keeps_same_scenario_groups_scoped_by_space(self) -> None:
        palace = self._build_palace()
        meeting_loc = PalaceLocation(building="hq", floor="2", room="meeting", locus="1")
        palace.add_space(PalaceSpace(space_id="meeting", name="Meeting Room", location=meeting_loc))
        for memory_id in ("ep-3", "ep-4"):
            palace.add_memory(
                EpisodicMemory(
                    memory_id=memory_id,
                    palace_id="palace-c",
                    location=meeting_loc,
                    content="Rollback failure follow-up discussion captured a separate meeting-space pattern.",
                    salience=SalienceProfile(0.82, 0.2, 0.3, 0.9),
                    state=DomainMemoryState(
                        state=MemoryLifecycleState.STABILIZING,
                        reinforcement_count=2,
                        stability_score=0.48,
                    ),
                    encoding=EncodingProfile(
                        trigger_profile=TriggerProfile(phrases=("rollback failure",)),
                        scenario_tags=("rollback failure",),
                        symbolic_tags=("causal",),
                    ),
                    metadata={"space_id": "meeting", "semantic_role": "action"},
                    episode_id=memory_id,
                    event_type="meeting",
                    participants=("incident-commander",),
                )
            )

        result = ConsolidateMemoryService().consolidate(palace)

        self.assertEqual(len(result.groups), 2)
        self.assertEqual({group.space_id for group in result.groups}, {"ops", "meeting"})
        summary_space_ids = {
            memory.metadata.get("space_id")
            for memory in palace.memories.values()
            if memory.metadata.get("consolidation_kind") == "summary_memory"
        }
        self.assertEqual(summary_space_ids, {"ops", "meeting"})
