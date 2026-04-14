import unittest

from memory_engine.benchmarking.application.public_benchmarks import ranked_node_ids_from_result
from memory_engine.memory.application.bridge import palace_to_store, store_to_palace
from memory_engine.memory.application.query_models import RecallPolicy, RecallQuery
from memory_engine.memory.application.retrieve_memory_service import RetrieveMemoryService
from memory_engine.memory.application.reinforce_memory_service import ReinforceMemoryService
from memory_engine.memory.domain.enums import MemoryLifecycleState, MemoryLinkType
from memory_engine.memory.domain.memory_state import DomainMemoryState
from memory_engine.memory.domain.memory_state_machine import MemoryStateMachine
from memory_engine.memory.domain.memory_types import EpisodicMemory, RouteMemory
from memory_engine.memory.domain.palace import MemoryLink, MemoryPalace, PalaceSpace
from memory_engine.memory.domain.retrieval_result import PalaceRecallResult, RecallRoute, RetrievedMemory
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile
from memory_engine.schema import EvidenceRef, MemoryNode, MemoryWeight
from memory_engine.memory_state import MemoryStatePolicy


class MemoryPalaceV1Tests(unittest.TestCase):
    def test_palace_bridge_round_trip_preserves_memory_kinds(self):
        palace = MemoryPalace(palace_id="palace-1")
        location = PalaceLocation(building="hq", floor="f1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=location))
        palace.add_memory(
            EpisodicMemory(
                memory_id="mem-1",
                palace_id="palace-1",
                location=location,
                content="Database failover was triggered.",
                salience=SalienceProfile(importance=0.8, risk=0.5, novelty=0.4, confidence=0.9),
                source=EvidenceRef(source_path="incident.md", section_id="1"),
                metadata={"space_id": "ops"},
                episode_id="episode-1",
            )
        )
        palace.add_memory(
            RouteMemory(
                memory_id="route-1",
                palace_id="palace-1",
                location=location,
                content="mem-1",
                salience=SalienceProfile(importance=0.5, risk=0.1, novelty=0.2, confidence=0.9),
                metadata={"space_id": "ops"},
                route_id="route-1",
                start_memory_id="mem-1",
                ordered_waypoints=("mem-1",),
            )
        )
        palace.add_link(
            MemoryLink(
                from_memory_id="route-1",
                to_memory_id="mem-1",
                link_type=MemoryLinkType.ROUTE_TO,
            )
        )

        restored = store_to_palace(palace_to_store(palace), palace_id="palace-1")

        self.assertEqual(restored.memories["mem-1"].kind.value, "episodic")
        self.assertEqual(restored.memories["route-1"].kind.value, "route")

    def test_public_benchmark_prefers_explicit_retrieved_memories(self):
        palace_result = PalaceRecallResult(
            query="q",
            retrieved_memories=(
                RetrievedMemory(memory_id="m-2", score=0.9, reason="best"),
                RetrievedMemory(memory_id="m-1", score=0.7, reason="support"),
            ),
            routes=(RecallRoute(route_id="r1", route_kind="legacy", step_memory_ids=("m-2", "m-1"), score=0.9),),
        )
        result = palace_result.to_legacy_retrieval_result()

        self.assertEqual(ranked_node_ids_from_result(result, top_k=2), ["m-2", "m-1"])

    def test_retrieve_memory_service_returns_palace_recall_result(self):
        palace = MemoryPalace(palace_id="palace-1")
        location = PalaceLocation(building="hq", floor="f1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=location))
        palace.add_memory(
            EpisodicMemory(
                memory_id="mem-1",
                palace_id="palace-1",
                location=location,
                content="The worker queue must be restarted after rollback failure.",
                salience=SalienceProfile(importance=0.9, risk=0.5, novelty=0.4, confidence=0.95),
                metadata={"space_id": "ops"},
                episode_id="episode-1",
            )
        )

        result = RetrieveMemoryService().recall(
            palace,
            RecallQuery(
                palace_id="palace-1",
                text="What should happen after rollback failure?",
                policy=RecallPolicy(retriever_mode="weighted_graph", top_k=1, max_hops=1),
            ),
        )

        self.assertIsInstance(result, PalaceRecallResult)
        self.assertEqual(result.retrieved_memories[0].memory_id, "mem-1")

    def test_state_machine_and_reinforce_service_update_lifecycle(self):
        palace = MemoryPalace(palace_id="palace-1")
        location = PalaceLocation(building="hq", room="ops")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=location))
        palace.add_memory(
            EpisodicMemory(
                memory_id="mem-1",
                palace_id="palace-1",
                location=location,
                content="Failover was executed.",
                salience=SalienceProfile(importance=0.7, risk=0.3, novelty=0.2, confidence=0.9),
                state=DomainMemoryState(),
                metadata={"space_id": "ops"},
                episode_id="episode-1",
            )
        )

        result = PalaceRecallResult(
            query="q",
            retrieved_memories=(RetrievedMemory(memory_id="mem-1", score=1.0, reason="seed"),),
            routes=(RecallRoute(route_id="r1", route_kind="timeline", step_memory_ids=("mem-1",), score=1.0),),
        )
        ReinforceMemoryService().reinforce_recall_result(palace, result)

        self.assertEqual(palace.memories["mem-1"].state.state, MemoryLifecycleState.ACTIVE)
        self.assertGreater(palace.memories["mem-1"].state.stability_score, 0.0)

    def test_memory_state_machine_can_consolidate_after_multiple_reinforcements(self):
        machine = MemoryStateMachine()
        state = DomainMemoryState()
        for _ in range(4):
            state = machine.reinforce(state)

        self.assertEqual(state.state, MemoryLifecycleState.CONSOLIDATED)

    def test_memory_state_policy_decay_cycle_updates_lifecycle_attributes(self):
        policy = MemoryStatePolicy()
        node = MemoryNode(
            id="n1",
            type="clause",
            content="test",
            weights=MemoryWeight(decay_factor=1.0),
        )
        policy.reinforce_node(node)
        self.assertEqual(node.attributes.get("lifecycle_state"), MemoryLifecycleState.ACTIVE.value)
        policy.decay_node(node, steps=3)
        self.assertNotEqual(node.attributes.get("lifecycle_state"), MemoryLifecycleState.ENCODED.value)
        self.assertLess(node.weights.decay_factor, 1.0)
