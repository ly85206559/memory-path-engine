import unittest

from memory_engine.memory.application.query_models import RecallPolicy, RecallQuery
from memory_engine.memory.application.retrieve_memory_service import RetrieveMemoryService
from memory_engine.memory.domain.enums import MemoryLifecycleState
from memory_engine.memory.domain.memory_state import DomainMemoryState
from memory_engine.memory.domain.memory_types import EpisodicMemory, RouteMemory
from memory_engine.memory.domain.palace import MemoryLink, MemoryPalace, PalaceSpace
from memory_engine.memory.domain.enums import MemoryLinkType
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile


class RetrieveMemoryServiceV2Tests(unittest.TestCase):
    def test_orchestration_prefers_route_memory_when_present(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops Center", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="mem-1",
                palace_id="p1",
                location=loc,
                content="Canary deployment failed during rollback window.",
                salience=SalienceProfile(0.9, 0.3, 0.4, 0.95),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            RouteMemory(
                memory_id="route-1",
                palace_id="p1",
                location=loc,
                content="route",
                salience=SalienceProfile(0.4, 0.1, 0.2, 0.8),
                metadata={"space_id": "ops"},
                route_id="r1",
                start_memory_id="mem-1",
                ordered_waypoints=("mem-1",),
                route_kind="timeline",
            )
        )
        palace.add_link(
            MemoryLink(
                from_memory_id="route-1",
                to_memory_id="mem-1",
                link_type=MemoryLinkType.ROUTE_TO,
            )
        )

        svc = RetrieveMemoryService()
        result = svc.recall(
            palace,
            RecallQuery(
                palace_id="p1",
                text="rollback canary deployment failure",
                policy=RecallPolicy(top_k=3, max_hops=2, max_spaces=2, max_seeds=4),
            ),
        )
        self.assertTrue(result.metadata.get("recall_orchestration"))
        self.assertTrue(result.routes)
        self.assertEqual(result.routes[0].route_source, "route_memory")
        self.assertIn("mem-1", [m.memory_id for m in result.retrieved_memories])
        self.assertTrue(result.activation_snapshot.steps)

    def test_legacy_fallback_when_native_produces_no_memories(self) -> None:
        palace = MemoryPalace(palace_id="empty-ish")
        svc = RetrieveMemoryService()
        result = svc.recall(
            palace,
            RecallQuery(
                palace_id="empty-ish",
                text="no memories here",
                policy=RecallPolicy(
                    allow_legacy_fallback=True,
                    retriever_mode="lexical_baseline",
                    top_k=2,
                ),
            ),
        )
        self.assertIn("fallback_reason", result.metadata)


    def test_native_activation_builds_dependency_route_and_snapshot(self) -> None:
        palace = MemoryPalace(palace_id="p2")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops Center", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="seed",
                palace_id="p2",
                location=loc,
                content="Rollback failure blocks recovery.",
                salience=SalienceProfile(0.9, 0.3, 0.4, 0.95),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="dep",
                palace_id="p2",
                location=loc,
                content="Queue drain depends on worker restart.",
                salience=SalienceProfile(0.8, 0.2, 0.3, 0.92),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_link(
            MemoryLink(
                from_memory_id="seed",
                to_memory_id="dep",
                link_type=MemoryLinkType.DEPENDS_ON,
                strength=1.8,
            )
        )

        result = RetrieveMemoryService().recall(
            palace,
            RecallQuery(
                palace_id="p2",
                text="what dependency is needed after rollback failure",
                policy=RecallPolicy(top_k=3, max_hops=2, retriever_mode="activation_spreading_v1"),
            ),
        )

        self.assertTrue(result.activation_snapshot.steps)
        self.assertTrue(any(route.route_kind == "dependency" for route in result.routes))
        self.assertTrue(any(route.route_source == "native_activation" for route in result.routes))
        self.assertIn("dep", [memory.memory_id for memory in result.retrieved_memories])
        self.assertTrue(all(route.explanation for route in result.routes))

    def test_static_mode_does_not_apply_lifecycle_bias_in_palace_recall(self) -> None:
        palace = MemoryPalace(palace_id="p3")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops Center", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="fading-first",
                palace_id="p3",
                location=loc,
                content="Recovery verification checklist",
                salience=SalienceProfile(0.8, 0.2, 0.3, 0.9),
                metadata={"space_id": "ops"},
                state=DomainMemoryState(
                    state=MemoryLifecycleState.FADING,
                    reinforcement_count=2,
                    stability_score=0.2,
                    decay_factor=0.75,
                ),
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="active-second",
                palace_id="p3",
                location=loc,
                content="Recovery verification checklist",
                salience=SalienceProfile(0.8, 0.2, 0.3, 0.9),
                metadata={"space_id": "ops"},
                state=DomainMemoryState(
                    state=MemoryLifecycleState.ACTIVE,
                    reinforcement_count=2,
                    stability_score=0.4,
                    decay_factor=1.0,
                ),
            )
        )

        service = RetrieveMemoryService()
        static_result = service.recall(
            palace,
            RecallQuery(
                palace_id="p3",
                text="recovery verification checklist",
                policy=RecallPolicy(top_k=2, max_hops=0, retriever_mode="activation_spreading_static"),
            ),
        )
        dynamic_result = service.recall(
            palace,
            RecallQuery(
                palace_id="p3",
                text="recovery verification checklist",
                policy=RecallPolicy(top_k=2, max_hops=0, retriever_mode="activation_spreading_v1"),
            ),
        )

        static_scores = {item.memory_id: item.score for item in static_result.retrieved_memories}
        dynamic_scores = {item.memory_id: item.score for item in dynamic_result.retrieved_memories}

        self.assertEqual(static_scores["fading-first"], static_scores["active-second"])
        self.assertGreater(dynamic_scores["active-second"], dynamic_scores["fading-first"])

    def test_low_confidence_space_fallback_stays_bounded(self) -> None:
        palace = MemoryPalace(palace_id="p4")
        ops_loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        lab_loc = PalaceLocation(building="annex", floor="2", room="lab", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops Center", location=ops_loc))
        palace.add_space(PalaceSpace(space_id="lab", name="Lab", location=lab_loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="ops-memory",
                palace_id="p4",
                location=ops_loc,
                content="Rollback failure requires worker restart.",
                salience=SalienceProfile(0.8, 0.2, 0.3, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="lab-memory",
                palace_id="p4",
                location=lab_loc,
                content="Experiment notes for unrelated lab setup.",
                salience=SalienceProfile(0.6, 0.1, 0.2, 0.8),
                metadata={"space_id": "lab"},
            )
        )

        result = RetrieveMemoryService().recall(
            palace,
            RecallQuery(
                palace_id="p4",
                text="zxqv mntr plko",
                policy=RecallPolicy(top_k=3, max_spaces=1, max_seeds=3),
            ),
        )

        self.assertEqual(result.metadata["selected_space_ids"], ["ops"])
        self.assertEqual(result.metadata["space_candidates"][0]["reason"], "fallback_low_confidence_scope")

    def test_route_merge_prefers_route_memory_over_duplicate_native_path(self) -> None:
        palace = MemoryPalace(palace_id="p5")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops Center", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="seed",
                palace_id="p5",
                location=loc,
                content="Rollback failure blocks recovery.",
                salience=SalienceProfile(0.9, 0.3, 0.4, 0.95),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="dep",
                palace_id="p5",
                location=loc,
                content="Queue drain depends on worker restart.",
                salience=SalienceProfile(0.8, 0.2, 0.3, 0.92),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            RouteMemory(
                memory_id="route-dep",
                palace_id="p5",
                location=loc,
                content="Persisted dependency route",
                salience=SalienceProfile(0.5, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
                route_id="route-dep",
                start_memory_id="seed",
                ordered_waypoints=("dep",),
                route_kind="dependency",
            )
        )
        palace.add_link(
            MemoryLink(
                from_memory_id="seed",
                to_memory_id="dep",
                link_type=MemoryLinkType.DEPENDS_ON,
                strength=1.8,
            )
        )

        result = RetrieveMemoryService().recall(
            palace,
            RecallQuery(
                palace_id="p5",
                text="what dependency is needed after rollback failure",
                policy=RecallPolicy(top_k=3, max_hops=2, retriever_mode="activation_spreading_v1"),
            ),
        )

        matching_routes = [
            route for route in result.routes if route.step_memory_ids == ("seed", "dep")
        ]
        self.assertEqual(len(matching_routes), 1)
        self.assertEqual(matching_routes[0].route_source, "route_memory")
