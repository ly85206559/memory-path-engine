import unittest

from memory_engine.memory.application.query_models import RecallPolicy, RecallQuery
from memory_engine.memory.application.retrieve_memory_service import RetrieveMemoryService
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
