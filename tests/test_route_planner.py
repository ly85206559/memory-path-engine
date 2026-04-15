import unittest

from memory_engine.memory.application.route_planner_service import DefaultRoutePlanner
from memory_engine.memory.domain.memory_types import EpisodicMemory, RouteMemory
from memory_engine.memory.domain.palace import MemoryLink, MemoryPalace, PalaceSpace
from memory_engine.memory.domain.enums import MemoryLinkType
from memory_engine.memory.domain.route_planning import RoutePlanningInput
from memory_engine.memory.domain.seed_selection import SeedActivation
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile


class RoutePlannerTests(unittest.TestCase):
    def test_route_memory_yields_route_source_route_memory(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="mem-1",
                palace_id="p1",
                location=loc,
                content="First step narrative.",
                salience=SalienceProfile(0.8, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            RouteMemory(
                memory_id="route-1",
                palace_id="p1",
                location=loc,
                content="timeline",
                salience=SalienceProfile(0.5, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
                route_id="timeline-1",
                start_memory_id="mem-1",
                ordered_waypoints=("mem-1",),
                route_kind="timeline",
            )
        )
        planner = DefaultRoutePlanner()
        seeds = (SeedActivation(memory_id="mem-1", score=0.9, reason="seed", space_id="ops"),)
        routes = planner.plan_routes(
            palace,
            RoutePlanningInput(text="narrative first step", top_k=2, max_hops=2),
            seeds,
        )
        self.assertTrue(routes)
        self.assertEqual(routes[0].route_source, "route_memory")
        self.assertEqual(routes[0].step_memory_ids, ("mem-1",))

    def test_legacy_graph_expansion_when_no_route_memory(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="a",
                palace_id="p1",
                location=loc,
                content="Seed node",
                salience=SalienceProfile(0.8, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="b",
                palace_id="p1",
                location=loc,
                content="Neighbor node",
                salience=SalienceProfile(0.7, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_link(
            MemoryLink(
                from_memory_id="a",
                to_memory_id="b",
                link_type=MemoryLinkType.NEXT,
                strength=2.0,
            )
        )
        planner = DefaultRoutePlanner()
        seeds = (SeedActivation(memory_id="a", score=1.0, reason="seed", space_id="ops"),)
        routes = planner.plan_routes(
            palace,
            RoutePlanningInput(text="anything", top_k=1, max_hops=2),
            seeds,
        )
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].route_source, "legacy_path")
        self.assertEqual(routes[0].step_memory_ids, ("a", "b"))

    def test_irrelevant_route_memory_does_not_block_legacy_fallback_path(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="a",
                palace_id="p1",
                location=loc,
                content="Rollback recovery seed node",
                salience=SalienceProfile(0.8, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="b",
                palace_id="p1",
                location=loc,
                content="Queue drain confirmation",
                salience=SalienceProfile(0.7, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            RouteMemory(
                memory_id="route-irrelevant",
                palace_id="p1",
                location=loc,
                content="totally unrelated route memory",
                salience=SalienceProfile(0.4, 0.1, 0.2, 0.8),
                metadata={"space_id": "ops"},
                route_id="route-irrelevant",
                start_memory_id="x",
                ordered_waypoints=("x", "y"),
                route_kind="timeline",
            )
        )
        palace.add_link(
            MemoryLink(
                from_memory_id="a",
                to_memory_id="b",
                link_type=MemoryLinkType.NEXT,
                strength=2.0,
            )
        )
        planner = DefaultRoutePlanner()
        seeds = (SeedActivation(memory_id="a", score=1.0, reason="seed", space_id="ops"),)
        routes = planner.plan_routes(
            palace,
            RoutePlanningInput(text="rollback recovery queue drain", top_k=1, max_hops=2),
            seeds,
        )
        self.assertEqual(routes[0].route_source, "legacy_path")
        self.assertEqual(routes[0].step_memory_ids, ("a", "b"))

    def test_route_kind_bonus_prefers_dependency_route(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="seed",
                palace_id="p1",
                location=loc,
                content="Service recovery depends on queue drain.",
                salience=SalienceProfile(0.8, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            RouteMemory(
                memory_id="route-dep",
                palace_id="p1",
                location=loc,
                content="Queue drain depends on worker restart.",
                salience=SalienceProfile(0.5, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
                route_id="route-dep",
                start_memory_id="seed",
                ordered_waypoints=("seed",),
                route_kind="dependency",
            )
        )
        palace.add_memory(
            RouteMemory(
                memory_id="route-time",
                palace_id="p1",
                location=loc,
                content="First restart, then validate metrics.",
                salience=SalienceProfile(0.5, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
                route_id="route-time",
                start_memory_id="seed",
                ordered_waypoints=("seed",),
                route_kind="timeline",
            )
        )
        routes = DefaultRoutePlanner().plan_routes(
            palace,
            RoutePlanningInput(text="what dependency causes queue drain recovery", top_k=2, max_hops=2),
            (SeedActivation(memory_id="seed", score=0.9, reason="seed", space_id="ops"),),
        )
        self.assertEqual(routes[0].route_kind, "dependency")

    def test_route_kind_bonus_prefers_semantic_summary_route(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="seed",
                palace_id="p1",
                location=loc,
                content="Repeated incident pattern.",
                salience=SalienceProfile(0.8, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            RouteMemory(
                memory_id="route-summary",
                palace_id="p1",
                location=loc,
                content="Abstract summary of the repeated recovery pattern.",
                salience=SalienceProfile(0.5, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
                route_id="route-summary",
                start_memory_id="seed",
                ordered_waypoints=("seed",),
                route_kind="semantic-summary",
            )
        )
        routes = DefaultRoutePlanner().plan_routes(
            palace,
            RoutePlanningInput(text="give me the abstract summary pattern", top_k=1, max_hops=2),
            (SeedActivation(memory_id="seed", score=0.9, reason="seed", space_id="ops"),),
        )
        self.assertEqual(routes[0].route_kind, "semantic-summary")

    def test_legacy_fallback_route_kind_can_be_inferred_from_query(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="a",
                palace_id="p1",
                location=loc,
                content="Base rule",
                salience=SalienceProfile(0.8, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="b",
                palace_id="p1",
                location=loc,
                content="Exception override",
                salience=SalienceProfile(0.7, 0.1, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_link(
            MemoryLink(
                from_memory_id="a",
                to_memory_id="b",
                link_type=MemoryLinkType.EXCEPTION_TO,
                strength=1.5,
            )
        )
        routes = DefaultRoutePlanner().plan_routes(
            palace,
            RoutePlanningInput(text="what exception overrides the base rule", top_k=1, max_hops=1),
            (SeedActivation(memory_id="a", score=1.0, reason="seed", space_id="ops"),),
        )
        self.assertEqual(routes[0].route_kind, "exception")
