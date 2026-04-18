import unittest

from memory_engine.memory.application.result_ranker_service import DefaultRecallResultRanker
from memory_engine.memory.domain.memory_types import EpisodicMemory
from memory_engine.memory.domain.palace import MemoryPalace, PalaceSpace
from memory_engine.memory.domain.retrieval_result import RecallRoute, RetrievedMemory
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile


class ResultRankerServiceTests(unittest.TestCase):
    def test_query_alignment_reranks_retrieved_memories(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops Center", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="generic",
                palace_id="p1",
                location=loc,
                content="Audit summary for weekly operations review.",
                salience=SalienceProfile(0.8, 0.2, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="target",
                palace_id="p1",
                location=loc,
                content="Worker restart is required after rollback failure.",
                salience=SalienceProfile(0.8, 0.2, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )

        bundle = DefaultRecallResultRanker().rank(
            palace,
            "what should we do after rollback failure",
            seeds=(),
            activated_memories=(
                RetrievedMemory(memory_id="generic", score=0.92, reason="activation"),
                RetrievedMemory(memory_id="target", score=0.81, reason="activation"),
            ),
            routes=(),
            top_k=2,
        )

        self.assertEqual(bundle.retrieved_memories[0].memory_id, "target")

    def test_query_alignment_reranks_routes(self) -> None:
        palace = MemoryPalace(palace_id="p2")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops Center", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="timeline",
                palace_id="p2",
                location=loc,
                content="Weekly audit summary and staffing review.",
                salience=SalienceProfile(0.8, 0.2, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id="dependency",
                palace_id="p2",
                location=loc,
                content="Restart the worker service if rollback does not recover traffic.",
                salience=SalienceProfile(0.8, 0.2, 0.2, 0.9),
                metadata={"space_id": "ops"},
            )
        )

        bundle = DefaultRecallResultRanker().rank(
            palace,
            "if rollback does not recover traffic, what should we restart",
            seeds=(),
            activated_memories=(),
            routes=(
                RecallRoute(
                    route_id="r1",
                    route_kind="timeline",
                    step_memory_ids=("timeline",),
                    score=0.91,
                    route_source="native_activation",
                ),
                RecallRoute(
                    route_id="r2",
                    route_kind="dependency",
                    step_memory_ids=("dependency",),
                    score=0.8,
                    route_source="native_activation",
                ),
            ),
            top_k=2,
        )

        self.assertEqual(bundle.routes[0].route_id, "r2")
        self.assertEqual(bundle.retrieved_memories[0].memory_id, "dependency")
