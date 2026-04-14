import unittest

from memory_engine.memory.application.seed_selection_service import (
    LexicalSeedSelector,
    default_hybrid_seed_selector,
)
from memory_engine.memory.domain.enums import MemoryKind
from memory_engine.memory.domain.memory_types import EpisodicMemory
from memory_engine.memory.domain.palace import MemoryPalace, PalaceSpace
from memory_engine.memory.domain.seed_selection import SeedSelectionInput
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile


class SeedSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.palace = MemoryPalace(palace_id="p1")
        loc_a = PalaceLocation(building="a", floor="1", room="r1", locus="x")
        loc_b = PalaceLocation(building="b", floor="1", room="r2", locus="y")
        self.palace.add_space(PalaceSpace(space_id="s-a", name="Space A", location=loc_a))
        self.palace.add_space(PalaceSpace(space_id="s-b", name="Space B", location=loc_b))
        self.palace.add_memory(
            EpisodicMemory(
                memory_id="mem-a",
                palace_id="p1",
                location=loc_a,
                content="Payment terms are net thirty days.",
                salience=SalienceProfile(0.7, 0.1, 0.2, 0.9),
                metadata={"space_id": "s-a"},
            )
        )
        self.palace.add_memory(
            EpisodicMemory(
                memory_id="mem-b",
                palace_id="p1",
                location=loc_b,
                content="Defective goods must be returned within fourteen days.",
                salience=SalienceProfile(0.7, 0.1, 0.2, 0.9),
                metadata={"space_id": "s-b"},
            )
        )

    def test_seeds_respect_allowed_space_ids(self) -> None:
        sel = LexicalSeedSelector()
        inp = SeedSelectionInput(
            text="payment terms net thirty",
            allowed_space_ids=("s-b",),
            max_seeds=3,
        )
        out = sel.select_seeds(self.palace, inp)
        self.assertEqual(len(out), 0)

        inp2 = SeedSelectionInput(
            text="payment terms net thirty",
            allowed_space_ids=("s-a",),
            max_seeds=3,
        )
        out2 = sel.select_seeds(self.palace, inp2)
        self.assertEqual(out2[0].memory_id, "mem-a")
        self.assertEqual(out2[0].space_id, "s-a")

    def test_empty_allowed_space_ids_searches_all_spaces(self) -> None:
        sel = LexicalSeedSelector()
        inp = SeedSelectionInput(
            text="defective goods return",
            allowed_space_ids=(),
            max_seeds=2,
        )
        out = sel.select_seeds(self.palace, inp)
        self.assertEqual(out[0].memory_id, "mem-b")

    def test_allowed_memory_kinds_filters_results(self) -> None:
        sel = default_hybrid_seed_selector()
        inp = SeedSelectionInput(
            text="payment net thirty",
            allowed_space_ids=(),
            max_seeds=2,
            allowed_memory_kinds=(MemoryKind.SEMANTIC,),
        )
        out = sel.select_seeds(self.palace, inp)
        self.assertEqual(out, ())
