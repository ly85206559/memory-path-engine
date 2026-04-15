import unittest

from memory_engine.memory.domain.memory_state import DomainMemoryState
from memory_engine.memory.domain.enums import MemoryLifecycleState
from memory_engine.memory.domain.encoding import EncodingProfile, TriggerProfile
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

    def test_trigger_matching_boosts_seed_ranking(self) -> None:
        self.palace.memories["mem-a"].encoding = EncodingProfile(
            trigger_profile=TriggerProfile(
                phrases=("payment dispute",),
                situations=("during payment dispute",),
            ),
            scenario_tags=("payment dispute",),
            symbolic_tags=("conflict",),
        )
        sel = default_hybrid_seed_selector()
        inp = SeedSelectionInput(
            text="how to resolve a payment dispute",
            allowed_space_ids=(),
            max_seeds=2,
        )
        out = sel.select_seeds(self.palace, inp)
        self.assertEqual(out[0].memory_id, "mem-a")
        self.assertIn("trigger", out[0].reason)

    def test_state_aware_seed_ranking_prefers_active_over_fading(self) -> None:
        self.palace.memories["mem-a"].content = "Recovery verification checklist"
        self.palace.memories["mem-b"].content = "Recovery verification checklist"
        self.palace.memories["mem-a"].state = DomainMemoryState(
            state=MemoryLifecycleState.ACTIVE,
            reinforcement_count=2,
            stability_score=0.4,
            decay_factor=1.0,
        )
        self.palace.memories["mem-b"].state = DomainMemoryState(
            state=MemoryLifecycleState.FADING,
            reinforcement_count=2,
            stability_score=0.2,
            decay_factor=0.75,
        )
        sel = LexicalSeedSelector()
        out = sel.select_seeds(
            self.palace,
            SeedSelectionInput(
                text="recovery verification checklist",
                allowed_space_ids=(),
                max_seeds=2,
            ),
        )
        self.assertEqual(out[0].memory_id, "mem-a")
