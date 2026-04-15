import unittest

from memory_engine.memory.application.space_selection_service import (
    HybridSpaceSelector,
    KeywordSpaceSelector,
    MetadataSpaceSelector,
)
from memory_engine.memory.domain.memory_types import EpisodicMemory
from memory_engine.memory.domain.palace import MemoryPalace, PalaceSpace
from memory_engine.memory.domain.space_selection import SpaceSelectionInput
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile


class SpaceSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.palace = MemoryPalace(palace_id="p1")
        loc_ops = PalaceLocation(building="hq", floor="1", room="ops", locus="a")
        loc_meet = PalaceLocation(building="hq", floor="2", room="meeting", locus="b")
        self.palace.add_space(
            PalaceSpace(space_id="ops", name="Operations war room", location=loc_ops, tags=("incident",)),
        )
        self.palace.add_space(
            PalaceSpace(space_id="meet", name="Meeting room", location=loc_meet, tags=("standup",)),
        )
        self.palace.add_memory(
            EpisodicMemory(
                memory_id="m-ops",
                palace_id="p1",
                location=loc_ops,
                content="Rollback failure requires queue restart.",
                salience=SalienceProfile(0.8, 0.2, 0.3, 0.9),
                metadata={"space_id": "ops"},
            )
        )
        self.palace.add_memory(
            EpisodicMemory(
                memory_id="m-meet",
                palace_id="p1",
                location=loc_meet,
                content="Daily standup notes.",
                salience=SalienceProfile(0.5, 0.1, 0.2, 0.8),
                metadata={"space_id": "meet"},
            )
        )

    def test_keyword_selector_prefers_matching_space_name(self) -> None:
        sel = KeywordSpaceSelector()
        inp = SpaceSelectionInput(text="operations war room incident", max_spaces=2)
        out = sel.select_spaces(self.palace, inp)
        self.assertGreaterEqual(len(out), 1)
        self.assertEqual(out[0].space_id, "ops")
        self.assertGreater(out[0].score, 0.0)

    def test_metadata_selector_uses_memory_content_per_space(self) -> None:
        sel = MetadataSpaceSelector()
        inp = SpaceSelectionInput(text="rollback queue restart", max_spaces=2)
        out = sel.select_spaces(self.palace, inp)
        self.assertEqual(out[0].space_id, "ops")

    def test_preferred_space_ids_boost_ranking(self) -> None:
        sel = HybridSpaceSelector(
            keyword_selector=KeywordSpaceSelector(),
            metadata_selector=MetadataSpaceSelector(),
        )
        inp = SpaceSelectionInput(
            text="unrelated xyzabc",
            preferred_space_ids=("meet",),
            max_spaces=2,
        )
        out = sel.select_spaces(self.palace, inp)
        self.assertEqual(out[0].space_id, "meet")

    def test_zero_hit_fallback_returns_all_spaces_not_truncated(self) -> None:
        sel = KeywordSpaceSelector()
        inp = SpaceSelectionInput(text="totally unrelated query", max_spaces=1)
        out = sel.select_spaces(self.palace, inp)
        self.assertEqual({candidate.space_id for candidate in out}, {"ops", "meet"})
        self.assertTrue(all(candidate.reason == "fallback_all_spaces" for candidate in out))

    def test_location_prior_matches_building_and_room_tokens(self) -> None:
        sel = HybridSpaceSelector(
            keyword_selector=KeywordSpaceSelector(),
            metadata_selector=MetadataSpaceSelector(),
        )
        inp = SpaceSelectionInput(text="hq meeting follow-up", max_spaces=1)
        out = sel.select_spaces(self.palace, inp)
        self.assertEqual(out[0].space_id, "meet")
        self.assertIn("location_prior", out[0].reason)
