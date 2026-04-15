import unittest

from memory_engine.memory.application.bridge import palace_to_store, store_to_palace
from memory_engine.memory.application.encoding_service import (
    build_encoding_profile,
    infer_scenario_tags,
    infer_symbolic_tags,
    trigger_match_score,
)
from memory_engine.memory.domain.encoding import EncodingProfile, TriggerProfile
from memory_engine.memory.domain.memory_types import EpisodicMemory
from memory_engine.memory.domain.palace import MemoryPalace, PalaceSpace
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile


class MemoryEncodingTests(unittest.TestCase):
    def test_build_encoding_profile_infers_trigger_scenario_and_symbolic_tags(self) -> None:
        profile = build_encoding_profile(
            "Unless rollback failure is resolved, escalate immediately after queue drain stalls.",
            semantic_role="exception",
        )
        self.assertIn("rollback failure", profile.trigger_profile.phrases)
        self.assertIn("rollback failure", " ".join(profile.trigger_profile.phrases))
        self.assertIn("rollback failure", " ".join(profile.trigger_profile.situations))
        self.assertIn("rollback failure", profile.scenario_tags)
        self.assertIn("exception", profile.symbolic_tags)
        self.assertIn("urgency", profile.symbolic_tags)

    def test_trigger_match_score_rewards_encoded_signals(self) -> None:
        profile = EncodingProfile(
            trigger_profile=TriggerProfile(
                phrases=("rollback failure",),
                situations=("when urgency is high",),
            ),
            scenario_tags=("rollback failure",),
            symbolic_tags=("urgency",),
        )
        score = trigger_match_score("rollback failure is urgent", profile)
        self.assertGreater(score, 0.5)

    def test_palace_bridge_round_trip_preserves_encoding_profile(self) -> None:
        palace = MemoryPalace(palace_id="p1")
        loc = PalaceLocation(building="hq", room="ops")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        palace.add_memory(
            EpisodicMemory(
                memory_id="m1",
                palace_id="p1",
                location=loc,
                content="Rollback failure requires immediate escalation.",
                salience=SalienceProfile(0.8, 0.2, 0.3, 0.9),
                encoding=EncodingProfile(
                    trigger_profile=TriggerProfile(
                        phrases=("rollback failure",),
                        situations=("when escalation is required",),
                    ),
                    scenario_tags=("rollback failure",),
                    symbolic_tags=("escalation", "urgency"),
                ),
                metadata={"space_id": "ops"},
            )
        )
        restored = store_to_palace(palace_to_store(palace), palace_id="p1")
        restored_encoding = restored.memories["m1"].encoding
        self.assertEqual(restored_encoding.trigger_profile.phrases, ("rollback failure",))
        self.assertIn("urgency", restored_encoding.symbolic_tags)

    def test_direct_inference_helpers_cover_scenarios_and_tags(self) -> None:
        scenarios = infer_scenario_tags("Payment dispute caused by defective goods.")
        tags = infer_symbolic_tags("Escalate immediately because the override exception applies.")
        self.assertIn("payment dispute", scenarios)
        self.assertIn("escalation", tags)
        self.assertIn("causal", tags)
