import unittest

from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
from memory_engine.benchmarking.domain.models import StructuredBenchmarkDataset
from memory_engine.memory.application.bridge import palace_to_store
from memory_engine.memory.application.consolidate_memory_service import ConsolidateMemoryService
from memory_engine.memory.application.query_models import RecallPolicy, RecallQuery
from memory_engine.memory.application.retrieve_memory_service import RetrieveMemoryService
from memory_engine.memory.domain.encoding import EncodingProfile, TriggerProfile
from memory_engine.memory.domain.enums import MemoryLifecycleState
from memory_engine.memory.domain.memory_state import DomainMemoryState
from memory_engine.memory.domain.memory_types import EpisodicMemory
from memory_engine.memory.domain.palace import MemoryPalace, PalaceSpace
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile


class ConsolidationBenchmarkTests(unittest.TestCase):
    def _build_palace(self) -> MemoryPalace:
        palace = MemoryPalace(palace_id="bench-palace")
        loc = PalaceLocation(building="hq", floor="1", room="ops", locus="1")
        palace.add_space(PalaceSpace(space_id="ops", name="Ops", location=loc))
        for memory_id, content, participants in (
            (
                "ep-1",
                "Rollback failure required worker restart after queue drain stalled.",
                ("worker-service",),
            ),
            (
                "ep-2",
                "Rollback failure recovery depended on queue drain confirmation and restart validation.",
                ("worker-service", "incident-commander"),
            ),
            (
                "ep-3",
                "Rollback failure triggered escalation when recovery still lagged after restart.",
                ("incident-commander",),
            ),
        ):
            palace.add_memory(
                EpisodicMemory(
                    memory_id=memory_id,
                    palace_id="bench-palace",
                    location=loc,
                    content=content,
                    salience=SalienceProfile(0.88, 0.25, 0.35, 0.94),
                    state=DomainMemoryState(
                        state=MemoryLifecycleState.STABILIZING,
                        reinforcement_count=3,
                        stability_score=0.62,
                    ),
                    encoding=EncodingProfile(
                        trigger_profile=TriggerProfile(phrases=("rollback failure",)),
                        scenario_tags=("rollback failure",),
                        symbolic_tags=("causal",),
                    ),
                    metadata={"space_id": "ops", "semantic_role": "action"},
                    episode_id=memory_id,
                    event_type="incident",
                    participants=participants,
                )
            )
        return palace

    def test_consolidation_improves_generalized_and_cross_episode_recall(self) -> None:
        dataset = StructuredBenchmarkDataset.model_validate(
            {
                "dataset_id": "consolidation-palace-benchmark-v1",
                "dataset_name": "Consolidation palace benchmark",
                "domain_pack_name": "example_runbook_pack",
                "document_directory": "runbooks",
                "cases": [
                    {
                        "case_id": "cg-001",
                        "query": "What generalized rule applies to rollback failure?",
                        "expectation": {
                            "evidence_node_ids": ["semantic:rollback_failure:generalized_rule_memory"],
                            "minimum_evidence_matches": 1,
                            "required_consolidation_kinds": ["generalized_rule_memory"],
                            "required_scenario_tags": ["rollback failure"],
                        },
                    },
                    {
                        "case_id": "cg-002",
                        "query": "What cross episode abstraction links rollback failure incidents?",
                        "expectation": {
                            "evidence_node_ids": ["semantic:rollback_failure:cross_episode_abstraction"],
                            "minimum_evidence_matches": 1,
                            "required_consolidation_kinds": ["cross_episode_abstraction"],
                            "required_scenario_tags": ["rollback failure"],
                        },
                    },
                ],
            }
        )

        before = _PalaceBenchmarkRetriever(self._build_palace(), consolidate=False)
        after = _PalaceBenchmarkRetriever(self._build_palace(), consolidate=True)
        runner = StructuredBenchmarkRunner()

        before_report = runner.run(
            dataset=dataset,
            retriever_name="before_consolidation",
            retriever=before,
            top_k=4,
        )
        after_report = runner.run(
            dataset=dataset,
            retriever_name="after_consolidation",
            retriever=after,
            top_k=4,
        )

        self.assertLess(before_report.evidence_hit_rate, after_report.evidence_hit_rate)
        self.assertEqual(after_report.evidence_hit_rate, 1.0)
        self.assertTrue(all(report.semantic_hit for report in after_report.case_reports))
        self.assertTrue(
            any(
                "generalized_rule_memory" in report.surfaced_consolidation_kinds
                for report in after_report.case_reports
            )
        )
        self.assertTrue(
            any(
                "cross_episode_abstraction" in report.surfaced_consolidation_kinds
                for report in after_report.case_reports
            )
        )


class _PalaceBenchmarkRetriever:
    def __init__(self, palace: MemoryPalace, *, consolidate: bool) -> None:
        self.palace = palace
        if consolidate:
            ConsolidateMemoryService().consolidate(self.palace)
        self.store = palace_to_store(self.palace)
        self.service = RetrieveMemoryService()

    def search(self, query: str, top_k: int = 3):
        result = self.service.recall(
            self.palace,
            RecallQuery(
                palace_id=self.palace.palace_id,
                text=query,
                policy=RecallPolicy(top_k=top_k, max_hops=2),
            ),
        )
        return result.to_legacy_retrieval_result()
