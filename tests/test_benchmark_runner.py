import json
import tempfile
import unittest
from pathlib import Path


class StructuredBenchmarkRunnerTests(unittest.TestCase):
    def test_runner_returns_typed_summary_and_case_reports(self):
        from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
        from memory_engine.benchmarking.infrastructure.json_repository import (
            JsonStructuredBenchmarkDatasetRepository,
        )
        from memory_engine.domain_pack import get_domain_pack
        from memory_engine.ingest import ingest_document
        from memory_engine.retrieve import WeightedGraphRetriever
        from memory_engine.store import MemoryStore

        dataset_payload = {
            "dataset_id": "runbook-benchmark-v1",
            "dataset_name": "Runbook benchmark",
            "domain_pack_name": "example_runbook_pack",
            "document_directory": "runbooks",
            "cases": [
                {
                    "case_id": "rb-001",
                    "query": "What should we do if rollback does not recover the service?",
                    "expectation": {
                        "evidence_node_ids": ["01_api_incident_runbook:2"],
                        "minimum_evidence_matches": 1,
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runbooks_dir = root / "runbooks"
            runbooks_dir.mkdir()
            (runbooks_dir / "01_api_incident_runbook.md").write_text(
                "\n".join(
                    [
                        "# API Incident Runbook",
                        "",
                        "## Mitigation",
                        "1 If the latest deployment is implicated, roll back the release and verify latency recovery.",
                        "2 If rollback does not recover service, restart the worker service and confirm queue drain behavior.",
                    ]
                ),
                encoding="utf-8",
            )

            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")

            dataset = JsonStructuredBenchmarkDatasetRepository().load(dataset_path)

            store = MemoryStore()
            for path in runbooks_dir.glob("*.md"):
                ingest_document(path, store, domain_pack=get_domain_pack(dataset.domain_pack_name))

            runner = StructuredBenchmarkRunner()
            report = runner.run(
                dataset=dataset,
                retriever_name="weighted_graph",
                retriever=WeightedGraphRetriever(store),
                top_k=3,
            )

        self.assertEqual(report.dataset_id, "runbook-benchmark-v1")
        self.assertEqual(report.retriever_name, "weighted_graph")
        self.assertEqual(report.questions, 1)
        self.assertEqual(report.evidence_hit_rate, 1.0)
        self.assertEqual(len(report.case_reports), 1)
        self.assertTrue(report.case_reports[0].evidence_hit)
        self.assertTrue(report.case_reports[0].hit)

    def test_runner_supports_optional_path_expectation(self):
        from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
        from memory_engine.benchmarking.domain.models import StructuredBenchmarkDataset
        from memory_engine.schema import MemoryPath, PathStep, RetrievalResult

        class FakeRetriever:
            def search(self, query: str, top_k: int = 3) -> RetrievalResult:
                del query, top_k
                return RetrievalResult(
                    query="rollback recovery",
                    paths=[
                        MemoryPath(
                            query="rollback recovery",
                            steps=[
                                PathStep(
                                    node_id="01_api_incident_runbook:4",
                                    reason="seed",
                                    score=0.8,
                                ),
                                PathStep(
                                    node_id="01_api_incident_runbook:5",
                                    reason="expansion",
                                    score=0.9,
                                    via_edge_type="depends_on",
                                ),
                            ],
                            final_answer="Restart the worker service and confirm queue drain behavior.",
                            final_score=0.9,
                        )
                    ],
                )

        dataset = StructuredBenchmarkDataset.model_validate(
            {
                "dataset_id": "path-benchmark-v1",
                "dataset_name": "Path benchmark",
                "domain_pack_name": "example_runbook_pack",
                "document_directory": "runbooks",
                "cases": [
                    {
                        "case_id": "rb-path-001",
                        "query": "What should happen after rollback does not recover service?",
                        "expectation": {
                            "evidence_node_ids": ["01_api_incident_runbook:5"],
                            "minimum_evidence_matches": 1,
                            "path_scope": "best_path",
                            "path": {
                                "match_mode": "prefix",
                                "steps": [
                                    {"node_id": "01_api_incident_runbook:4"},
                                    {
                                        "node_id": "01_api_incident_runbook:5",
                                        "via_edge_type": "depends_on",
                                    },
                                ],
                            },
                        },
                    }
                ],
            }
        )

        report = StructuredBenchmarkRunner().run(
            dataset=dataset,
            retriever_name="fake_weighted_graph",
            retriever=FakeRetriever(),
            top_k=3,
        )

        self.assertTrue(report.case_reports[0].hit)
        self.assertTrue(report.case_reports[0].path_hit)

    def test_runner_supports_activation_trace_expectation(self):
        from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
        from memory_engine.benchmarking.domain.models import StructuredBenchmarkDataset
        from memory_engine.schema import (
            ActivationTraceStep,
            MemoryPath,
            PathStep,
            RetrievalResult,
        )

        class FakeRetriever:
            def search(self, query: str, top_k: int = 3) -> RetrievalResult:
                del query, top_k
                return RetrievalResult(
                    query="rollback recovery",
                    paths=[
                        MemoryPath(
                            query="rollback recovery",
                            steps=[
                                PathStep(
                                    node_id="01_api_incident_runbook:4",
                                    reason="seed",
                                    score=0.8,
                                ),
                                PathStep(
                                    node_id="01_api_incident_runbook:5",
                                    reason="expansion",
                                    score=0.9,
                                    via_edge_type="depends_on",
                                ),
                            ],
                            activation_trace=[
                                ActivationTraceStep(
                                    node_id="01_api_incident_runbook:4",
                                    hop=0,
                                    incoming_activation=0.82,
                                    propagated_activation=0.82,
                                    is_seed=True,
                                ),
                                ActivationTraceStep(
                                    node_id="01_api_incident_runbook:5",
                                    source_node_id="01_api_incident_runbook:4",
                                    edge_type="depends_on",
                                    hop=1,
                                    incoming_activation=0.74,
                                    propagated_activation=0.74,
                                    activated_score=0.91,
                                ),
                                ActivationTraceStep(
                                    node_id="01_api_incident_runbook:6",
                                    source_node_id="01_api_incident_runbook:5",
                                    edge_type="related_to",
                                    hop=2,
                                    incoming_activation=0.04,
                                    propagated_activation=0.0,
                                    stopped_reason="below_threshold",
                                ),
                            ],
                            final_answer="Restart the worker service and confirm queue drain behavior.",
                            final_score=0.9,
                        )
                    ],
                )

        dataset = StructuredBenchmarkDataset.model_validate(
            {
                "dataset_id": "trace-benchmark-v1",
                "dataset_name": "Trace benchmark",
                "domain_pack_name": "example_runbook_pack",
                "document_directory": "runbooks",
                "cases": [
                    {
                        "case_id": "rb-trace-001",
                        "query": "What should happen after rollback does not recover service?",
                        "expectation": {
                            "evidence_node_ids": ["01_api_incident_runbook:5"],
                            "minimum_evidence_matches": 1,
                            "min_activation_trace_length": 3,
                            "max_activation_trace_length": 3,
                            "required_trace_stop_reasons": ["below_threshold"],
                            "activation_trace": {
                                "match_mode": "prefix",
                                "steps": [
                                    {
                                        "node_id": "01_api_incident_runbook:4",
                                        "is_seed": True,
                                        "hop": 0,
                                    },
                                    {
                                        "node_id": "01_api_incident_runbook:5",
                                        "edge_type": "depends_on",
                                        "hop": 1,
                                    },
                                ],
                            },
                        },
                    }
                ],
            }
        )

        report = StructuredBenchmarkRunner().run(
            dataset=dataset,
            retriever_name="fake_activation_spreading",
            retriever=FakeRetriever(),
            top_k=3,
        )

        self.assertTrue(report.case_reports[0].hit)
        self.assertTrue(report.case_reports[0].activation_trace_hit)
        self.assertEqual(report.case_reports[0].activation_trace_length, 3)
        self.assertEqual(
            report.case_reports[0].activation_stopped_reasons,
            ["below_threshold"],
        )

    def test_runner_supports_route_and_activation_snapshot_expectations(self):
        from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
        from memory_engine.benchmarking.domain.models import StructuredBenchmarkDataset
        from memory_engine.memory.domain.retrieval_result import (
            ActivationSnapshot,
            ActivationSnapshotEntry,
            PalaceRecallResult,
            RecallRoute,
            RetrievedMemory,
        )

        class FakeRetriever:
            def search(self, query: str, top_k: int = 3):
                del query, top_k
                palace_result = PalaceRecallResult(
                    query="rollback recovery",
                    retrieved_memories=(
                        RetrievedMemory(memory_id="01_api_incident_runbook:4", score=0.8, reason="seed"),
                        RetrievedMemory(memory_id="01_api_incident_runbook:5", score=0.9, reason="route"),
                    ),
                    routes=(
                        RecallRoute(
                            route_id="route-1",
                            route_kind="dependency",
                            step_memory_ids=("01_api_incident_runbook:4", "01_api_incident_runbook:5"),
                            score=0.9,
                        ),
                    ),
                    activation_snapshot=ActivationSnapshot(
                        steps=(
                            ActivationSnapshotEntry(
                                memory_id="01_api_incident_runbook:4",
                                hop=0,
                                incoming_activation=0.82,
                                propagated_activation=0.82,
                                is_seed=True,
                            ),
                            ActivationSnapshotEntry(
                                memory_id="01_api_incident_runbook:5",
                                source_memory_id="01_api_incident_runbook:4",
                                edge_type="depends_on",
                                hop=1,
                                incoming_activation=0.74,
                                propagated_activation=0.74,
                                activated_score=0.91,
                            ),
                        )
                    ),
                )
                return palace_result.to_legacy_retrieval_result()

        dataset = StructuredBenchmarkDataset.model_validate(
            {
                "dataset_id": "route-snapshot-benchmark-v1",
                "dataset_name": "Route snapshot benchmark",
                "domain_pack_name": "example_runbook_pack",
                "document_directory": "runbooks",
                "cases": [
                    {
                        "case_id": "rb-route-001",
                        "query": "What should happen after rollback does not recover service?",
                        "expectation": {
                            "evidence_node_ids": ["01_api_incident_runbook:5"],
                            "minimum_evidence_matches": 1,
                            "route": {
                                "match_mode": "prefix",
                                "steps": [
                                    {"node_id": "01_api_incident_runbook:4"},
                                    {"node_id": "01_api_incident_runbook:5"},
                                ],
                            },
                            "activation_snapshot": {
                                "match_mode": "prefix",
                                "steps": [
                                    {"node_id": "01_api_incident_runbook:4", "hop": 0, "is_seed": True},
                                    {
                                        "node_id": "01_api_incident_runbook:5",
                                        "edge_type": "depends_on",
                                        "hop": 1,
                                    },
                                ],
                            },
                        },
                    }
                ],
            }
        )

        report = StructuredBenchmarkRunner().run(
            dataset=dataset,
            retriever_name="fake_palace",
            retriever=FakeRetriever(),
            top_k=3,
        )

        self.assertTrue(report.case_reports[0].route_hit)
        self.assertTrue(report.case_reports[0].activation_snapshot_hit)
        self.assertTrue(report.case_reports[0].hit)

    def test_runner_reports_evidence_hit_even_when_full_hit_fails(self):
        from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
        from memory_engine.benchmarking.domain.models import StructuredBenchmarkDataset
        from memory_engine.schema import MemoryPath, PathStep, RetrievalResult

        class FakeRetriever:
            def search(self, query: str, top_k: int = 3) -> RetrievalResult:
                del query, top_k
                return RetrievalResult(
                    query="rollback recovery",
                    paths=[
                        MemoryPath(
                            query="rollback recovery",
                            steps=[
                                PathStep(
                                    node_id="01_api_incident_runbook:4",
                                    reason="seed",
                                    score=0.8,
                                ),
                                PathStep(
                                    node_id="01_api_incident_runbook:5",
                                    reason="expansion",
                                    score=0.9,
                                    via_edge_type="depends_on",
                                ),
                            ],
                            final_answer="Restart the worker service and confirm queue drain behavior.",
                            final_score=0.9,
                        )
                    ],
                )

        dataset = StructuredBenchmarkDataset.model_validate(
            {
                "dataset_id": "metric-honesty-benchmark-v1",
                "dataset_name": "Metric honesty benchmark",
                "domain_pack_name": "example_runbook_pack",
                "document_directory": "runbooks",
                "cases": [
                    {
                        "case_id": "rb-metric-001",
                        "query": "What should happen after rollback does not recover service?",
                        "expectation": {
                            "evidence_node_ids": ["01_api_incident_runbook:5"],
                            "minimum_evidence_matches": 1,
                            "path_scope": "best_path",
                            "path": {
                                "match_mode": "prefix",
                                "steps": [
                                    {"node_id": "01_api_incident_runbook:4"},
                                    {
                                        "node_id": "01_api_incident_runbook:5",
                                        "via_edge_type": "exception_to",
                                    },
                                ],
                            },
                        },
                    }
                ],
            }
        )

        report = StructuredBenchmarkRunner().run(
            dataset=dataset,
            retriever_name="fake_weighted_graph",
            retriever=FakeRetriever(),
            top_k=3,
        )

        self.assertEqual(report.evidence_hit_rate, 1.0)
        self.assertEqual(report.evidence_recall, 0.0)
        self.assertTrue(report.case_reports[0].evidence_hit)
        self.assertFalse(report.case_reports[0].path_hit)
        self.assertFalse(report.case_reports[0].hit)

    def test_evaluation_service_runs_dataset_file_end_to_end(self):
        from memory_engine.benchmarking.application.service import (
            StructuredBenchmarkEvaluationService,
        )

        dataset_payload = {
            "dataset_id": "runbook-benchmark-v1",
            "dataset_name": "Runbook benchmark",
            "domain_pack_name": "example_runbook_pack",
            "document_directory": "runbooks",
            "cases": [
                {
                    "case_id": "rb-001",
                    "query": "What should we do if rollback does not recover the service?",
                    "expectation": {
                        "evidence_node_ids": ["01_api_incident_runbook:2"],
                        "minimum_evidence_matches": 1,
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runbooks_dir = root / "runbooks"
            runbooks_dir.mkdir()
            (runbooks_dir / "01_api_incident_runbook.md").write_text(
                "\n".join(
                    [
                        "# API Incident Runbook",
                        "",
                        "## Mitigation",
                        "1 If the latest deployment is implicated, roll back the release and verify latency recovery.",
                        "2 If rollback does not recover service, restart the worker service and confirm queue drain behavior.",
                    ]
                ),
                encoding="utf-8",
            )

            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")

            report = StructuredBenchmarkEvaluationService().run_from_dataset_path(
                dataset_path=dataset_path,
                retriever_mode="weighted_graph",
                top_k=3,
            )

        self.assertEqual(report.dataset_id, "runbook-benchmark-v1")
        self.assertEqual(report.retriever_name, "weighted_graph")
        self.assertTrue(report.case_reports[0].hit)
