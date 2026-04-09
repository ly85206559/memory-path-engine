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
        self.assertEqual(len(report.case_reports), 1)
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
