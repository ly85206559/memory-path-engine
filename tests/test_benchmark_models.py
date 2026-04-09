import json
import tempfile
import unittest
from pathlib import Path


class StructuredBenchmarkModelTests(unittest.TestCase):
    def test_json_repository_loads_typed_dataset(self):
        from memory_engine.benchmarking.infrastructure.json_repository import (
            JsonStructuredBenchmarkDatasetRepository,
        )

        dataset_payload = {
            "dataset_id": "runbook-benchmark-v1",
            "dataset_name": "Runbook benchmark",
            "domain_pack_name": "example_runbook_pack",
            "document_directory": "examples/runbook_pack/runbooks",
            "cases": [
                {
                    "case_id": "rb-001",
                    "query": "What should we do if rollback does not recover the service?",
                    "tags": ["runbook", "rollback"],
                    "expectation": {
                        "evidence_node_ids": ["01_api_incident_runbook:5"],
                        "minimum_evidence_matches": 1,
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dataset.json"
            path.write_text(json.dumps(dataset_payload), encoding="utf-8")

            dataset = JsonStructuredBenchmarkDatasetRepository().load(path)

        self.assertEqual(dataset.dataset_id, "runbook-benchmark-v1")
        self.assertEqual(dataset.domain_pack_name, "example_runbook_pack")
        self.assertEqual(dataset.cases[0].expectation.evidence_node_ids, ["01_api_incident_runbook:5"])

    def test_dataset_rejects_empty_expected_evidence(self):
        from pydantic import ValidationError

        from memory_engine.benchmarking.domain.models import (
            StructuredBenchmarkCase,
            StructuredBenchmarkExpectation,
        )

        with self.assertRaises(ValidationError):
            StructuredBenchmarkCase(
                case_id="invalid-case",
                query="What is the answer?",
                expectation=StructuredBenchmarkExpectation(
                    evidence_node_ids=[],
                    minimum_evidence_matches=1,
                ),
            )

    def test_expectation_accepts_optional_path_shape(self):
        from memory_engine.benchmarking.domain.models import StructuredBenchmarkExpectation

        expectation = StructuredBenchmarkExpectation(
            evidence_node_ids=["runbook:1", "runbook:2"],
            minimum_evidence_matches=1,
            required_edge_types=["depends_on"],
            required_semantic_roles=["escalation"],
            path_scope="best_path",
            path={
                "match_mode": "subsequence",
                "steps": [
                    {"node_id": "runbook:1"},
                    {"node_id": "runbook:2", "via_edge_type": "depends_on"},
                ],
            },
        )

        self.assertEqual(expectation.path_scope, "best_path")
        self.assertEqual(expectation.path.steps[1].via_edge_type, "depends_on")
        self.assertEqual(expectation.required_semantic_roles, ["escalation"])

    def test_path_step_rejects_blank_edge_type(self):
        from pydantic import ValidationError

        from memory_engine.benchmarking.domain.models import StructuredBenchmarkExpectation

        with self.assertRaises(ValidationError):
            StructuredBenchmarkExpectation(
                evidence_node_ids=["runbook:1"],
                minimum_evidence_matches=1,
                path={
                    "steps": [
                        {"node_id": "runbook:1", "via_edge_type": "   "},
                    ],
                },
            )
