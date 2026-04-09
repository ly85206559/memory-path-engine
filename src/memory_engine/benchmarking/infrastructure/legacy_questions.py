from __future__ import annotations

import json
from pathlib import Path

from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkCase,
    StructuredBenchmarkDataset,
    StructuredBenchmarkExpectation,
)


def load_legacy_questions_dataset(
    *,
    questions_path: Path,
    dataset_id: str,
    dataset_name: str,
    domain_pack_name: str,
    document_directory: str,
) -> StructuredBenchmarkDataset:
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    return StructuredBenchmarkDataset(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        domain_pack_name=domain_pack_name,
        document_directory=document_directory,
        cases=[
            StructuredBenchmarkCase(
                case_id=question["id"],
                query=question["query"],
                tags=question.get("tags", []),
                expectation=StructuredBenchmarkExpectation(
                    evidence_node_ids=question["evidence_node_ids"],
                    minimum_evidence_matches=1,
                ),
            )
            for question in payload
        ],
    )
