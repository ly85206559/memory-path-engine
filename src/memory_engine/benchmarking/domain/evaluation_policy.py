from __future__ import annotations

from memory_engine.benchmarking.domain.models import (
    PathShapeExpectation,
    PathStepExpectation,
    StructuredBenchmarkExpectation,
)
from memory_engine.schema import MemoryPath, PathStep, RetrievalResult


def collect_returned_node_ids(result: RetrievalResult) -> list[str]:
    return sorted(
        {
            step.node_id
            for path in result.paths
            for step in path.steps
        }
    )


def collect_matched_evidence(
    expectation: StructuredBenchmarkExpectation,
    returned_node_ids: list[str],
) -> list[str]:
    returned_node_id_set = set(returned_node_ids)
    return [
        node_id
        for node_id in expectation.evidence_node_ids
        if node_id in returned_node_id_set
    ]


def evaluate_path_hit(
    expectation: StructuredBenchmarkExpectation,
    result: RetrievalResult,
) -> bool | None:
    if expectation.path is None:
        return None

    candidate_paths = result.paths
    if expectation.path_scope == "best_path":
        candidate_paths = [result.best_path()] if result.paths else []
    return any(_path_matches(path, expectation.path) for path in candidate_paths)


def _path_matches(path: MemoryPath, expectation: PathShapeExpectation) -> bool:
    actual_steps = path.steps
    expected_steps = expectation.steps

    if expectation.match_mode == "exact":
        return len(actual_steps) == len(expected_steps) and all(
            _step_matches(actual_step, expected_step)
            for actual_step, expected_step in zip(actual_steps, expected_steps)
        )

    if expectation.match_mode == "prefix":
        return len(actual_steps) >= len(expected_steps) and all(
            _step_matches(actual_step, expected_step)
            for actual_step, expected_step in zip(actual_steps, expected_steps)
        )

    expected_index = 0
    for actual_step in actual_steps:
        if _step_matches(actual_step, expected_steps[expected_index]):
            expected_index += 1
            if expected_index == len(expected_steps):
                return True
    return False


def _step_matches(actual_step: PathStep, expected_step: PathStepExpectation) -> bool:
    if actual_step.node_id != expected_step.node_id:
        return False
    if expected_step.via_edge_type is None:
        return True
    return actual_step.via_edge_type == expected_step.via_edge_type
