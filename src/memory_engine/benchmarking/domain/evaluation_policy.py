from __future__ import annotations

from memory_engine.benchmarking.domain.models import (
    ActivationTraceShapeExpectation,
    ActivationTraceStepExpectation,
    PathShapeExpectation,
    PathStepExpectation,
    StructuredBenchmarkExpectation,
)
from memory_engine.schema import ActivationTraceStep, MemoryPath, PathStep, RetrievalResult


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


def evaluate_activation_trace_hit(
    expectation: StructuredBenchmarkExpectation,
    *,
    activation_trace: list[ActivationTraceStep],
) -> bool | None:
    checks = []
    if expectation.activation_trace is not None:
        checks.append(_activation_trace_matches(activation_trace, expectation.activation_trace))
    if expectation.required_trace_stop_reasons:
        actual_reasons = {
            step.stopped_reason
            for step in activation_trace
            if step.stopped_reason is not None
        }
        checks.append(
            all(reason in actual_reasons for reason in expectation.required_trace_stop_reasons)
        )
    if expectation.min_activation_trace_length is not None:
        checks.append(len(activation_trace) >= expectation.min_activation_trace_length)
    if expectation.max_activation_trace_length is not None:
        checks.append(len(activation_trace) <= expectation.max_activation_trace_length)
    if not checks:
        return None
    return all(checks)


def evaluate_semantic_hit(
    expectation: StructuredBenchmarkExpectation,
    *,
    surfaced_semantic_roles: list[str],
    path_edge_types: list[str],
) -> bool | None:
    checks = []
    if expectation.required_semantic_roles:
        checks.append(
            all(role in surfaced_semantic_roles for role in expectation.required_semantic_roles)
        )
    if expectation.required_edge_types:
        checks.append(
            all(edge_type in path_edge_types for edge_type in expectation.required_edge_types)
        )
    if not checks:
        return None
    return all(checks)


def evaluate_contradiction_hit(
    expectation: StructuredBenchmarkExpectation,
    *,
    surfaced_contradictions: list[tuple[str, str]],
) -> bool | None:
    if not expectation.required_contradiction_pairs:
        return None

    surfaced_pairs = {tuple(sorted(pair)) for pair in surfaced_contradictions}
    expected_pairs = {
        tuple(sorted(pair))
        for pair in expectation.required_contradiction_pairs
    }
    return expected_pairs.issubset(surfaced_pairs)


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


def _activation_trace_matches(
    actual_steps: list[ActivationTraceStep],
    expectation: ActivationTraceShapeExpectation,
) -> bool:
    expected_steps = expectation.steps
    if expectation.match_mode == "exact":
        return len(actual_steps) == len(expected_steps) and all(
            _activation_trace_step_matches(actual_step, expected_step)
            for actual_step, expected_step in zip(actual_steps, expected_steps)
        )

    if expectation.match_mode == "prefix":
        return len(actual_steps) >= len(expected_steps) and all(
            _activation_trace_step_matches(actual_step, expected_step)
            for actual_step, expected_step in zip(actual_steps, expected_steps)
        )

    expected_index = 0
    for actual_step in actual_steps:
        if _activation_trace_step_matches(actual_step, expected_steps[expected_index]):
            expected_index += 1
            if expected_index == len(expected_steps):
                return True
    return False


def _activation_trace_step_matches(
    actual_step: ActivationTraceStep,
    expected_step: ActivationTraceStepExpectation,
) -> bool:
    if actual_step.node_id != expected_step.node_id:
        return False
    if expected_step.edge_type is not None and actual_step.edge_type != expected_step.edge_type:
        return False
    if expected_step.hop is not None and actual_step.hop != expected_step.hop:
        return False
    if expected_step.is_seed is not None and actual_step.is_seed != expected_step.is_seed:
        return False
    if (
        expected_step.stopped_reason is not None
        and actual_step.stopped_reason != expected_step.stopped_reason
    ):
        return False
    return True
