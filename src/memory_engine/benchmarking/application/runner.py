from __future__ import annotations

from time import perf_counter

from memory_engine.benchmarking.domain.evaluation_policy import (
    collect_matched_evidence,
    evaluate_contradiction_hit,
    evaluate_activation_trace_hit,
    collect_returned_node_ids,
    evaluate_path_hit,
    evaluate_semantic_hit,
)
from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkCaseReport,
    StructuredBenchmarkDataset,
    StructuredBenchmarkReport,
)
from memory_engine.semantics import (
    contradiction_candidates,
    semantic_score_signals,
    surfaced_contradictions,
)


class StructuredBenchmarkRunner:
    def run(
        self,
        *,
        dataset: StructuredBenchmarkDataset,
        retriever_name: str,
        retriever,
        top_k: int = 3,
    ) -> StructuredBenchmarkReport:
        case_reports: list[StructuredBenchmarkCaseReport] = []

        for case in dataset.cases:
            started = perf_counter()
            result = retriever.search(case.query, top_k=top_k)
            latency_ms = (perf_counter() - started) * 1000

            returned_node_ids = collect_returned_node_ids(result)
            matched_evidence = collect_matched_evidence(case.expectation, returned_node_ids)
            evidence_hit = len(matched_evidence) >= case.expectation.minimum_evidence_matches
            path_hit = evaluate_path_hit(case.expectation, result)
            store = getattr(retriever, "store", None)
            contradiction_pairs = (
                contradiction_candidates(store.nodes(), store.edges())
                if store is not None
                else []
            )
            surfaced_semantic_roles = sorted(
                {
                    store.get_node(node_id).attributes.get("semantic_role")
                    for node_id in returned_node_ids
                    if store is not None and store.get_node(node_id).attributes.get("semantic_role")
                }
            )
            best_path = result.best_path() if result.paths else None
            path_edge_types = [
                step.via_edge_type
                for step in (best_path.steps if best_path else [])
                if step.via_edge_type is not None
            ]
            activation_trace = best_path.activation_trace if best_path is not None else []
            activation_stopped_reasons = sorted(
                {
                    step.stopped_reason
                    for step in activation_trace
                    if step.stopped_reason is not None
                }
            )
            semantic_hit = evaluate_semantic_hit(
                case.expectation,
                surfaced_semantic_roles=surfaced_semantic_roles,
                path_edge_types=path_edge_types,
            )
            activation_trace_hit = evaluate_activation_trace_hit(
                case.expectation,
                activation_trace=activation_trace,
            )
            surfaced_contradiction_pairs = surfaced_contradictions(
                returned_node_ids,
                contradiction_pairs,
            )
            best_path_exception_score = 0.0
            best_path_contradiction_score = 0.0
            if best_path is not None and store is not None:
                previous_node_id = None
                for step in best_path.steps:
                    node = store.get_node(step.node_id)
                    signals = semantic_score_signals(
                        node,
                        source_node_id=previous_node_id,
                    )
                    best_path_exception_score = max(
                        best_path_exception_score,
                        signals.exception_score,
                    )
                    best_path_contradiction_score = max(
                        best_path_contradiction_score,
                        signals.contradiction_score,
                    )
                    previous_node_id = step.node_id
            contradiction_hit = evaluate_contradiction_hit(
                case.expectation,
                surfaced_contradictions=surfaced_contradiction_pairs,
            )
            hit = (
                evidence_hit
                and (path_hit if path_hit is not None else True)
                and (activation_trace_hit if activation_trace_hit is not None else True)
                and (semantic_hit if semantic_hit is not None else True)
                and (contradiction_hit if contradiction_hit is not None else True)
            )

            case_reports.append(
                StructuredBenchmarkCaseReport(
                    case_id=case.case_id,
                    query=case.query,
                    tags=case.tags,
                    hit=hit,
                    path_hit=path_hit,
                    activation_trace_hit=activation_trace_hit,
                    semantic_hit=semantic_hit,
                    contradiction_hit=contradiction_hit,
                    expected_evidence=case.expectation.evidence_node_ids,
                    matched_evidence=matched_evidence,
                    missing_evidence=[
                        node_id
                        for node_id in case.expectation.evidence_node_ids
                        if node_id not in returned_node_ids
                    ],
                    returned_node_ids=returned_node_ids,
                    surfaced_semantic_roles=surfaced_semantic_roles,
                    surfaced_contradictions=surfaced_contradiction_pairs,
                    path_edge_types=path_edge_types,
                    activated_node_count=len(returned_node_ids),
                    activation_trace_length=len(activation_trace),
                    activation_stopped_reasons=activation_stopped_reasons,
                    best_path_hops=max(0, len(best_path.steps) - 1) if best_path else 0,
                    best_path_exception_score=best_path_exception_score,
                    best_path_contradiction_score=best_path_contradiction_score,
                    best_answer=best_path.final_answer if best_path else "",
                    latency_ms=round(latency_ms, 3),
                )
            )

        questions = len(case_reports)
        evidence_recall = (
            sum(1 for report in case_reports if report.hit) / questions
            if questions
            else 0.0
        )
        avg_latency_ms = (
            round(sum(report.latency_ms for report in case_reports) / questions, 3)
            if questions
            else 0.0
        )

        return StructuredBenchmarkReport(
            dataset_id=dataset.dataset_id,
            retriever_name=retriever_name,
            questions=questions,
            evidence_recall=evidence_recall,
            avg_latency_ms=avg_latency_ms,
            case_reports=case_reports,
        )
