from __future__ import annotations

from time import perf_counter

from memory_engine.benchmarking.domain.evaluation_policy import (
    collect_matched_evidence,
    collect_returned_node_ids,
    evaluate_path_hit,
    evaluate_semantic_hit,
)
from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkCaseReport,
    StructuredBenchmarkDataset,
    StructuredBenchmarkReport,
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
            semantic_hit = evaluate_semantic_hit(
                case.expectation,
                surfaced_semantic_roles=surfaced_semantic_roles,
                path_edge_types=path_edge_types,
            )
            hit = (
                evidence_hit
                and (path_hit if path_hit is not None else True)
                and (semantic_hit if semantic_hit is not None else True)
            )

            case_reports.append(
                StructuredBenchmarkCaseReport(
                    case_id=case.case_id,
                    query=case.query,
                    tags=case.tags,
                    hit=hit,
                    path_hit=path_hit,
                    semantic_hit=semantic_hit,
                    expected_evidence=case.expectation.evidence_node_ids,
                    matched_evidence=matched_evidence,
                    missing_evidence=[
                        node_id
                        for node_id in case.expectation.evidence_node_ids
                        if node_id not in returned_node_ids
                    ],
                    returned_node_ids=returned_node_ids,
                    surfaced_semantic_roles=surfaced_semantic_roles,
                    path_edge_types=path_edge_types,
                    activated_node_count=len(returned_node_ids),
                    best_path_hops=max(0, len(best_path.steps) - 1) if best_path else 0,
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
