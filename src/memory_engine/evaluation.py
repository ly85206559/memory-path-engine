from __future__ import annotations

import os
import warnings
from pathlib import Path

from memory_engine.benchmarking.application.service import (
    DEFAULT_RETRIEVER_MODES,
    StructuredBenchmarkEvaluationService,
)
from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkComparisonReport,
    StructuredBenchmarkReport,
    StructuredBenchmarkSuiteReport,
)
from memory_engine.benchmarking.infrastructure.legacy_questions import (
    load_legacy_questions_dataset,
)

_LEGACY_RETRIEVER_MODES = {
    "baseline": "lexical_baseline",
    "embedding": "embedding_baseline",
    "structure_only": "structure_only",
    "weighted": "weighted_graph",
}


def _legacy_dataset_for_questions(contracts_dir: Path, questions_path: Path):
    document_directory = os.path.relpath(contracts_dir, questions_path.parent)
    return load_legacy_questions_dataset(
        questions_path=questions_path,
        dataset_id=f"legacy-{questions_path.stem}",
        dataset_name=f"Legacy questions from {questions_path.name}",
        domain_pack_name="example_contract_pack",
        document_directory=document_directory,
    )


def _run_legacy_mode(
    *,
    contracts_dir: Path,
    questions_path: Path,
    retriever_mode: str,
    top_k: int,
) -> StructuredBenchmarkReport:
    dataset = _legacy_dataset_for_questions(contracts_dir, questions_path)
    service = StructuredBenchmarkEvaluationService()
    return service.run(
        dataset=dataset,
        dataset_root=questions_path.parent,
        retriever_mode=retriever_mode,
        top_k=top_k,
    )


def _legacy_summary(report: StructuredBenchmarkReport, *, detailed: bool) -> dict:
    summary = {
        "questions": report.questions,
        "evidence_hit_rate": report.evidence_hit_rate,
        "evidence_recall": report.evidence_recall,
        "avg_latency_ms": report.avg_latency_ms,
    }
    if detailed:
        summary["details"] = [
            {
                "question_id": case_report.case_id,
                "query": case_report.query,
                "tags": case_report.tags,
                "evidence_hit": case_report.evidence_hit,
                "hit": case_report.hit,
                "path_hit": case_report.path_hit,
                "activation_trace_hit": case_report.activation_trace_hit,
                "semantic_hit": case_report.semantic_hit,
                "contradiction_hit": case_report.contradiction_hit,
                "expected_evidence": case_report.expected_evidence,
                "matched_evidence": case_report.matched_evidence,
                "missing_evidence": case_report.missing_evidence,
                "returned_node_ids": case_report.returned_node_ids,
                "surfaced_semantic_roles": case_report.surfaced_semantic_roles,
                "surfaced_contradictions": case_report.surfaced_contradictions,
                "path_edge_types": case_report.path_edge_types,
                "activated_node_count": case_report.activated_node_count,
                "activation_trace_length": case_report.activation_trace_length,
                "activation_stopped_reasons": case_report.activation_stopped_reasons,
                "best_path_hops": case_report.best_path_hops,
                "best_path_exception_score": case_report.best_path_exception_score,
                "best_path_contradiction_score": case_report.best_path_contradiction_score,
                "best_answer": case_report.best_answer,
                "latency_ms": case_report.latency_ms,
            }
            for case_report in report.case_reports
        ]
    return summary


def _legacy_comparison_report(comparison: StructuredBenchmarkComparisonReport) -> dict:
    return {
        "per_question": [
            {
                "question_id": case_report.case_id,
                "modes": {
                    mode_name: {
                        "evidence_hit": mode_result.evidence_hit,
                        "hit": mode_result.hit,
                        "path_hit": mode_result.path_hit,
                        "activation_trace_hit": mode_result.activation_trace_hit,
                        "semantic_hit": mode_result.semantic_hit,
                        "contradiction_hit": mode_result.contradiction_hit,
                        "matched_evidence": mode_result.matched_evidence,
                        "latency_ms": mode_result.latency_ms,
                    }
                    for mode_name, mode_result in case_report.modes.items()
                },
                "best_modes": case_report.best_modes,
                "missed_by_all": case_report.missed_by_all,
            }
            for case_report in comparison.per_question
        ],
        "mode_summary": {
            mode_name: {
                "evidence_hit_rate": mode_summary.evidence_hit_rate,
                "evidence_recall": mode_summary.evidence_recall,
                "avg_latency_ms": mode_summary.avg_latency_ms,
                "path_hit_rate": mode_summary.path_hit_rate,
                "activation_trace_hit_rate": mode_summary.activation_trace_hit_rate,
                "semantic_hit_rate": mode_summary.semantic_hit_rate,
                "contradiction_hit_rate": mode_summary.contradiction_hit_rate,
                "avg_activated_nodes": mode_summary.avg_activated_nodes,
                "avg_propagation_depth": mode_summary.avg_propagation_depth,
                "avg_activation_trace_length": mode_summary.avg_activation_trace_length,
            }
            for mode_name, mode_summary in comparison.mode_summary.items()
        },
    }


def _legacy_suite_summary(suite_report: StructuredBenchmarkSuiteReport, *, detailed: bool) -> dict:
    mode_results = {
        mode_name: _legacy_summary(report, detailed=detailed)
        for mode_name, report in suite_report.modes.items()
    }
    if not detailed:
        return mode_results
    return {
        "modes": mode_results,
        "comparison": _legacy_comparison_report(suite_report.comparison),
    }


def run_baseline_evaluation(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    warnings.warn(
        "run_baseline_evaluation is a legacy wrapper; prefer StructuredBenchmarkEvaluationService.",
        DeprecationWarning,
        stacklevel=2,
    )
    report = _run_legacy_mode(
        contracts_dir=contracts_dir,
        questions_path=questions_path,
        retriever_mode=_LEGACY_RETRIEVER_MODES["baseline"],
        top_k=top_k,
    )
    return _legacy_summary(report, detailed=detailed)


def run_embedding_evaluation(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    warnings.warn(
        "run_embedding_evaluation is a legacy wrapper; prefer StructuredBenchmarkEvaluationService.",
        DeprecationWarning,
        stacklevel=2,
    )
    report = _run_legacy_mode(
        contracts_dir=contracts_dir,
        questions_path=questions_path,
        retriever_mode=_LEGACY_RETRIEVER_MODES["embedding"],
        top_k=top_k,
    )
    return _legacy_summary(report, detailed=detailed)


def run_structure_only_evaluation(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    warnings.warn(
        "run_structure_only_evaluation is a legacy wrapper; prefer StructuredBenchmarkEvaluationService.",
        DeprecationWarning,
        stacklevel=2,
    )
    report = _run_legacy_mode(
        contracts_dir=contracts_dir,
        questions_path=questions_path,
        retriever_mode=_LEGACY_RETRIEVER_MODES["structure_only"],
        top_k=top_k,
    )
    return _legacy_summary(report, detailed=detailed)


def run_weighted_evaluation(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    warnings.warn(
        "run_weighted_evaluation is a legacy wrapper; prefer StructuredBenchmarkEvaluationService.",
        DeprecationWarning,
        stacklevel=2,
    )
    report = _run_legacy_mode(
        contracts_dir=contracts_dir,
        questions_path=questions_path,
        retriever_mode=_LEGACY_RETRIEVER_MODES["weighted"],
        top_k=top_k,
    )
    return _legacy_summary(report, detailed=detailed)


def run_evaluation_suite(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    warnings.warn(
        "run_evaluation_suite is a legacy wrapper; prefer StructuredBenchmarkEvaluationService.",
        DeprecationWarning,
        stacklevel=2,
    )
    dataset = _legacy_dataset_for_questions(contracts_dir, questions_path)
    suite_report = StructuredBenchmarkEvaluationService().run_suite(
        dataset=dataset,
        dataset_root=questions_path.parent,
        retriever_modes=DEFAULT_RETRIEVER_MODES,
        top_k=top_k,
    )
    return _legacy_suite_summary(suite_report, detailed=detailed)


def build_comparison_report(mode_results: dict[str, dict]) -> dict:
    first_mode = next(
        mode_result
        for mode_result in mode_results.values()
        if "details" in mode_result
    )
    question_ids = [detail["question_id"] for detail in first_mode["details"]]

    per_question = []
    for question_id in question_ids:
        per_mode = {}
        for mode_name, mode_result in mode_results.items():
            detail = next(item for item in mode_result["details"] if item["question_id"] == question_id)
            per_mode[mode_name] = {
                "evidence_hit": detail.get("evidence_hit"),
                "hit": detail["hit"],
                "path_hit": detail.get("path_hit"),
                "activation_trace_hit": detail.get("activation_trace_hit"),
                "semantic_hit": detail.get("semantic_hit"),
                "contradiction_hit": detail.get("contradiction_hit"),
                "matched_evidence": detail["matched_evidence"],
                "latency_ms": detail["latency_ms"],
            }
        per_question.append(
            {
                "question_id": question_id,
                "modes": per_mode,
                "best_modes": [
                    mode_name
                    for mode_name, info in per_mode.items()
                    if info["hit"]
                ],
                "missed_by_all": not any(info["hit"] for info in per_mode.values()),
            }
        )

    return {
        "per_question": per_question,
        "mode_summary": {
            mode_name: {
                "evidence_hit_rate": mode_result.get("evidence_hit_rate"),
                "evidence_recall": mode_result["evidence_recall"],
                "avg_latency_ms": mode_result["avg_latency_ms"],
            }
            for mode_name, mode_result in mode_results.items()
        },
    }
