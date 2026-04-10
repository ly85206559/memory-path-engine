from __future__ import annotations

from pathlib import Path

from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkComparisonCaseReport,
    StructuredBenchmarkComparisonReport,
    StructuredBenchmarkDataset,
    StructuredBenchmarkModeCaseResult,
    StructuredBenchmarkModeSummary,
    StructuredBenchmarkReport,
    StructuredBenchmarkSuiteReport,
)
from memory_engine.benchmarking.infrastructure.json_repository import (
    JsonStructuredBenchmarkDatasetRepository,
)
from memory_engine.domain_pack import get_domain_pack
from memory_engine.ingest import ingest_document
from memory_engine.memory_state import MemoryStatePolicy, StaticMemoryStatePolicy
from memory_engine.retrieve import (
    ActivationSpreadingRetriever,
    BaselineTopKRetriever,
    EmbeddingTopKRetriever,
    StructureAwareRetriever,
    WeightedGraphRetriever,
)
from memory_engine.store import MemoryStore

DEFAULT_RETRIEVER_MODES = (
    "lexical_baseline",
    "embedding_baseline",
    "structure_only",
    "weighted_graph",
    "activation_spreading_v1",
    "weighted_graph_static",
    "weighted_graph_dynamic",
    "activation_spreading_static",
    "activation_spreading_dynamic",
)


def build_store_for_dataset(dataset: StructuredBenchmarkDataset, dataset_root: Path) -> MemoryStore:
    store = MemoryStore()
    documents_dir = dataset_root / dataset.document_directory
    domain_pack = get_domain_pack(dataset.domain_pack_name)
    for path in sorted(documents_dir.glob("*.md")):
        ingest_document(path, store, domain_pack=domain_pack)
    return store


def build_retriever(retriever_mode: str, store: MemoryStore):
    memory_policies: dict[str, MemoryStatePolicy] = {
        "lexical_baseline": StaticMemoryStatePolicy(),
        "embedding_baseline": StaticMemoryStatePolicy(),
        "structure_only": StaticMemoryStatePolicy(),
        "weighted_graph": MemoryStatePolicy(),
        "activation_spreading_v1": MemoryStatePolicy(),
        "weighted_graph_static": StaticMemoryStatePolicy(),
        "weighted_graph_dynamic": MemoryStatePolicy(),
        "activation_spreading_static": StaticMemoryStatePolicy(),
        "activation_spreading_dynamic": MemoryStatePolicy(),
    }
    retriever_builders = {
        "lexical_baseline": BaselineTopKRetriever,
        "embedding_baseline": EmbeddingTopKRetriever,
        "structure_only": StructureAwareRetriever,
        "weighted_graph": WeightedGraphRetriever,
        "activation_spreading_v1": ActivationSpreadingRetriever,
        "weighted_graph_static": WeightedGraphRetriever,
        "weighted_graph_dynamic": WeightedGraphRetriever,
        "activation_spreading_static": ActivationSpreadingRetriever,
        "activation_spreading_dynamic": ActivationSpreadingRetriever,
    }
    try:
        retriever_builder = retriever_builders[retriever_mode]
        memory_state_policy = memory_policies[retriever_mode]
    except KeyError as exc:
        available = ", ".join(sorted(retriever_builders))
        raise ValueError(
            f"Unknown retriever mode '{retriever_mode}'. Available: {available}"
        ) from exc
    return retriever_builder(store, memory_state_policy=memory_state_policy)


def build_comparison_report(
    mode_reports: dict[str, StructuredBenchmarkReport],
) -> StructuredBenchmarkComparisonReport:
    first_mode = next(iter(mode_reports.values()))
    case_ids = [case_report.case_id for case_report in first_mode.case_reports]

    per_question: list[StructuredBenchmarkComparisonCaseReport] = []
    for case_id in case_ids:
        per_mode = {}
        for mode_name, mode_report in mode_reports.items():
            case_report = next(
                item for item in mode_report.case_reports if item.case_id == case_id
            )
            per_mode[mode_name] = StructuredBenchmarkModeCaseResult(
                evidence_hit=case_report.evidence_hit,
                hit=case_report.hit,
                path_hit=case_report.path_hit,
                activation_trace_hit=case_report.activation_trace_hit,
                semantic_hit=case_report.semantic_hit,
                contradiction_hit=case_report.contradiction_hit,
                matched_evidence=case_report.matched_evidence,
                latency_ms=case_report.latency_ms,
            )
        per_question.append(
            StructuredBenchmarkComparisonCaseReport(
                case_id=case_id,
                modes=per_mode,
                best_modes=[
                    mode_name for mode_name, info in per_mode.items() if info.hit
                ],
                missed_by_all=not any(info.hit for info in per_mode.values()),
            )
        )

    return StructuredBenchmarkComparisonReport(
        per_question=per_question,
        mode_summary={
            mode_name: StructuredBenchmarkModeSummary(
                evidence_hit_rate=mode_report.evidence_hit_rate,
                evidence_recall=mode_report.evidence_recall,
                avg_latency_ms=mode_report.avg_latency_ms,
                path_hit_rate=(
                    sum(1 for report in mode_report.case_reports if report.path_hit) / mode_report.questions
                    if mode_report.questions
                    else 0.0
                ),
                activation_trace_hit_rate=(
                    sum(1 for report in mode_report.case_reports if report.activation_trace_hit)
                    / mode_report.questions
                    if mode_report.questions
                    else 0.0
                ),
                semantic_hit_rate=(
                    sum(1 for report in mode_report.case_reports if report.semantic_hit) / mode_report.questions
                    if mode_report.questions
                    else 0.0
                ),
                contradiction_hit_rate=(
                    sum(1 for report in mode_report.case_reports if report.contradiction_hit)
                    / mode_report.questions
                    if mode_report.questions
                    else 0.0
                ),
                avg_activated_nodes=(
                    round(
                        sum(report.activated_node_count for report in mode_report.case_reports)
                        / mode_report.questions,
                        3,
                    )
                    if mode_report.questions
                    else 0.0
                ),
                avg_propagation_depth=(
                    round(
                        sum(report.best_path_hops for report in mode_report.case_reports)
                        / mode_report.questions,
                        3,
                    )
                    if mode_report.questions
                    else 0.0
                ),
                avg_activation_trace_length=(
                    round(
                        sum(report.activation_trace_length for report in mode_report.case_reports)
                        / mode_report.questions,
                        3,
                    )
                    if mode_report.questions
                    else 0.0
                ),
            )
            for mode_name, mode_report in mode_reports.items()
        },
    )


class StructuredBenchmarkEvaluationService:
    def __init__(
        self,
        dataset_repository: JsonStructuredBenchmarkDatasetRepository | None = None,
        runner: StructuredBenchmarkRunner | None = None,
    ) -> None:
        self.dataset_repository = dataset_repository or JsonStructuredBenchmarkDatasetRepository()
        self.runner = runner or StructuredBenchmarkRunner()

    def run_from_dataset_path(
        self,
        *,
        dataset_path: Path,
        retriever_mode: str,
        top_k: int = 3,
    ):
        dataset = self.dataset_repository.load(dataset_path)
        return self.run(
            dataset=dataset,
            dataset_root=dataset_path.parent,
            retriever_mode=retriever_mode,
            top_k=top_k,
        )

    def run(
        self,
        *,
        dataset: StructuredBenchmarkDataset,
        dataset_root: Path,
        retriever_mode: str,
        top_k: int = 3,
    ):
        store = build_store_for_dataset(dataset, dataset_root)
        retriever = build_retriever(retriever_mode, store)
        return self.runner.run(
            dataset=dataset,
            retriever_name=retriever_mode,
            retriever=retriever,
            top_k=top_k,
        )

    def run_suite_from_dataset_path(
        self,
        *,
        dataset_path: Path,
        retriever_modes: tuple[str, ...] = DEFAULT_RETRIEVER_MODES,
        top_k: int = 3,
    ) -> StructuredBenchmarkSuiteReport:
        dataset = self.dataset_repository.load(dataset_path)
        return self.run_suite(
            dataset=dataset,
            dataset_root=dataset_path.parent,
            retriever_modes=retriever_modes,
            top_k=top_k,
        )

    def run_suite(
        self,
        *,
        dataset: StructuredBenchmarkDataset,
        dataset_root: Path,
        retriever_modes: tuple[str, ...] = DEFAULT_RETRIEVER_MODES,
        top_k: int = 3,
    ) -> StructuredBenchmarkSuiteReport:
        mode_reports = {
            retriever_mode: self.runner.run(
                dataset=dataset,
                retriever_name=retriever_mode,
                retriever=build_retriever(
                    retriever_mode,
                    build_store_for_dataset(dataset, dataset_root),
                ),
                top_k=top_k,
            )
            for retriever_mode in retriever_modes
        }
        return StructuredBenchmarkSuiteReport(
            dataset_id=dataset.dataset_id,
            modes=mode_reports,
            comparison=build_comparison_report(mode_reports),
        )
