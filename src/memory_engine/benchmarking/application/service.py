from __future__ import annotations

from dataclasses import dataclass, field
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
from memory_engine.memory.application.bridge import palace_to_store, store_to_palace
from memory_engine.memory.application.query_models import RecallPolicy, RecallQuery
from memory_engine.memory.application.retrieve_memory_service import RetrieveMemoryService
from memory_engine.retrieval_factory import build_legacy_retriever
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


@dataclass(slots=True)
class PalaceRecallBenchmarkAdapter:
    """Adapt palace-native recall to the benchmark runner's retriever contract."""

    palace: object
    retriever_mode: str
    service: RetrieveMemoryService
    max_hops: int = 3
    max_spaces: int = 3
    max_seeds: int = 5
    allow_legacy_fallback: bool = False
    store: MemoryStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = palace_to_store(self.palace)

    def search(self, query: str, top_k: int = 3):
        palace_result = self.service.recall(
            self.palace,
            RecallQuery(
                palace_id=self.palace.palace_id,
                text=query,
                policy=RecallPolicy(
                    retriever_mode=self.retriever_mode,
                    top_k=top_k,
                    max_hops=self.max_hops,
                    max_spaces=self.max_spaces,
                    max_seeds=self.max_seeds,
                    allow_legacy_fallback=self.allow_legacy_fallback,
                ),
            ),
        )
        return palace_result.to_legacy_retrieval_result()


def build_store_for_dataset(dataset: StructuredBenchmarkDataset, dataset_root: Path) -> MemoryStore:
    store = MemoryStore()
    documents_dir = dataset_root / dataset.document_directory
    domain_pack = get_domain_pack(dataset.domain_pack_name)
    for path in sorted(documents_dir.glob("*.md")):
        ingest_document(path, store, domain_pack=domain_pack)
    return store


def build_palace_for_dataset(dataset: StructuredBenchmarkDataset, dataset_root: Path):
    return store_to_palace(
        build_store_for_dataset(dataset, dataset_root),
        palace_id=dataset.dataset_id,
    )


def build_retriever(retriever_mode: str, store: MemoryStore):
    """Construct a legacy :class:`MemoryStore` retriever (v0 graph stack).

    v1 palace recall uses :class:`memory_engine.memory.application.retrieve_memory_service.RetrieveMemoryService`
    with :func:`memory_engine.memory.application.bridge.palace_to_store` internally; this factory stays the
    stable entry for structured benchmarks and scripts that operate on a plain store.
    """
    return build_legacy_retriever(retriever_mode, store)


def build_palace_retriever(
    retriever_mode: str,
    palace,
    *,
    service: RetrieveMemoryService | None = None,
    max_hops: int = 3,
    max_spaces: int = 3,
    max_seeds: int = 5,
    allow_legacy_fallback: bool = False,
):
    return PalaceRecallBenchmarkAdapter(
        palace=palace,
        retriever_mode=retriever_mode,
        service=service or RetrieveMemoryService(),
        max_hops=max_hops,
        max_spaces=max_spaces,
        max_seeds=max_seeds,
        allow_legacy_fallback=allow_legacy_fallback,
    )


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
                route_hit=case_report.route_hit,
                space_hit=case_report.space_hit,
                activation_trace_hit=case_report.activation_trace_hit,
                activation_snapshot_hit=case_report.activation_snapshot_hit,
                semantic_hit=case_report.semantic_hit,
                contradiction_hit=case_report.contradiction_hit,
                lifecycle_hit=case_report.lifecycle_hit,
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
                path_hit_rate=_optional_hit_rate(mode_report.case_reports, "path_hit")[0],
                path_hit_cases=_optional_hit_rate(mode_report.case_reports, "path_hit")[1],
                route_hit_rate=_optional_hit_rate(mode_report.case_reports, "route_hit")[0],
                route_hit_cases=_optional_hit_rate(mode_report.case_reports, "route_hit")[1],
                space_hit_rate=_optional_hit_rate(mode_report.case_reports, "space_hit")[0],
                space_hit_cases=_optional_hit_rate(mode_report.case_reports, "space_hit")[1],
                activation_trace_hit_rate=_optional_hit_rate(
                    mode_report.case_reports,
                    "activation_trace_hit",
                )[0],
                activation_trace_hit_cases=_optional_hit_rate(
                    mode_report.case_reports,
                    "activation_trace_hit",
                )[1],
                activation_snapshot_hit_rate=_optional_hit_rate(
                    mode_report.case_reports,
                    "activation_snapshot_hit",
                )[0],
                activation_snapshot_hit_cases=_optional_hit_rate(
                    mode_report.case_reports,
                    "activation_snapshot_hit",
                )[1],
                semantic_hit_rate=_optional_hit_rate(mode_report.case_reports, "semantic_hit")[0],
                semantic_hit_cases=_optional_hit_rate(mode_report.case_reports, "semantic_hit")[1],
                contradiction_hit_rate=_optional_hit_rate(
                    mode_report.case_reports,
                    "contradiction_hit",
                )[0],
                contradiction_hit_cases=_optional_hit_rate(
                    mode_report.case_reports,
                    "contradiction_hit",
                )[1],
                lifecycle_hit_rate=_optional_hit_rate(mode_report.case_reports, "lifecycle_hit")[0],
                lifecycle_hit_cases=_optional_hit_rate(mode_report.case_reports, "lifecycle_hit")[1],
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


def _optional_hit_rate(case_reports, field_name: str) -> tuple[float, int]:
    applicable = [
        getattr(report, field_name)
        for report in case_reports
        if getattr(report, field_name) is not None
    ]
    if not applicable:
        return 0.0, 0
    return sum(1 for value in applicable if value) / len(applicable), len(applicable)


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

    def run_palace_from_dataset_path(
        self,
        *,
        dataset_path: Path,
        retriever_mode: str,
        top_k: int = 3,
        max_hops: int = 3,
        max_spaces: int = 3,
        max_seeds: int = 5,
        allow_legacy_fallback: bool = False,
    ):
        dataset = self.dataset_repository.load(dataset_path)
        return self.run_palace(
            dataset=dataset,
            dataset_root=dataset_path.parent,
            retriever_mode=retriever_mode,
            top_k=top_k,
            max_hops=max_hops,
            max_spaces=max_spaces,
            max_seeds=max_seeds,
            allow_legacy_fallback=allow_legacy_fallback,
        )

    def run_palace(
        self,
        *,
        dataset: StructuredBenchmarkDataset,
        dataset_root: Path,
        retriever_mode: str,
        top_k: int = 3,
        max_hops: int = 3,
        max_spaces: int = 3,
        max_seeds: int = 5,
        allow_legacy_fallback: bool = False,
    ):
        palace = build_palace_for_dataset(dataset, dataset_root)
        retriever = build_palace_retriever(
            retriever_mode,
            palace,
            max_hops=max_hops,
            max_spaces=max_spaces,
            max_seeds=max_seeds,
            allow_legacy_fallback=allow_legacy_fallback,
        )
        return self.runner.run(
            dataset=dataset,
            retriever_name=f"palace_{retriever_mode}",
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

    def run_palace_suite_from_dataset_path(
        self,
        *,
        dataset_path: Path,
        retriever_modes: tuple[str, ...] = DEFAULT_RETRIEVER_MODES,
        top_k: int = 3,
        max_hops: int = 3,
        max_spaces: int = 3,
        max_seeds: int = 5,
        allow_legacy_fallback: bool = False,
    ) -> StructuredBenchmarkSuiteReport:
        dataset = self.dataset_repository.load(dataset_path)
        return self.run_palace_suite(
            dataset=dataset,
            dataset_root=dataset_path.parent,
            retriever_modes=retriever_modes,
            top_k=top_k,
            max_hops=max_hops,
            max_spaces=max_spaces,
            max_seeds=max_seeds,
            allow_legacy_fallback=allow_legacy_fallback,
        )

    def run_palace_suite(
        self,
        *,
        dataset: StructuredBenchmarkDataset,
        dataset_root: Path,
        retriever_modes: tuple[str, ...] = DEFAULT_RETRIEVER_MODES,
        top_k: int = 3,
        max_hops: int = 3,
        max_spaces: int = 3,
        max_seeds: int = 5,
        allow_legacy_fallback: bool = False,
    ) -> StructuredBenchmarkSuiteReport:
        palace = build_palace_for_dataset(dataset, dataset_root)
        mode_reports = {
            f"palace_{retriever_mode}": self.runner.run(
                dataset=dataset,
                retriever_name=f"palace_{retriever_mode}",
                retriever=build_palace_retriever(
                    retriever_mode,
                    palace,
                    max_hops=max_hops,
                    max_spaces=max_spaces,
                    max_seeds=max_seeds,
                    allow_legacy_fallback=allow_legacy_fallback,
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
