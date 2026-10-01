from __future__ import annotations

from math import log2

from memory_engine.benchmarking.domain.public_models import (
    PublicBenchmarkCaseResult,
    PublicBenchmarkModeReport,
)
from memory_engine.schema import RetrievalResult


def ranked_node_ids_from_result(result: RetrievalResult, *, top_k: int) -> list[str]:
    """Extract a public-KPI ranking from a retrieval result.

    Prefer score-ordered unique memories (seed preferred on ties). Path-walk
    order alone can bury gold seeds behind low-score expansion neighbors and
    collapse R@5 / NDCG@10 on multi-session haystacks.
    """
    if result.palace_result is not None and result.palace_result.retrieved_memories:
        best: dict[str, tuple[float, int]] = {}
        for item in result.palace_result.retrieved_memories:
            seed_bonus = 1 if item.retrieval_role == "seed" else 0
            previous = best.get(item.memory_id)
            candidate = (item.score, seed_bonus)
            if previous is None or candidate > previous:
                best[item.memory_id] = candidate
        return [
            memory_id
            for memory_id, _score in sorted(
                best.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )[:top_k]
        ]

    best: dict[str, tuple[float, int]] = {}
    for path in result.paths:
        for step_index, step in enumerate(path.steps):
            seed_bonus = 1 if step_index == 0 else 0
            previous = best.get(step.node_id)
            candidate = (step.score, seed_bonus)
            if previous is None or candidate > previous:
                best[step.node_id] = candidate
    return [
        memory_id
        for memory_id, _score in sorted(
            best.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
    ]


def matched_ranks(gold_items: list[str], retrieved_items: list[str]) -> list[int]:
    positions = {node_id: idx + 1 for idx, node_id in enumerate(retrieved_items)}
    return sorted(positions[item] for item in gold_items if item in positions)


def compute_recall_at_k(gold_items: list[str], retrieved_items: list[str], *, k: int) -> float:
    if not gold_items:
        return 0.0
    retrieved_top_k = set(retrieved_items[:k])
    return 1.0 if any(item in retrieved_top_k for item in gold_items) else 0.0


def compute_ndcg_at_k(gold_items: list[str], retrieved_items: list[str], *, k: int) -> float:
    if not gold_items or k <= 0:
        return 0.0

    gold_set = set(gold_items)
    dcg = 0.0
    for rank, node_id in enumerate(retrieved_items[:k], start=1):
        if node_id in gold_set:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def build_public_case_result(
    *,
    case_id: str,
    query: str,
    gold_items: list[str],
    result: RetrievalResult,
    top_k: int,
    latency_ms: float,
    metadata: dict | None = None,
) -> PublicBenchmarkCaseResult:
    retrieved_items = ranked_node_ids_from_result(result, top_k=top_k)
    ranks = matched_ranks(gold_items, retrieved_items)
    return PublicBenchmarkCaseResult(
        case_id=case_id,
        query=query,
        gold_items=gold_items,
        retrieved_items=retrieved_items,
        matched_ranks=ranks,
        hit_at_5=compute_recall_at_k(gold_items, retrieved_items, k=5) > 0.0,
        hit_at_10=compute_recall_at_k(gold_items, retrieved_items, k=10) > 0.0,
        ndcg_at_10=round(compute_ndcg_at_k(gold_items, retrieved_items, k=10), 6),
        latency_ms=round(latency_ms, 3),
        metadata=metadata or {},
    )


def build_public_mode_report(
    *,
    benchmark_name: str,
    dataset_id: str,
    retriever_name: str,
    case_reports: list[PublicBenchmarkCaseResult],
    metadata: dict | None = None,
) -> PublicBenchmarkModeReport:
    questions = len(case_reports)
    recall_at_5 = (
        sum(1 for report in case_reports if report.hit_at_5) / questions if questions else 0.0
    )
    recall_at_10 = (
        sum(1 for report in case_reports if report.hit_at_10) / questions if questions else 0.0
    )
    ndcg_at_10 = (
        round(sum(report.ndcg_at_10 for report in case_reports) / questions, 6) if questions else 0.0
    )
    avg_latency_ms = (
        round(sum(report.latency_ms for report in case_reports) / questions, 3) if questions else 0.0
    )
    return PublicBenchmarkModeReport(
        benchmark_name=benchmark_name,
        dataset_id=dataset_id,
        retriever_name=retriever_name,
        questions=questions,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        ndcg_at_10=ndcg_at_10,
        avg_latency_ms=avg_latency_ms,
        case_reports=case_reports,
        metadata=metadata or {},
    )
