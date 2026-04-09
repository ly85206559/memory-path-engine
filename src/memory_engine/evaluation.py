from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from memory_engine.ingest import ingest_document
from memory_engine.retrieve import (
    BaselineTopKRetriever,
    EmbeddingTopKRetriever,
    StructureAwareRetriever,
    WeightedGraphRetriever,
)
from memory_engine.store import MemoryStore


def load_example_benchmark_store(contracts_dir: Path) -> MemoryStore:
    store = MemoryStore()
    for path in sorted(contracts_dir.glob("*.md")):
        ingest_document(path, store, domain_pack="example_contract_pack")
    return store


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_baseline_evaluation(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    store = load_example_benchmark_store(contracts_dir)
    retriever = BaselineTopKRetriever(store)
    return evaluate_retriever(
        retriever,
        load_questions(questions_path),
        top_k=top_k,
        detailed=detailed,
    )


def run_embedding_evaluation(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    store = load_example_benchmark_store(contracts_dir)
    retriever = EmbeddingTopKRetriever(store)
    return evaluate_retriever(
        retriever,
        load_questions(questions_path),
        top_k=top_k,
        detailed=detailed,
    )


def run_structure_only_evaluation(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    store = load_example_benchmark_store(contracts_dir)
    retriever = StructureAwareRetriever(store)
    return evaluate_retriever(
        retriever,
        load_questions(questions_path),
        top_k=top_k,
        detailed=detailed,
    )


def run_weighted_evaluation(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    store = load_example_benchmark_store(contracts_dir)
    retriever = WeightedGraphRetriever(store)
    return evaluate_retriever(
        retriever,
        load_questions(questions_path),
        top_k=top_k,
        detailed=detailed,
    )


def run_evaluation_suite(
    contracts_dir: Path,
    questions_path: Path,
    top_k: int = 3,
    detailed: bool = False,
) -> dict:
    mode_results = {
        "lexical_baseline": run_baseline_evaluation(
            contracts_dir,
            questions_path,
            top_k=top_k,
            detailed=detailed,
        ),
        "embedding_baseline": run_embedding_evaluation(
            contracts_dir,
            questions_path,
            top_k=top_k,
            detailed=detailed,
        ),
        "structure_only": run_structure_only_evaluation(
            contracts_dir,
            questions_path,
            top_k=top_k,
            detailed=detailed,
        ),
        "weighted_graph": run_weighted_evaluation(
            contracts_dir,
            questions_path,
            top_k=top_k,
            detailed=detailed,
        ),
    }
    if not detailed:
        return mode_results
    return {
        "modes": mode_results,
        "comparison": build_comparison_report(mode_results),
    }


def evaluate_retriever(retriever, questions: list[dict], top_k: int = 3, detailed: bool = False) -> dict:
    total = len(questions)
    evidence_hits = 0
    total_latency_ms = 0.0
    details = []

    for question in questions:
        started = perf_counter()
        result = retriever.search(question["query"], top_k=top_k)
        latency_ms = (perf_counter() - started) * 1000
        total_latency_ms += latency_ms
        returned_node_ids = {
            step.node_id
            for path in result.paths
            for step in path.steps
        }
        matched_evidence = [
            node_id for node_id in question["evidence_node_ids"] if node_id in returned_node_ids
        ]
        hit = bool(matched_evidence)
        if hit:
            evidence_hits += 1

        if detailed:
            details.append(
                {
                    "question_id": question["id"],
                    "query": question["query"],
                    "tags": question.get("tags", []),
                    "hit": hit,
                    "expected_evidence": question["evidence_node_ids"],
                    "matched_evidence": matched_evidence,
                    "missing_evidence": [
                        node_id
                        for node_id in question["evidence_node_ids"]
                        if node_id not in returned_node_ids
                    ],
                    "returned_node_ids": sorted(returned_node_ids),
                    "best_answer": result.best_path().final_answer if result.paths else "",
                    "latency_ms": round(latency_ms, 3),
                }
            )

    summary = {
        "questions": total,
        "evidence_recall": evidence_hits / total if total else 0.0,
        "avg_latency_ms": round(total_latency_ms / total, 3) if total else 0.0,
    }
    if detailed:
        summary["details"] = details
    return summary


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
                "hit": detail["hit"],
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
                "evidence_recall": mode_result["evidence_recall"],
                "avg_latency_ms": mode_result["avg_latency_ms"],
            }
            for mode_name, mode_result in mode_results.items()
        },
    }
