from __future__ import annotations

import json
from pathlib import Path

from memory_engine.ingest import ingest_contract_markdown
from memory_engine.retrieve import BaselineTopKRetriever, WeightedGraphRetriever
from memory_engine.store import MemoryStore


def load_contract_pack_store(contracts_dir: Path) -> MemoryStore:
    store = MemoryStore()
    for path in sorted(contracts_dir.glob("*.md")):
        ingest_contract_markdown(path, store, domain_pack="contract_pack")
    return store


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_baseline_evaluation(contracts_dir: Path, questions_path: Path, top_k: int = 3) -> dict:
    store = load_contract_pack_store(contracts_dir)
    retriever = BaselineTopKRetriever(store)
    return evaluate_retriever(retriever, load_questions(questions_path), top_k=top_k)


def run_weighted_evaluation(contracts_dir: Path, questions_path: Path, top_k: int = 3) -> dict:
    store = load_contract_pack_store(contracts_dir)
    retriever = WeightedGraphRetriever(store)
    return evaluate_retriever(retriever, load_questions(questions_path), top_k=top_k)


def evaluate_retriever(retriever, questions: list[dict], top_k: int = 3) -> dict:
    total = len(questions)
    evidence_hits = 0
    for question in questions:
        result = retriever.search(question["query"], top_k=top_k)
        returned_node_ids = {
            step.node_id
            for path in result.paths
            for step in path.steps
        }
        if any(node_id in returned_node_ids for node_id in question["evidence_node_ids"]):
            evidence_hits += 1
    return {
        "questions": total,
        "evidence_recall": evidence_hits / total if total else 0.0,
    }
