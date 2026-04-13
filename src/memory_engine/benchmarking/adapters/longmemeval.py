from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter

from memory_engine.benchmarking.application.public_benchmarks import (
    build_public_case_result,
    build_public_mode_report,
)
from memory_engine.benchmarking.application.service import build_retriever
from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkCase,
    StructuredBenchmarkExpectation,
)
from memory_engine.benchmarking.domain.public_models import PublicBenchmarkSuiteReport
from memory_engine.schema import EvidenceRef, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore

_NEXT_SESSION_EDGE = "next_session"


def validate_longmemeval_sample(sample: dict) -> None:
    required_fields = {
        "question_id",
        "question",
        "haystack_session_ids",
        "haystack_sessions",
        "answer_session_ids",
    }
    missing = required_fields - set(sample)
    if missing:
        raise ValueError(
            f"LongMemEval sample {sample.get('question_id', '<unknown>')} is missing required fields: {sorted(missing)}"
        )

    session_ids = [str(item) for item in sample.get("haystack_session_ids", [])]
    sessions = sample.get("haystack_sessions", [])
    answer_session_ids = [str(item) for item in sample.get("answer_session_ids", [])]

    if not session_ids:
        raise ValueError(f"LongMemEval sample {sample['question_id']} must include haystack_session_ids")
    if not sessions:
        raise ValueError(f"LongMemEval sample {sample['question_id']} must include haystack_sessions")
    if len(session_ids) != len(sessions):
        raise ValueError(
            f"LongMemEval sample {sample['question_id']} must have matching haystack_session_ids and haystack_sessions lengths"
        )
    unknown_answer_ids = sorted(set(answer_session_ids) - set(session_ids))
    if unknown_answer_ids:
        raise ValueError(
            f"LongMemEval sample {sample['question_id']} references unknown answer_session_ids: {unknown_answer_ids}"
        )


def load_longmemeval_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data)}")
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise ValueError(f"Expected object at index {idx} in {path}, got {type(sample)}")
        validate_longmemeval_sample(sample)
    return data


def normalize_session_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip())
    return normalized.strip("_") or "session"


def longmemeval_session_node_id(sample: dict, session_id: str) -> str:
    question_id = str(sample.get("question_id", "longmemeval"))
    return f"{question_id}:{normalize_session_id(session_id)}"


def session_turns_to_text(turns: list[dict]) -> str:
    lines = []
    for idx, turn in enumerate(turns, start=1):
        role = str(turn.get("role", "unknown")).strip() or "unknown"
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        marker = " [gold_turn]" if turn.get("has_answer") else ""
        lines.append(f"[Turn {idx}] {role}: {content}{marker}")
    return "\n".join(lines)


def build_longmemeval_memory_store(sample: dict, *, granularity: str = "session") -> MemoryStore:
    if granularity != "session":
        raise ValueError(f"Unsupported granularity '{granularity}'. Only 'session' is supported.")

    store = MemoryStore()
    question_id = str(sample.get("question_id", "longmemeval"))
    session_ids = [str(item) for item in sample.get("haystack_session_ids", [])]
    session_dates = sample.get("haystack_dates", [])
    sessions = sample.get("haystack_sessions", [])

    if len(session_ids) != len(sessions):
        raise ValueError(
            "LongMemEval sample must have the same number of haystack_session_ids and haystack_sessions"
        )

    for idx, (session_id, turns) in enumerate(zip(session_ids, sessions)):
        node_id = longmemeval_session_node_id(sample, session_id)
        session_text = session_turns_to_text(turns)
        node = MemoryNode(
            id=node_id,
            type="session",
            content=session_text,
            attributes={
                "question_id": question_id,
                "session_id": session_id,
                "session_index": idx,
                "question_type": sample.get("question_type", "unknown"),
                "question_date": sample.get("question_date"),
                "session_date": session_dates[idx] if idx < len(session_dates) else None,
                "has_answer_turn": any(turn.get("has_answer") for turn in turns),
            },
            weights=MemoryWeight(
                importance=0.45,
                risk=0.1,
                novelty=0.2,
                confidence=0.95,
            ),
            source_ref=EvidenceRef(
                source_path=f"longmemeval:{question_id}",
                section_id=session_id,
                metadata={
                    "session_index": idx,
                    "question_type": sample.get("question_type", "unknown"),
                },
            ),
        )
        store.add_node(node)
        if idx + 1 < len(session_ids):
            store.add_edge(
                MemoryEdge(
                    from_id=node_id,
                    to_id=longmemeval_session_node_id(sample, session_ids[idx + 1]),
                    edge_type=_NEXT_SESSION_EDGE,
                    weight=0.6,
                    confidence=0.95,
                    bidirectional=True,
                )
            )
    return store


def longmemeval_gold_node_ids(sample: dict) -> list[str]:
    session_ids = sample.get("answer_session_ids", [])
    seen: set[str] = set()
    out: list[str] = []
    for session_id in session_ids:
        node_id = longmemeval_session_node_id(sample, str(session_id))
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


def longmemeval_question_to_benchmark_case(sample: dict) -> StructuredBenchmarkCase:
    gold_items = longmemeval_gold_node_ids(sample)
    question_id = str(sample.get("question_id", "longmemeval-question"))
    return StructuredBenchmarkCase(
        case_id=question_id,
        query=str(sample["question"]),
        tags=[
            "longmemeval",
            "public",
            str(sample.get("question_type", "unknown")),
        ],
        expectation=StructuredBenchmarkExpectation(
            evidence_node_ids=gold_items,
            minimum_evidence_matches=1,
        ),
    )


def run_longmemeval_benchmark(
    samples: list[dict],
    *,
    retriever_modes: tuple[str, ...] = ("embedding_baseline", "weighted_graph"),
    top_k: int = 10,
    granularity: str = "session",
    dataset_id: str = "longmemeval-adapter-eval",
) -> PublicBenchmarkSuiteReport:
    mode_reports = {}

    for mode in retriever_modes:
        case_reports = []
        for sample in samples:
            store = build_longmemeval_memory_store(sample, granularity=granularity)
            retriever = build_retriever(mode, store)
            started = perf_counter()
            result = retriever.search(sample["question"], top_k=top_k)
            latency_ms = (perf_counter() - started) * 1000
            case_reports.append(
                build_public_case_result(
                    case_id=str(sample.get("question_id", "longmemeval-question")),
                    query=str(sample["question"]),
                    gold_items=longmemeval_gold_node_ids(sample),
                    result=result,
                    top_k=top_k,
                    latency_ms=latency_ms,
                    metadata={
                        "question_type": sample.get("question_type", "unknown"),
                        "question_date": sample.get("question_date"),
                    },
                )
            )

        mode_reports[mode] = build_public_mode_report(
            benchmark_name="LongMemEval",
            dataset_id=dataset_id,
            retriever_name=mode,
            case_reports=case_reports,
            metadata={
                "granularity": granularity,
                "top_k": top_k,
            },
        )

    return PublicBenchmarkSuiteReport(
        benchmark_name="LongMemEval",
        dataset_id=dataset_id,
        modes=mode_reports,
    )
