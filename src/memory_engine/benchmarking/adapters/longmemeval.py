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
from memory_engine.memory.application.bridge import palace_to_store
from memory_engine.memory.domain.enums import MemoryLinkType
from memory_engine.memory.domain.memory_types import EpisodicMemory, RouteMemory
from memory_engine.memory.domain.palace import MemoryLink, MemoryPalace, PalaceSpace
from memory_engine.memory.domain.value_objects import PalaceLocation, SalienceProfile
from memory_engine.schema import EvidenceRef
from memory_engine.store import MemoryStore


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


def longmemeval_turn_node_id(sample: dict, session_id: str, turn_index: int) -> str:
    question_id = str(sample.get("question_id", "longmemeval"))
    return f"{question_id}:{normalize_session_id(session_id)}:t{turn_index}"


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


def _iter_nonempty_turns(turns: list[dict]) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for idx, turn in enumerate(turns, start=1):
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        out.append((idx, turn))
    return out


def build_longmemeval_memory_palace(sample: dict, *, granularity: str = "session") -> MemoryPalace:
    if granularity not in {"session", "turn"}:
        raise ValueError(
            f"Unsupported granularity '{granularity}'. Supported: session, turn."
        )
    if granularity == "turn":
        return _build_turn_palace(sample)
    return _build_session_palace(sample)


def _build_session_palace(sample: dict) -> MemoryPalace:
    question_id = str(sample.get("question_id", "longmemeval"))
    palace = MemoryPalace(palace_id=f"longmemeval:{question_id}")
    session_ids = [str(item) for item in sample.get("haystack_session_ids", [])]
    session_dates = sample.get("haystack_dates", [])
    sessions = sample.get("haystack_sessions", [])
    question_type = str(sample.get("question_type", "unknown"))

    waypoint_ids: list[str] = []
    previous_memory_id: str | None = None
    for idx, (session_id, turns) in enumerate(zip(session_ids, sessions)):
        memory_id = longmemeval_session_node_id(sample, session_id)
        session_text = session_turns_to_text(turns)
        space_id = f"{question_id}:space:{normalize_session_id(session_id)}"
        location = PalaceLocation(
            building="longmemeval",
            floor=question_type,
            room=normalize_session_id(session_id),
            locus=str(idx),
        )
        palace.add_space(
            PalaceSpace(
                space_id=space_id,
                name=f"session-{session_id}",
                location=location,
                tags=("longmemeval", question_type),
            )
        )
        palace.add_memory(
            EpisodicMemory(
                memory_id=memory_id,
                palace_id=palace.palace_id,
                location=location,
                content=session_text,
                salience=SalienceProfile(
                    importance=0.45,
                    risk=0.1,
                    novelty=0.2,
                    confidence=0.95,
                    recency=1.0 if idx == len(session_ids) - 1 else 0.4,
                ),
                source=EvidenceRef(
                    source_path=f"longmemeval:{question_id}",
                    section_id=session_id,
                    metadata={
                        "session_index": idx,
                        "question_type": question_type,
                        "granularity": "session",
                    },
                ),
                metadata={
                    "question_id": question_id,
                    "session_id": session_id,
                    "session_index": idx,
                    "question_type": question_type,
                    "question_date": sample.get("question_date"),
                    "session_date": session_dates[idx] if idx < len(session_dates) else None,
                    "has_answer_turn": any(turn.get("has_answer") for turn in turns),
                    "space_id": space_id,
                    "granularity": "session",
                },
                episode_id=session_id,
                timestamp=session_dates[idx] if idx < len(session_dates) else None,
                participants=tuple(
                    sorted({str(turn.get("role", "unknown")).strip() or "unknown" for turn in turns})
                ),
                event_type="session",
            )
        )
        waypoint_ids.append(memory_id)
        if previous_memory_id is not None:
            palace.add_link(
                MemoryLink(
                    from_memory_id=previous_memory_id,
                    to_memory_id=memory_id,
                    link_type=MemoryLinkType.NEXT,
                    strength=0.6,
                    confidence=0.95,
                    bidirectional=True,
                )
            )
        previous_memory_id = memory_id

    _add_timeline_route(palace, sample, waypoint_ids, question_type)
    return palace


def _build_turn_palace(sample: dict) -> MemoryPalace:
    """One episodic node per non-empty turn — closer to product drawer units."""
    question_id = str(sample.get("question_id", "longmemeval"))
    palace = MemoryPalace(palace_id=f"longmemeval:{question_id}:turn")
    session_ids = [str(item) for item in sample.get("haystack_session_ids", [])]
    session_dates = sample.get("haystack_dates", [])
    sessions = sample.get("haystack_sessions", [])
    question_type = str(sample.get("question_type", "unknown"))

    waypoint_ids: list[str] = []
    previous_memory_id: str | None = None
    for session_idx, (session_id, turns) in enumerate(zip(session_ids, sessions)):
        space_id = f"{question_id}:space:{normalize_session_id(session_id)}"
        location = PalaceLocation(
            building="longmemeval",
            floor=question_type,
            room=normalize_session_id(session_id),
            locus=str(session_idx),
        )
        palace.add_space(
            PalaceSpace(
                space_id=space_id,
                name=f"session-{session_id}",
                location=location,
                tags=("longmemeval", question_type, "turn"),
            )
        )
        nonempty = _iter_nonempty_turns(turns)
        for turn_pos, (turn_index, turn) in enumerate(nonempty):
            memory_id = longmemeval_turn_node_id(sample, session_id, turn_index)
            role = str(turn.get("role", "unknown")).strip() or "unknown"
            content = str(turn.get("content", "")).strip()
            has_answer = bool(turn.get("has_answer"))
            turn_location = PalaceLocation(
                building="longmemeval",
                floor=question_type,
                room=normalize_session_id(session_id),
                locus=f"t{turn_index}",
            )
            palace.add_memory(
                EpisodicMemory(
                    memory_id=memory_id,
                    palace_id=palace.palace_id,
                    location=turn_location,
                    content=f"{role}: {content}",
                    salience=SalienceProfile(
                        importance=0.7 if has_answer else 0.35,
                        risk=0.1,
                        novelty=0.35 if has_answer else 0.15,
                        confidence=0.95,
                        recency=1.0
                        if session_idx == len(session_ids) - 1 and turn_pos == len(nonempty) - 1
                        else 0.45,
                    ),
                    source=EvidenceRef(
                        source_path=f"longmemeval:{question_id}",
                        section_id=f"{session_id}:t{turn_index}",
                        metadata={
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "has_answer": has_answer,
                            "granularity": "turn",
                        },
                    ),
                    metadata={
                        "question_id": question_id,
                        "session_id": session_id,
                        "session_index": session_idx,
                        "turn_index": turn_index,
                        "question_type": question_type,
                        "question_date": sample.get("question_date"),
                        "session_date": session_dates[session_idx]
                        if session_idx < len(session_dates)
                        else None,
                        "has_answer_turn": has_answer,
                        "role": role,
                        "space_id": space_id,
                        "granularity": "turn",
                    },
                    episode_id=f"{session_id}:t{turn_index}",
                    timestamp=session_dates[session_idx] if session_idx < len(session_dates) else None,
                    participants=(role,),
                    event_type="turn",
                )
            )
            waypoint_ids.append(memory_id)
            if previous_memory_id is not None:
                palace.add_link(
                    MemoryLink(
                        from_memory_id=previous_memory_id,
                        to_memory_id=memory_id,
                        link_type=MemoryLinkType.NEXT,
                        strength=0.75 if turn_pos > 0 else 0.55,
                        confidence=0.95,
                        bidirectional=True,
                    )
                )
            previous_memory_id = memory_id

    _add_timeline_route(palace, sample, waypoint_ids, question_type)
    return palace


def _add_timeline_route(
    palace: MemoryPalace,
    sample: dict,
    waypoint_ids: list[str],
    question_type: str,
) -> None:
    if not waypoint_ids:
        return
    question_id = str(sample.get("question_id", "longmemeval"))
    route_id = f"{question_id}:route:timeline"
    route_location = PalaceLocation(
        building="longmemeval",
        floor=question_type,
        room="timeline",
        locus="route",
    )
    palace.add_memory(
        RouteMemory(
            memory_id=route_id,
            palace_id=palace.palace_id,
            location=route_location,
            content=" -> ".join(waypoint_ids),
            salience=SalienceProfile(
                importance=0.55,
                risk=0.05,
                novelty=0.15,
                confidence=0.95,
            ),
            source=EvidenceRef(source_path=f"longmemeval:{question_id}", section_id="timeline"),
            metadata={
                "question_id": question_id,
                "route_kind": "timeline",
                "space_id": f"{question_id}:space:timeline",
            },
            route_id=route_id,
            start_memory_id=waypoint_ids[0],
            ordered_waypoints=tuple(waypoint_ids),
            route_kind="timeline",
        )
    )
    for waypoint_id in waypoint_ids:
        palace.add_link(
            MemoryLink(
                from_memory_id=route_id,
                to_memory_id=waypoint_id,
                link_type=MemoryLinkType.ROUTE_TO,
                strength=0.4,
                confidence=0.9,
            )
        )


def build_longmemeval_memory_store(sample: dict, *, granularity: str = "session") -> MemoryStore:
    palace = build_longmemeval_memory_palace(sample, granularity=granularity)
    return palace_to_store(palace)


def longmemeval_gold_node_ids(sample: dict, *, granularity: str = "session") -> list[str]:
    if granularity == "turn":
        return _longmemeval_gold_turn_ids(sample)
    if granularity != "session":
        raise ValueError(f"Unsupported granularity '{granularity}'. Supported: session, turn.")
    session_ids = sample.get("answer_session_ids", [])
    seen: set[str] = set()
    out: list[str] = []
    for session_id in session_ids:
        node_id = longmemeval_session_node_id(sample, str(session_id))
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


def _longmemeval_gold_turn_ids(sample: dict) -> list[str]:
    """Gold turns: ``has_answer`` turns inside ``answer_session_ids`` (fallback: all turns there)."""
    answer_sessions = {str(item) for item in sample.get("answer_session_ids", [])}
    session_ids = [str(item) for item in sample.get("haystack_session_ids", [])]
    sessions = sample.get("haystack_sessions", [])
    gold: list[str] = []
    fallback: list[str] = []
    for session_id, turns in zip(session_ids, sessions):
        if session_id not in answer_sessions:
            continue
        for turn_index, turn in _iter_nonempty_turns(turns):
            node_id = longmemeval_turn_node_id(sample, session_id, turn_index)
            fallback.append(node_id)
            if turn.get("has_answer"):
                gold.append(node_id)
    return gold or fallback


def longmemeval_question_to_benchmark_case(
    sample: dict,
    *,
    granularity: str = "session",
) -> StructuredBenchmarkCase:
    gold_items = longmemeval_gold_node_ids(sample, granularity=granularity)
    question_id = str(sample.get("question_id", "longmemeval-question"))
    return StructuredBenchmarkCase(
        case_id=question_id,
        query=str(sample["question"]),
        tags=[
            "longmemeval",
            "public",
            str(sample.get("question_type", "unknown")),
            granularity,
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
            palace = build_longmemeval_memory_palace(sample, granularity=granularity)
            store = palace_to_store(palace)
            retriever = build_retriever(mode, store)
            started = perf_counter()
            result = retriever.search(sample["question"], top_k=top_k)
            latency_ms = (perf_counter() - started) * 1000
            case_reports.append(
                build_public_case_result(
                    case_id=str(sample.get("question_id", "longmemeval-question")),
                    query=str(sample["question"]),
                    gold_items=longmemeval_gold_node_ids(sample, granularity=granularity),
                    result=result,
                    top_k=top_k,
                    latency_ms=latency_ms,
                    metadata={
                        "question_type": sample.get("question_type", "unknown"),
                        "question_date": sample.get("question_date"),
                        "granularity": granularity,
                        "memory_kind_distribution": {
                            "episodic": sum(
                                1
                                for memory in palace.memories.values()
                                if getattr(memory, "kind", None).value == "episodic"
                            ),
                            "route": sum(
                                1
                                for memory in palace.memories.values()
                                if getattr(memory, "kind", None).value == "route"
                            ),
                        },
                        "space_count": len(palace.spaces),
                        "route_count": sum(
                            1
                            for memory in palace.memories.values()
                            if getattr(memory, "kind", None).value == "route"
                        ),
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
                "v1_memory_architecture": True,
            },
        )

    return PublicBenchmarkSuiteReport(
        benchmark_name="LongMemEval",
        dataset_id=dataset_id,
        modes=mode_reports,
    )
