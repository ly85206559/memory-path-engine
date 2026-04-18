from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from memory_engine.schema import MemoryEdge, MemoryNode


class SemanticRole(str, Enum):
    OBLIGATION = "obligation"
    CONDITION = "condition"
    EXCEPTION = "exception"
    REMEDY = "remedy"
    ESCALATION = "escalation"
    ACTION = "action"


@dataclass(frozen=True, slots=True)
class ExceptionLink:
    source_node_id: str
    target_node_id: str
    edge_type: str = "exception_to"
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class ContradictionCandidate:
    left_node_id: str
    right_node_id: str
    explanation: str


@dataclass(frozen=True, slots=True)
class SemanticScoreSignals:
    exception_score: float = 0.0
    contradiction_score: float = 0.0


def infer_semantic_role(text: str, *, node_type: str) -> SemanticRole:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("unless", "except", "notwithstanding")):
        return SemanticRole.EXCEPTION
    if any(keyword in lowered for keyword in ("terminate", "damages", "withhold", "recover")):
        return SemanticRole.REMEDY
    if any(keyword in lowered for keyword in ("escalate", "page")):
        return SemanticRole.ESCALATION
    if any(keyword in lowered for keyword in ("restart", "roll back", "verify", "notify")):
        return SemanticRole.ACTION
    if any(keyword in lowered for keyword in ("if", "when", "after", "once", "subject to")):
        return SemanticRole.CONDITION
    if node_type in {"clause", "step"} and any(keyword in lowered for keyword in ("shall", "must")):
        return SemanticRole.OBLIGATION
    return SemanticRole.ACTION if node_type == "step" else SemanticRole.OBLIGATION


def semantic_activation_bonus(node: MemoryNode) -> float:
    role_name = node.attributes.get("semantic_role")
    if role_name == SemanticRole.EXCEPTION.value:
        return 0.15
    if role_name in {SemanticRole.REMEDY.value, SemanticRole.ESCALATION.value}:
        return 0.08
    return 0.0


def contradiction_candidates(nodes: list[MemoryNode], edges: list[MemoryEdge]) -> list[ContradictionCandidate]:
    candidates: list[ContradictionCandidate] = []
    node_map = {node.id: node for node in nodes}
    for edge in edges:
        if edge.edge_type != "exception_to":
            continue
        source = node_map.get(edge.from_id)
        target = node_map.get(edge.to_id)
        if source is None or target is None:
            continue
        if source.attributes.get("semantic_role") == SemanticRole.EXCEPTION.value:
            candidates.append(
                ContradictionCandidate(
                    left_node_id=target.id,
                    right_node_id=source.id,
                    explanation="exception link may override the general rule",
                )
            )
    return candidates


def semantic_score_signals(
    node: MemoryNode,
    *,
    source_node_id: str | None = None,
) -> SemanticScoreSignals:
    role_name = node.attributes.get("semantic_role")
    exception_score = 0.0
    if role_name == SemanticRole.EXCEPTION.value:
        exception_score = 1.0
    elif role_name in {SemanticRole.REMEDY.value, SemanticRole.ESCALATION.value}:
        exception_score = 0.45

    contradiction_targets = set(node.attributes.get("contradiction_targets", []))
    contradiction_score = 0.0
    if contradiction_targets:
        contradiction_score = 0.45
    if source_node_id is not None and source_node_id in contradiction_targets:
        contradiction_score = 1.0

    return SemanticScoreSignals(
        exception_score=exception_score,
        contradiction_score=contradiction_score,
    )


def query_role_alignment_score(query: str, node: MemoryNode) -> float:
    lowered = query.lower()
    role_name = node.attributes.get("semantic_role")
    if role_name == SemanticRole.ESCALATION.value and any(
        token in lowered for token in ("escalate", "escalation", "page", "who should")
    ):
        return 1.0
    return 0.0


def contradiction_bonus(
    *,
    node_id: str,
    candidates: list[ContradictionCandidate],
    source_node_id: str | None = None,
) -> float:
    contradiction_targets = set()
    for candidate in candidates:
        pair = {candidate.left_node_id, candidate.right_node_id}
        if node_id not in pair:
            continue
        contradiction_targets.update(pair - {node_id})
    probe_node = MemoryNode(
        id=node_id,
        type="semantic_probe",
        content="",
        attributes={"contradiction_targets": sorted(contradiction_targets)},
    )
    score = semantic_score_signals(
        probe_node,
        source_node_id=source_node_id,
    ).contradiction_score
    return 0.14 if score >= 1.0 else 0.06 if score > 0 else 0.0


def surfaced_contradictions(
    returned_node_ids: list[str] | set[str],
    candidates: list[ContradictionCandidate],
) -> list[tuple[str, str]]:
    returned_node_id_set = set(returned_node_ids)
    surfaced: list[tuple[str, str]] = []
    for candidate in candidates:
        if (
            candidate.left_node_id in returned_node_id_set
            and candidate.right_node_id in returned_node_id_set
        ):
            surfaced.append((candidate.left_node_id, candidate.right_node_id))
    return surfaced
