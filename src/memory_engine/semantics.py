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
