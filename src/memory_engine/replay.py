from __future__ import annotations

from memory_engine.schema import ActivationTraceStep, MemoryNode, MemoryPath, PathStep


def path_answer(
    query: str,
    chain: list[tuple[MemoryNode, float, str, str | None]],
    *,
    activation_trace: list[ActivationTraceStep] | None = None,
    path_score: float | None = None,
) -> MemoryPath:
    chain = _normalize_replay_chain(query, chain)
    supporting_evidence = [item[0].source_ref for item in chain if item[0].source_ref is not None]
    steps = [
        PathStep(
            node_id=node.id,
            reason=reason,
            score=score,
            via_edge_type=edge_type,
        )
        for node, score, reason, edge_type in chain
    ]
    final_answer = summarize_answer(query, [item[0] for item in chain])
    final_score = path_score if path_score is not None else max((item[1] for item in chain), default=0.0)
    return MemoryPath(
        query=query,
        steps=steps,
        activation_trace=activation_trace or [],
        supporting_evidence=supporting_evidence,
        final_answer=final_answer,
        final_score=final_score,
    )


def _normalize_replay_chain(
    query: str,
    chain: list[tuple[MemoryNode, float, str, str | None]],
) -> list[tuple[MemoryNode, float, str, str | None]]:
    if len(chain) < 2:
        return chain
    edge_types = [edge_type for *_rest, edge_type in chain[1:]]
    if not edge_types or any(edge_type != "depends_on" for edge_type in edge_types):
        return chain
    if not any(token in query.lower() for token in ("escalate", "escalation", "page", "who should")):
        return chain

    reversed_chain = list(reversed(chain))
    normalized: list[tuple[MemoryNode, float, str, str | None]] = []
    for idx, (node, score, reason, _edge_type) in enumerate(reversed_chain):
        normalized.append((node, score, reason, None if idx == 0 else "depends_on"))
    return normalized


def summarize_answer(query: str, nodes: list[MemoryNode]) -> str:
    if not nodes:
        return f"No evidence path found for query: {query}"
    return " | ".join(node.content for node in nodes[:3])
