from __future__ import annotations

from memory_engine.schema import MemoryNode, MemoryPath, PathStep


def path_answer(query: str, chain: list[tuple[MemoryNode, float, str, str | None]]) -> MemoryPath:
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
    final_score = max((item[1] for item in chain), default=0.0)
    return MemoryPath(
        query=query,
        steps=steps,
        supporting_evidence=supporting_evidence,
        final_answer=final_answer,
        final_score=final_score,
    )


def summarize_answer(query: str, nodes: list[MemoryNode]) -> str:
    if not nodes:
        return f"No evidence path found for query: {query}"
    return " | ".join(node.content for node in nodes[:3])
