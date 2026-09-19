from __future__ import annotations

from dataclasses import dataclass

from memory_engine.schema import MemoryPath, RetrievalResult
from memory_engine.store import MemoryStore


@dataclass(frozen=True, slots=True)
class ReasonedAnswer:
    query: str
    answer: str
    cited_node_ids: tuple[str, ...]
    path_edge_types: tuple[str, ...]
    hop_explanations: tuple[str, ...]
    confidence: float
    source_track: str = "legacy_path"


@dataclass(slots=True)
class PathReasoner:
    """
    Deterministic query → memory-path → answer composition.

    This is intentionally not an LLM caller. It makes the path causally useful by
    exposing hop citations and edge-typed glue text for inspection and tests.
    """

    max_hops_in_answer: int = 4

    def reason(
        self,
        *,
        query: str,
        path: MemoryPath,
        store: MemoryStore | None = None,
    ) -> ReasonedAnswer:
        steps = path.steps[: self.max_hops_in_answer]
        if not steps:
            return ReasonedAnswer(
                query=query,
                answer="",
                cited_node_ids=(),
                path_edge_types=(),
                hop_explanations=(),
                confidence=0.0,
            )

        cited = tuple(step.node_id for step in steps)
        edge_types = tuple(
            step.via_edge_type for step in steps[1:] if step.via_edge_type is not None
        )
        hop_explanations: list[str] = []
        answer_parts: list[str] = []
        for index, step in enumerate(steps):
            via = step.via_edge_type or "seed"
            content = _content_for_step(step.node_id, store=store, fallback=path.final_answer)
            explanation = f"hop {index}: {step.node_id} via={via} score={step.score:.3f}"
            hop_explanations.append(explanation)
            if index == 0:
                answer_parts.append(content)
            else:
                glue = _edge_glue(via)
                answer_parts.append(f"{glue} {content}")

        confidence = _confidence_from_scores([step.score for step in steps])
        return ReasonedAnswer(
            query=query,
            answer=" ".join(part.strip() for part in answer_parts if part.strip()),
            cited_node_ids=cited,
            path_edge_types=edge_types,
            hop_explanations=tuple(hop_explanations),
            confidence=confidence,
            source_track="legacy_path",
        )

    def reason_from_retrieval(
        self,
        *,
        query: str,
        result: RetrievalResult,
        store: MemoryStore | None = None,
    ) -> ReasonedAnswer:
        if not result.paths:
            return ReasonedAnswer(
                query=query,
                answer="",
                cited_node_ids=(),
                path_edge_types=(),
                hop_explanations=(),
                confidence=0.0,
            )
        return self.reason(query=query, path=result.best_path(), store=store)


def _content_for_step(
    node_id: str,
    *,
    store: MemoryStore | None,
    fallback: str,
) -> str:
    if store is not None:
        try:
            return store.get_node(node_id).content
        except KeyError:
            pass
    return f"[{node_id}]" if not fallback else fallback.split("|")[0].strip()


def _edge_glue(edge_type: str) -> str:
    mapping = {
        "depends_on": "Given that dependency,",
        "exception_to": "As an exception,",
        "contradicts": "In tension with that,",
        "causes": "Therefore,",
        "next_unit": "Next,",
        "summarizes": "In generalized form,",
        "recalls": "Recalling the episode,",
        "cites": "Supported by,",
    }
    return mapping.get(edge_type, "Then,")


def _confidence_from_scores(scores: list[float]) -> float:
    if not scores:
        return 0.0
    average = sum(scores) / len(scores)
    return round(max(0.0, min(average, 1.0)), 3)
