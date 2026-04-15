from __future__ import annotations

from dataclasses import dataclass

from memory_engine.embeddings import lexical_overlap
from memory_engine.memory.domain.enums import MemoryLinkType
from memory_engine.memory.domain.memory_types import RouteMemory
from memory_engine.memory.domain.palace import MemoryPalace
from memory_engine.memory.domain.retrieval_result import RecallRoute
from memory_engine.memory.domain.route_planning import RoutePlanningInput
from memory_engine.memory.domain.seed_selection import SeedActivation


def _route_memory_steps(route: RouteMemory) -> tuple[str, ...]:
    if route.start_memory_id and route.start_memory_id not in route.ordered_waypoints:
        return (route.start_memory_id,) + route.ordered_waypoints
    if route.ordered_waypoints:
        return route.ordered_waypoints
    if route.start_memory_id:
        return (route.start_memory_id,)
    return ()


def _route_memory_blob(palace: MemoryPalace, step_ids: tuple[str, ...]) -> str:
    parts: list[str] = []
    for mid in step_ids:
        mem = palace.memories.get(mid)
        if mem is not None:
            parts.append(mem.content)
    return "\n".join(parts)


def _query_route_kind(query_text: str) -> str | None:
    lowered = query_text.lower()
    if any(token in lowered for token in ("timeline", "sequence", "before", "after", "first", "next")):
        return "timeline"
    if any(token in lowered for token in ("dependency", "depends", "because", "cause", "causal")):
        return "dependency"
    if any(token in lowered for token in ("exception", "override", "unless", "contradict")):
        return "exception"
    if any(token in lowered for token in ("summary", "general rule", "abstract", "pattern")):
        return "semantic-summary"
    return None


def _route_kind_bonus(route_kind: str, query_kind: str | None) -> float:
    if query_kind is None:
        return 0.0
    return 0.2 if route_kind == query_kind else 0.0


def _preferred_edge_types(route_kind: str | None) -> tuple[str, ...]:
    mapping = {
        "timeline": (MemoryLinkType.NEXT.value,),
        "dependency": (MemoryLinkType.DEPENDS_ON.value, MemoryLinkType.PART_OF.value, MemoryLinkType.CAUSES.value),
        "exception": (MemoryLinkType.EXCEPTION_TO.value, MemoryLinkType.CONTRADICTS.value),
        "semantic-summary": (MemoryLinkType.SUMMARIZES.value, MemoryLinkType.RECALLS.value, MemoryLinkType.PART_OF.value),
    }
    return mapping.get(route_kind or "", ())


@dataclass(slots=True)
class DefaultRoutePlanner:
    """Prefer persisted RouteMemory; otherwise expand along MemoryLink edges (legacy graph path)."""

    def plan_routes(
        self,
        palace: MemoryPalace,
        planning: RoutePlanningInput,
        seeds: tuple[SeedActivation, ...],
    ) -> tuple[RecallRoute, ...]:
        seed_ids = {s.memory_id for s in seeds}
        query_kind = _query_route_kind(planning.text)
        scored: list[RecallRoute] = []

        for memory in palace.memories.values():
            if not isinstance(memory, RouteMemory):
                continue
            steps = _route_memory_steps(memory)
            if not steps:
                continue
            waypoint_set = set(steps)
            overlap = len(seed_ids & waypoint_set) / max(1, len(waypoint_set))
            blob = _route_memory_blob(palace, steps)
            lex = lexical_overlap(planning.text, blob)
            kind_bonus = _route_kind_bonus(memory.route_kind or "route_memory", query_kind)
            score = 0.5 * overlap + 0.35 * lex + (0.1 if seed_ids & waypoint_set else 0.0) + kind_bonus
            if score <= 0.0:
                continue
            rid = memory.route_id or memory.memory_id
            scored.append(
                RecallRoute(
                    route_id=rid,
                    route_kind=memory.route_kind or "route_memory",
                    step_memory_ids=steps,
                    score=min(1.0, score),
                    route_source="route_memory",
                ),
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        if scored:
            return tuple(scored[: max(1, planning.top_k)])

        if not seeds:
            return ()

        # Legacy graph expansion: walk strongest outbound edge per hop from the top seed.
        start = seeds[0].memory_id
        path: list[str] = [start]
        current = start
        preferred_edge_types = _preferred_edge_types(query_kind)
        for _ in range(max(0, planning.max_hops)):
            outbound = palace.outbound_links(current)
            if not outbound:
                break
            ranked = sorted(
                outbound,
                key=lambda ln: (
                    -(1 if ln.link_type.value in preferred_edge_types else 0),
                    -ln.strength,
                    ln.to_memory_id,
                ),
            )
            best = ranked[0]
            nxt = best.to_memory_id
            if nxt in path:
                break
            path.append(nxt)
            current = nxt

        route_kind = query_kind or "graph_expansion"

        return (
            RecallRoute(
                route_id=f"legacy-path-{start}",
                route_kind=route_kind,
                step_memory_ids=tuple(path),
                score=seeds[0].score,
                route_source="legacy_path",
            ),
        )
