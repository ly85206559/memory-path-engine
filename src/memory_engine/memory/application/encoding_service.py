from __future__ import annotations

from memory_engine.embeddings import tokenize
from memory_engine.memory.domain.encoding import EncodingProfile, TriggerProfile


_SCENARIO_KEYWORDS: dict[str, tuple[str, ...]] = {
    "payment dispute": ("payment", "invoice", "refund", "defective", "goods", "chargeback", "liable"),
    "rollback failure": ("rollback", "recover", "recovery", "deployment", "restart", "queue"),
    "exception override": ("unless", "except", "override", "exception", "notwithstanding"),
    "incident escalation": ("escalate", "page", "incident", "sev", "notify"),
    "causal recovery": ("because", "cause", "causal", "depends", "dependency", "after"),
}

_TRIGGER_PHRASES: dict[str, tuple[str, ...]] = {
    "payment dispute": ("payment dispute", "defective goods", "invoice dispute"),
    "rollback failure": ("rollback failure", "rollback does not recover", "queue drain"),
    "exception override": ("exception override", "unless clause", "override rule"),
    "incident escalation": ("incident escalation", "page on call", "escalate immediately"),
    "causal recovery": ("service dependency", "depends on", "recovery cause"),
}


def build_encoding_profile(
    text: str,
    *,
    semantic_role: str | None = None,
    existing_scenario_tags: tuple[str, ...] = (),
    existing_symbolic_tags: tuple[str, ...] = (),
) -> EncodingProfile:
    scenario_tags = existing_scenario_tags or infer_scenario_tags(text)
    symbolic_tags = existing_symbolic_tags or infer_symbolic_tags(text, semantic_role=semantic_role)
    trigger_profile = TriggerProfile(
        phrases=infer_trigger_phrases(text, scenario_tags=scenario_tags),
        situations=infer_trigger_situations(scenario_tags=scenario_tags, symbolic_tags=symbolic_tags),
    )
    return EncodingProfile(
        trigger_profile=trigger_profile,
        scenario_tags=scenario_tags,
        symbolic_tags=symbolic_tags,
    )


def infer_scenario_tags(text: str) -> tuple[str, ...]:
    lowered_tokens = set(tokenize(text))
    matched = [
        scenario
        for scenario, keywords in _SCENARIO_KEYWORDS.items()
        if lowered_tokens & {token for keyword in keywords for token in tokenize(keyword)}
    ]
    return tuple(sorted(matched))


def infer_symbolic_tags(text: str, *, semantic_role: str | None = None) -> tuple[str, ...]:
    lowered = text.lower()
    tags: set[str] = set()
    if any(token in lowered for token in ("contradict", "conflict", "dispute")):
        tags.add("conflict")
    if any(token in lowered for token in ("urgent", "immediately", "asap", "sev", "critical")):
        tags.add("urgency")
    if any(token in lowered for token in ("unless", "except", "override", "exception", "notwithstanding")):
        tags.add("exception")
    if any(token in lowered for token in ("escalate", "page", "notify leadership")):
        tags.add("escalation")
    if any(token in lowered for token in ("because", "depends", "dependency", "cause", "after", "once", "if", "when")):
        tags.add("causal")
    if semantic_role in {"exception", "escalation"}:
        tags.add(semantic_role)
    return tuple(sorted(tags))


def infer_trigger_phrases(text: str, *, scenario_tags: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.lower()
    phrases: set[str] = set()
    for scenario in scenario_tags:
        phrases.update(_TRIGGER_PHRASES.get(scenario, (scenario,)))
    for phrase in (
        "rollback failure",
        "payment dispute",
        "exception override",
        "queue drain",
        "worker restart",
        "defective goods",
    ):
        if phrase in lowered:
            phrases.add(phrase)
    return tuple(sorted(phrases))


def infer_trigger_situations(
    *,
    scenario_tags: tuple[str, ...],
    symbolic_tags: tuple[str, ...],
) -> tuple[str, ...]:
    situations: set[str] = set()
    for scenario in scenario_tags:
        situations.add(f"during {scenario}")
        situations.add(f"after {scenario}")
    for tag in symbolic_tags:
        if tag == "urgency":
            situations.add("when urgency is high")
        if tag == "exception":
            situations.add("when an exception applies")
        if tag == "escalation":
            situations.add("when escalation is required")
        if tag == "causal":
            situations.add("when a dependency chain is involved")
    return tuple(sorted(situations))


def trigger_match_score(query_text: str, encoding: EncodingProfile) -> float:
    lowered = query_text.lower()
    query_tokens = set(tokenize(query_text))
    score = 0.0

    for phrase in encoding.trigger_profile.phrases:
        if phrase and phrase in lowered:
            score += 0.3
    for situation in encoding.trigger_profile.situations:
        situation_tokens = set(tokenize(situation))
        if len(query_tokens & situation_tokens) >= 2:
            score += 0.18
    for scenario in encoding.scenario_tags:
        scenario_tokens = set(tokenize(scenario))
        if scenario_tokens and scenario_tokens <= query_tokens:
            score += 0.25
    for tag in encoding.symbolic_tags:
        if tag in lowered:
            score += 0.12

    return min(score, 1.0)
