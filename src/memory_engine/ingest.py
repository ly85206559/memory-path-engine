from __future__ import annotations

from pathlib import Path
import re

from memory_engine.embeddings import tokenize
from memory_engine.schema import EvidenceRef, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


SECTION_PATTERN = re.compile(r"^##\s+(?P<title>.+)$")
CLAUSE_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\s+(?P<body>.+)$")


def ingest_contract_markdown(path: Path, store: MemoryStore, domain_pack: str = "contract_pack") -> None:
    current_section = "root"
    previous_node_id: str | None = None
    lines = path.read_text(encoding="utf-8").splitlines()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        section_match = SECTION_PATTERN.match(line)
        if section_match:
            current_section = normalize_id(section_match.group("title"))
            previous_node_id = None
            continue

        clause_match = CLAUSE_PATTERN.match(line)
        if not clause_match:
            continue

        clause_number = clause_match.group("number")
        clause_body = clause_match.group("body")
        node_id = f"{path.stem}:{clause_number}"
        weight = infer_weight(clause_body)
        node = MemoryNode(
            id=node_id,
            type="clause",
            content=clause_body,
            attributes={
                "domain_pack": domain_pack,
                "document_id": path.stem,
                "section": current_section,
                "clause_number": clause_number,
            },
            weights=weight,
            source_ref=EvidenceRef(
                source_path=str(path),
                section_id=clause_number,
                line_start=line_number,
                line_end=line_number,
            ),
        )
        store.add_node(node)

        if previous_node_id:
            store.add_edge(
                MemoryEdge(
                    from_id=previous_node_id,
                    to_id=node_id,
                    edge_type="next_clause",
                    weight=0.4,
                    bidirectional=True,
                    source_ref=node.source_ref,
                )
            )

        previous_node_id = node_id
        create_semantic_edges(store, node)


def create_semantic_edges(store: MemoryStore, node: MemoryNode) -> None:
    keywords = {
        "depends_on": ["subject to", "conditioned on", "if"],
        "exception_to": ["except", "unless", "notwithstanding"],
        "causes": ["shall pay", "liable", "terminate", "damages"],
    }

    text = node.content.lower()
    for existing in store.nodes():
        if existing.id == node.id:
            continue

        for edge_type, triggers in keywords.items():
            if any(trigger in text for trigger in triggers) and shared_tokens(existing.content, node.content) >= 2:
                store.add_edge(
                    MemoryEdge(
                        from_id=node.id,
                        to_id=existing.id,
                        edge_type=edge_type,
                        weight=0.7,
                        bidirectional=edge_type == "exception_to",
                        source_ref=node.source_ref,
                    )
                )
                break


def infer_weight(text: str) -> MemoryWeight:
    lowered = text.lower()
    risk = 0.9 if any(word in lowered for word in ["terminate", "damages", "indemnify", "breach", "penalty"]) else 0.3
    importance = 0.8 if any(word in lowered for word in ["shall", "must", "liable", "exclusive"]) else 0.4
    novelty = 0.85 if any(word in lowered for word in ["except", "unless", "notwithstanding"]) else 0.25
    confidence = 0.95
    return MemoryWeight(
        importance=importance,
        risk=risk,
        novelty=novelty,
        confidence=confidence,
    )


def normalize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def shared_tokens(left: str, right: str) -> int:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    return len(left_tokens & right_tokens)
