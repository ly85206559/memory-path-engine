from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol

from memory_engine.embeddings import tokenize
from memory_engine.schema import EvidenceRef, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.semantics import SemanticRole, infer_semantic_role
from memory_engine.store import MemoryStore


SECTION_PATTERN = re.compile(r"^##\s+(?P<title>.+)$")
CLAUSE_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\s+(?P<body>.+)$")


class DomainPack(Protocol):
    name: str

    def ingest_document(self, path: Path, store: MemoryStore) -> None:
        """Ingest one source document into the memory store."""


@dataclass(frozen=True, slots=True)
class EdgeRule:
    edge_type: str
    triggers: tuple[str, ...]
    minimum_shared_tokens: int = 2
    weight: float = 0.7
    bidirectional: bool = False


class RuleBasedSectionedDocumentPack:
    """
    Shared strategy layer for benchmark packs built from sectioned, numbered documents.

    Concrete packs customize:
    - node typing
    - metadata
    - weighting heuristics
    - semantic edge rules
    """

    name = "rule_based_pack"
    node_type = "memory_unit"

    def ingest_document(self, path: Path, store: MemoryStore) -> None:
        current_section = "root"
        previous_node_id: str | None = None
        lines = path.read_text(encoding="utf-8").splitlines()

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            section_match = SECTION_PATTERN.match(line)
            if section_match:
                current_section = self._normalize_id(section_match.group("title"))
                previous_node_id = None
                continue

            clause_match = CLAUSE_PATTERN.match(line)
            if not clause_match:
                continue

            unit_number = clause_match.group("number")
            unit_body = clause_match.group("body")
            node = self._build_node(
                path=path,
                section_id=current_section,
                unit_number=unit_number,
                unit_body=unit_body,
                line_number=line_number,
            )
            store.add_node(node)

            if previous_node_id:
                store.add_edge(
                    MemoryEdge(
                        from_id=previous_node_id,
                        to_id=node.id,
                        edge_type="next_unit",
                        weight=0.4,
                        bidirectional=True,
                        source_ref=node.source_ref,
                    )
                )

            previous_node_id = node.id
            self._create_semantic_edges(store, node)

    def _build_node(
        self,
        *,
        path: Path,
        section_id: str,
        unit_number: str,
        unit_body: str,
        line_number: int,
    ) -> MemoryNode:
        return MemoryNode(
            id=f"{path.stem}:{unit_number}",
            type=self.node_type,
            content=unit_body,
            attributes=self._build_attributes(path, section_id, unit_number, unit_body),
            weights=self._infer_weight(unit_body),
            source_ref=EvidenceRef(
                source_path=str(path),
                section_id=unit_number,
                line_start=line_number,
                line_end=line_number,
            ),
        )

    def _build_attributes(
        self,
        path: Path,
        section_id: str,
        unit_number: str,
        unit_body: str,
    ) -> dict:
        return {
            "domain_pack": self.name,
            "document_id": path.stem,
            "section": section_id,
            "space_id": f"{path.stem}:{section_id}",
            "unit_number": unit_number,
            "semantic_role": infer_semantic_role(unit_body, node_type=self.node_type).value,
        }

    def _create_semantic_edges(self, store: MemoryStore, node: MemoryNode) -> None:
        text = node.content.lower()
        for existing in store.nodes():
            if existing.id == node.id:
                continue

            for rule in self._edge_rules():
                if any(trigger in text for trigger in rule.triggers) and self._shared_tokens(
                    existing.content,
                    node.content,
                ) >= rule.minimum_shared_tokens:
                    store.add_edge(
                        MemoryEdge(
                            from_id=node.id,
                            to_id=existing.id,
                            edge_type=rule.edge_type,
                            weight=rule.weight,
                            bidirectional=rule.bidirectional,
                            source_ref=node.source_ref,
                        )
                    )
                    self._annotate_semantic_link(node, existing, rule)
                    break

    def _annotate_semantic_link(self, node: MemoryNode, existing: MemoryNode, rule: EdgeRule) -> None:
        if rule.edge_type == "exception_to":
            node.attributes["exception_target"] = existing.id
            node.attributes["exception_target_role"] = existing.attributes.get("semantic_role")
        if (
            rule.edge_type == "depends_on"
            and node.attributes.get("semantic_role") == SemanticRole.CONDITION.value
        ):
            node.attributes["depends_on_target"] = existing.id

    def _edge_rules(self) -> tuple[EdgeRule, ...]:
        raise NotImplementedError

    def _infer_weight(self, text: str) -> MemoryWeight:
        raise NotImplementedError

    def _shared_tokens(self, left: str, right: str) -> int:
        return len(set(tokenize(left)) & set(tokenize(right)))

    def _normalize_id(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class ExampleContractPack(RuleBasedSectionedDocumentPack):
    """
    Reference pack used to validate the memory architecture on
    structured, dependency-heavy documents.

    This pack is an example implementation, not the defining scope
    of the overall project.
    """

    name = "example_contract_pack"
    node_type = "clause"

    def _build_attributes(
        self,
        path: Path,
        section_id: str,
        unit_number: str,
        unit_body: str,
    ) -> dict:
        attributes = super()._build_attributes(path, section_id, unit_number, unit_body)
        attributes["clause_number"] = unit_number
        return attributes

    def _edge_rules(self) -> tuple[EdgeRule, ...]:
        return (
            EdgeRule("depends_on", ("subject to", "conditioned on", "if")),
            EdgeRule("exception_to", ("except", "unless", "notwithstanding"), bidirectional=True),
            EdgeRule("causes", ("shall pay", "liable", "terminate", "damages")),
        )

    def _infer_weight(self, text: str) -> MemoryWeight:
        lowered = text.lower()
        risk = (
            0.9
            if any(
                word in lowered
                for word in ["terminate", "damages", "indemnify", "breach", "penalty"]
            )
            else 0.3
        )
        importance = 0.8 if any(word in lowered for word in ["shall", "must", "liable", "exclusive"]) else 0.4
        novelty = 0.85 if any(word in lowered for word in ["except", "unless", "notwithstanding"]) else 0.25
        confidence = 0.95
        return MemoryWeight(
            importance=importance,
            risk=risk,
            novelty=novelty,
            confidence=confidence,
        )


class ExampleRunbookPack(RuleBasedSectionedDocumentPack):
    """
    Reference pack for operational playbooks and incident runbooks.

    It exists to show that the memory architecture is not tied to contract-like
    documents and can also represent process-heavy, action-oriented knowledge.
    """

    name = "example_runbook_pack"
    node_type = "step"

    def _build_attributes(
        self,
        path: Path,
        section_id: str,
        unit_number: str,
        unit_body: str,
    ) -> dict:
        attributes = super()._build_attributes(path, section_id, unit_number, unit_body)
        attributes["step_number"] = unit_number
        attributes["contains_action"] = any(
            keyword in unit_body.lower()
            for keyword in ("notify", "restart", "roll back", "escalate", "verify")
        )
        return attributes

    def _edge_rules(self) -> tuple[EdgeRule, ...]:
        return (
            EdgeRule("depends_on", ("if", "when", "after", "once")),
            EdgeRule("exception_to", ("unless", "except"), bidirectional=True),
            EdgeRule("causes", ("notify", "restart", "roll back", "escalate", "page")),
        )

    def _infer_weight(self, text: str) -> MemoryWeight:
        lowered = text.lower()
        risk = (
            0.9
            if any(
                word in lowered
                for word in ("severity", "incident", "outage", "rollback", "page", "degrade")
            )
            else 0.35
        )
        importance = (
            0.85
            if any(word in lowered for word in ("must", "immediately", "within", "verify", "required"))
            else 0.45
        )
        novelty = 0.8 if any(word in lowered for word in ("unless", "except", "manual approval")) else 0.2
        return MemoryWeight(
            importance=importance,
            risk=risk,
            novelty=novelty,
            confidence=0.95,
        )


_example_contract_pack = ExampleContractPack()
_example_runbook_pack = ExampleRunbookPack()


class HotpotQASentencePack:
    """
    Placeholder pack for HotpotQA adapter datasets.

    Hotpot graphs are built in code (`benchmarking.adapters.hotpotqa`); markdown
    ingest is not used for this benchmark path.
    """

    name = "hotpotqa_sentence_pack"

    def ingest_document(self, path: Path, store: MemoryStore) -> None:
        return


_hotpotqa_sentence_pack = HotpotQASentencePack()


class LongMemEvalSessionPack:
    """
    Placeholder pack for LongMemEval adapter datasets.

    LongMemEval session graphs are built directly in code
    (`benchmarking.adapters.longmemeval`); markdown ingest is not used.
    """

    name = "longmemeval_session_pack"

    def ingest_document(self, path: Path, store: MemoryStore) -> None:
        return


_longmemeval_session_pack = LongMemEvalSessionPack()

_DOMAIN_PACKS: dict[str, DomainPack] = {
    "example_contract_pack": _example_contract_pack,
    "example_runbook_pack": _example_runbook_pack,
    "hotpotqa_sentence_pack": _hotpotqa_sentence_pack,
    "longmemeval_session_pack": _longmemeval_session_pack,
    # Backward-compatible alias for the existing example dataset and helpers.
    "contract_pack": _example_contract_pack,
}


def get_domain_pack(name: str) -> DomainPack:
    try:
        return _DOMAIN_PACKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DOMAIN_PACKS))
        raise ValueError(f"Unknown domain pack '{name}'. Available: {available}") from exc


def register_domain_pack(domain_pack: DomainPack) -> None:
    _DOMAIN_PACKS[domain_pack.name] = domain_pack
