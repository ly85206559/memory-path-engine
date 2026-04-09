import unittest

from memory_engine.schema import MemoryEdge, MemoryNode
from memory_engine.semantics import (
    ContradictionCandidate,
    SemanticRole,
    contradiction_candidates,
    infer_semantic_role,
)


class SemanticTests(unittest.TestCase):
    def test_infer_semantic_role_identifies_exception(self):
        role = infer_semantic_role(
            "Unless goods are defective, Buyer must pay all invoices within 30 days.",
            node_type="clause",
        )
        self.assertEqual(role, SemanticRole.EXCEPTION)

    def test_exception_edges_produce_contradiction_candidates(self):
        nodes = [
            MemoryNode(
                id="clause:1",
                type="clause",
                content="Buyer must pay all invoices within 30 days.",
                attributes={"semantic_role": SemanticRole.OBLIGATION.value},
            ),
            MemoryNode(
                id="clause:2",
                type="clause",
                content="Unless goods are defective, Buyer must pay all invoices within 30 days.",
                attributes={"semantic_role": SemanticRole.EXCEPTION.value},
            ),
        ]
        edges = [
            MemoryEdge(
                from_id="clause:2",
                to_id="clause:1",
                edge_type="exception_to",
                weight=0.7,
            )
        ]

        candidates = contradiction_candidates(nodes, edges)

        self.assertEqual(len(candidates), 1)
        self.assertIsInstance(candidates[0], ContradictionCandidate)
