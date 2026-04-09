import unittest

from memory_engine.schema import EvidenceRef, MemoryNode, MemoryWeight


class SchemaTests(unittest.TestCase):
    def test_memory_weight_bounded_score_is_capped(self):
        weight = MemoryWeight(importance=2.0, risk=2.0, novelty=1.0, confidence=1.0)
        self.assertGreaterEqual(weight.bounded_score(), 0.0)
        self.assertLessEqual(weight.bounded_score(), 1.0)

    def test_memory_node_holds_source_reference(self):
        ref = EvidenceRef(source_path="sample.md", section_id="1.1", line_start=10, line_end=10)
        node = MemoryNode(id="n1", type="clause", content="Sample", source_ref=ref)
        self.assertIs(node.source_ref, ref)
        self.assertEqual(node.source_ref.section_id, "1.1")
