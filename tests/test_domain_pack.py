import tempfile
import unittest
from pathlib import Path

from memory_engine.domain_pack import (
    ExampleContractPack,
    ExampleRunbookPack,
    HotpotQASentencePack,
    get_domain_pack,
)
from memory_engine.ingest import ingest_document
from memory_engine.store import MemoryStore


class DomainPackTests(unittest.TestCase):
    def test_example_contract_pack_is_registered(self):
        domain_pack = get_domain_pack("example_contract_pack")
        self.assertIsInstance(domain_pack, ExampleContractPack)

    def test_contract_pack_alias_is_still_supported(self):
        domain_pack = get_domain_pack("contract_pack")
        self.assertIsInstance(domain_pack, ExampleContractPack)

    def test_example_runbook_pack_is_registered(self):
        domain_pack = get_domain_pack("example_runbook_pack")
        self.assertIsInstance(domain_pack, ExampleRunbookPack)

    def test_hotpotqa_sentence_pack_is_registered(self):
        domain_pack = get_domain_pack("hotpotqa_sentence_pack")
        self.assertIsInstance(domain_pack, HotpotQASentencePack)

    def test_generic_ingest_uses_domain_pack_registry(self):
        contract_text = "\n".join(
            [
                "# Demo",
                "",
                "## Delivery",
                "1 Supplier shall deliver the equipment within 10 days.",
                "2 If Supplier fails to deliver, Buyer may terminate the order.",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo_contract.md"
            path.write_text(contract_text, encoding="utf-8")

            store = MemoryStore()
            ingest_document(path, store, domain_pack="example_contract_pack")

            node_ids = {node.id for node in store.nodes()}
            self.assertIn("demo_contract:1", node_ids)
            self.assertIn("demo_contract:2", node_ids)

    def test_runbook_pack_ingests_step_nodes(self):
        runbook_text = "\n".join(
            [
                "# Demo",
                "",
                "## Detection",
                "1 When latency exceeds the threshold, notify the incident commander.",
                "2 If the issue persists, restart the worker service.",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo_runbook.md"
            path.write_text(runbook_text, encoding="utf-8")

            store = MemoryStore()
            ingest_document(path, store, domain_pack="example_runbook_pack")

            nodes = {node.id: node for node in store.nodes()}
            self.assertIn("demo_runbook:1", nodes)
            self.assertIn("demo_runbook:2", nodes)
            self.assertEqual(nodes["demo_runbook:1"].type, "step")

    def test_contract_pack_adds_semantic_role_and_exception_target(self):
        contract_text = "\n".join(
            [
                "# Demo",
                "",
                "## Payment",
                "1 Buyer must pay all invoices within 30 days.",
                "2 Unless goods are defective, Buyer must pay all invoices within 30 days.",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo_exception_contract.md"
            path.write_text(contract_text, encoding="utf-8")

            store = MemoryStore()
            ingest_document(path, store, domain_pack="example_contract_pack")

            nodes = {node.id: node for node in store.nodes()}
            self.assertEqual(nodes["demo_exception_contract:2"].attributes["semantic_role"], "exception")
            self.assertEqual(
                nodes["demo_exception_contract:2"].attributes["exception_target"],
                "demo_exception_contract:1",
            )
