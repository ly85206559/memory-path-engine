import tempfile
import unittest
from pathlib import Path

from memory_engine.domain_pack import ContractDomainPack, get_domain_pack
from memory_engine.ingest import ingest_document
from memory_engine.store import MemoryStore


class DomainPackTests(unittest.TestCase):
    def test_contract_domain_pack_is_registered(self):
        domain_pack = get_domain_pack("contract_pack")
        self.assertIsInstance(domain_pack, ContractDomainPack)

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
            ingest_document(path, store, domain_pack="contract_pack")

            node_ids = {node.id for node in store.nodes()}
            self.assertIn("demo_contract:1", node_ids)
            self.assertIn("demo_contract:2", node_ids)
