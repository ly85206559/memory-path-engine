import json
import tempfile
import unittest
from pathlib import Path

from memory_engine.cli import run as cli_run
from memory_engine.palace_workspace import init_palace, open_palace
from memory_engine.persistence import load_store, save_store
from memory_engine.product_benchmarks import run_longmemeval_baseline
from memory_engine.schema import EvidenceRef, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


def _sample_store() -> MemoryStore:
    store = MemoryStore()
    store.add_node(
        MemoryNode(
            id="n1",
            type="clause",
            content="Buyer must pay invoices within 30 days.",
            attributes={"semantic_role": "obligation"},
            weights=MemoryWeight(importance=0.6, confidence=0.9),
            source_ref=EvidenceRef(source_path="demo.md", section_id="1"),
        )
    )
    store.add_node(
        MemoryNode(
            id="n2",
            type="clause",
            content="Unless goods are defective, Buyer must pay invoices within 30 days.",
            attributes={"semantic_role": "exception"},
            weights=MemoryWeight(importance=0.8, novelty=0.7, confidence=0.95),
        )
    )
    store.add_edge(
        MemoryEdge(
            from_id="n2",
            to_id="n1",
            edge_type="exception_to",
            weight=0.8,
            bidirectional=True,
        )
    )
    return store


class SqlitePersistenceTests(unittest.TestCase):
    def test_roundtrip_preserves_nodes_and_edges(self):
        store = _sample_store()
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "store.sqlite"
            save_store(store, db)
            restored = load_store(db)

        self.assertEqual({n.id for n in restored.nodes()}, {"n1", "n2"})
        self.assertEqual(restored.get_node("n1").content, store.get_node("n1").content)
        self.assertEqual(
            restored.get_node("n1").attributes["semantic_role"],
            "obligation",
        )
        edge_keys = {(e.from_id, e.to_id, e.edge_type) for e in restored.edges()}
        self.assertIn(("n2", "n1", "exception_to"), edge_keys)
        self.assertIn(("n1", "n2", "exception_to"), edge_keys)


class PalaceWorkspaceTests(unittest.TestCase):
    def test_init_ingest_search_status_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".mpe"
            workspace = init_palace(root, name="demo", default_domain_pack="example_runbook_pack")
            self.assertTrue(workspace.config_path.exists())
            self.assertTrue(workspace.store_path.exists())

            runbook = (
                Path(__file__).resolve().parents[1]
                / "examples"
                / "runbook_pack"
                / "runbooks"
                / "01_api_incident_runbook.md"
            )
            result = workspace.ingest_paths([runbook])
            self.assertGreater(result["nodes"], 0)
            status = open_palace(root).status()
            self.assertEqual(status["nodes"], result["nodes"])
            self.assertEqual(status["name"], "demo")


class CliProductTests(unittest.TestCase):
    def test_cli_init_ingest_search_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            palace = Path(tmp) / "palace"
            self.assertEqual(cli_run(["--palace", str(palace), "init", str(palace)]), 0)
            runbooks = (
                Path(__file__).resolve().parents[1]
                / "examples"
                / "runbook_pack"
                / "runbooks"
            )
            self.assertEqual(
                cli_run(
                    [
                        "--palace",
                        str(palace),
                        "ingest",
                        str(runbooks),
                        "--pack",
                        "example_runbook_pack",
                    ]
                ),
                0,
            )
            self.assertEqual(
                cli_run(
                    [
                        "--palace",
                        str(palace),
                        "search",
                        "What if rollback does not recover the API?",
                        "--json",
                    ]
                ),
                0,
            )
            self.assertEqual(cli_run(["--palace", str(palace), "status"]), 0)


class LongMemEvalBaselineTests(unittest.TestCase):
    def test_baseline_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "baselines"
            result = run_longmemeval_baseline(
                limit=1,
                modes=("embedding_baseline", "weighted_graph"),
                output_dir=out,
                label="unit",
            )
            json_path = Path(result["json_path"])
            md_path = Path(result["markdown_path"])
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["product_kpi"])
            self.assertIn("embedding_baseline", payload["modes"])
            self.assertIn("R@5", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
