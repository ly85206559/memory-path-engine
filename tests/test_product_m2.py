import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from memory_engine.cli import run as cli_run
from memory_engine.hooks_install import install_hooks
from memory_engine.mcp_server import TOOLS, call_tool, handle_request
from memory_engine.product_service import palace_ingest, palace_ingest_memo, palace_search
from memory_engine.retrieval_factory import build_legacy_retriever
from memory_engine.retrieve import HybridRetriever
from memory_engine.schema import MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


class HybridRetrieverTests(unittest.TestCase):
    def test_hybrid_mode_is_registered(self):
        store = MemoryStore()
        store.add_node(
            MemoryNode(
                id="a",
                type="clause",
                content="rollback does not recover the API worker queue",
                weights=MemoryWeight(importance=0.7),
            )
        )
        retriever = build_legacy_retriever("hybrid", store)
        self.assertIsInstance(retriever._delegate, HybridRetriever)
        result = retriever.search("rollback recover API", top_k=1)
        self.assertTrue(result.paths)


class ProductServiceAndMcpTests(unittest.TestCase):
    def test_memo_search_and_mcp_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            palace = Path(tmp) / ".mpe"
            cli_run(["--palace", str(palace), "init", str(palace), "--mode", "hybrid"])
            runbooks = (
                Path(__file__).resolve().parents[1]
                / "examples"
                / "runbook_pack"
                / "runbooks"
            )
            palace_ingest([runbooks], palace=palace, domain_pack="example_runbook_pack")
            memo = palace_ingest_memo(
                "Prefer restarting the worker before paging the database owner.",
                palace=palace,
                title="decision",
            )
            self.assertIn("memo:", memo["node_id"])
            found = palace_search(
                "restart worker before paging database",
                palace=palace,
                mode="hybrid",
                top_k=2,
            )
            self.assertTrue(found["answer"])
            self.assertTrue(found["hop_explanations"] or found["paths"])

            names = {tool["name"] for tool in TOOLS}
            self.assertIn("mpe_search", names)
            self.assertIn("mpe_reinforce", names)
            status = call_tool("mpe_status", {"palace": str(palace)})
            self.assertFalse(status["isError"])
            init_msg = handle_request(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            )
            self.assertEqual(init_msg["result"]["serverInfo"]["name"], "memory-path-engine")
            listed = handle_request(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            )
            self.assertGreaterEqual(len(listed["result"]["tools"]), 6)


class HooksInstallTests(unittest.TestCase):
    def test_hooks_install_writes_cursor_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            written = install_hooks(root, force=True)
            self.assertTrue((root / ".cursor" / "mpe-hooks" / "mpe_session_start.sh").exists())
            self.assertTrue((root / ".cursor" / "mpe-hooks" / "mcp.local.json").exists())
            self.assertIn("mpe_stop_save.sh", written)


class CliM2Tests(unittest.TestCase):
    def test_memo_and_hooks_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            palace = Path(tmp) / "palace"
            project = Path(tmp) / "proj"
            project.mkdir()
            self.assertEqual(cli_run(["--palace", str(palace), "init", str(palace)]), 0)
            with patch("sys.stdin", StringIO("session note about graph paths")):
                self.assertEqual(
                    cli_run(
                        [
                            "--palace",
                            str(palace),
                            "memo",
                            "--title",
                            "note",
                            "--source",
                            "test",
                        ]
                    ),
                    0,
                )
            self.assertEqual(
                cli_run(["hooks", "install", "--project", str(project), "--force"]),
                0,
            )
            self.assertTrue((project / ".cursor" / "mpe-hooks" / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
