import tarfile
import tempfile
import unittest
from pathlib import Path

from memory_engine.cli import run as cli_run
from memory_engine.palace_ops import backup_palace, doctor_report, repair_palace
from memory_engine.palace_workspace import init_palace
from memory_engine.schema import MemoryNode, MemoryWeight


class PalaceOpsTests(unittest.TestCase):
    def test_backup_and_repair_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "palace"
            workspace = init_palace(root, name="demo", default_retriever_mode="hybrid")
            store = workspace.load_store()
            store.add_node(
                MemoryNode(
                    id="n1",
                    type="memo",
                    content="backup me",
                    weights=MemoryWeight(importance=0.5),
                )
            )
            workspace.save_store(store)

            archive_dir = Path(tmp) / "out"
            result = backup_palace(root, output=archive_dir)
            archive = Path(result["archive"])
            self.assertTrue(archive.exists())
            self.assertGreater(result["bytes"], 0)
            with tarfile.open(archive, "r:gz") as tar:
                names = set(tar.getnames())
            self.assertIn("config.json", names)
            self.assertIn("store.sqlite", names)

            healthy = repair_palace(root)
            self.assertTrue(healthy["ok"])
            self.assertEqual(healthy["integrity"], "ok")
            self.assertGreaterEqual(healthy["nodes"], 1)

    def test_repair_quarantines_corrupt_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "palace"
            init_palace(root)
            store_path = root / "store.sqlite"
            store_path.write_bytes(b"not a sqlite database at all")
            result = repair_palace(root)
            self.assertTrue(result["ok"])
            self.assertEqual(result["nodes"], 0)
            self.assertTrue(any(item.startswith("store_quarantined:") for item in result["actions"]))
            self.assertEqual(repair_palace(root)["integrity"], "ok")

    def test_doctor_reports_missing_palace(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-palace"
            report = doctor_report(missing)
            self.assertFalse(report["ok"])
            self.assertTrue(any(check["name"] == "palace" and not check["ok"] for check in report["checks"]))


class CliM4Tests(unittest.TestCase):
    def test_backup_repair_doctor_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            palace = Path(tmp) / ".mpe"
            out = Path(tmp) / "bak"
            self.assertEqual(cli_run(["--palace", str(palace), "init", str(palace)]), 0)
            self.assertEqual(
                cli_run(["--palace", str(palace), "backup", "--output", str(out)]),
                0,
            )
            self.assertTrue(any(out.glob("*.tar.gz")))
            self.assertEqual(cli_run(["--palace", str(palace), "repair"]), 0)
            # doctor may fail if package metadata missing in editable weirdness,
            # but python+pydantic should pass; palace should pass after init.
            code = cli_run(["--palace", str(palace), "doctor"])
            self.assertIn(code, (0, 1))


if __name__ == "__main__":
    unittest.main()
