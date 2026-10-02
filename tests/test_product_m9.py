import tempfile
import unittest
from pathlib import Path

from memory_engine.palace_ops import doctor_report


class DoctorInstallTipsTests(unittest.TestCase):
    def test_doctor_tips_prefer_pypi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = doctor_report(Path(tmp) / "missing-palace")
        tips = "\n".join(report["install_tips"])
        self.assertIn("pip install memory-path-engine", tips)
        self.assertIn("pipx install memory-path-engine", tips)
        self.assertNotIn("git+https://github.com/ly85206559/memory-path-engine.git", tips)


if __name__ == "__main__":
    unittest.main()
