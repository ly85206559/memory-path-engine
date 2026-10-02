import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageMetadataTests(unittest.TestCase):
    def test_pyproject_has_pypi_ready_metadata(self) -> None:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = data["project"]
        self.assertEqual(project["name"], "memory-path-engine")
        self.assertRegex(project["version"], r"^\d+\.\d+\.\d+")
        self.assertIn("Homepage", project["urls"])
        self.assertIn("Changelog", project["urls"])
        self.assertIn("Publish", project["urls"])
        self.assertGreaterEqual(len(project.get("classifiers", [])), 5)
        self.assertIn("pydantic>=2.0", project["dependencies"])
        optional = project["optional-dependencies"]
        self.assertIn("embed", optional)
        self.assertIn("dev", optional)
        scripts = project["scripts"]
        self.assertEqual(scripts["mpe"], "memory_engine.cli:main")
        self.assertEqual(scripts["mpe-mcp"], "memory_engine.mcp_server:main")

    def test_changelog_mentions_current_version(self) -> None:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        version = data["project"]["version"]
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"## {version}", changelog)

    def test_publish_workflow_exists(self) -> None:
        workflow = ROOT / ".github" / "workflows" / "publish.yml"
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("pypa/gh-action-pypi-publish", text)
        self.assertIn("id-token: write", text)
        self.assertIn("environment: pypi", text)


if __name__ == "__main__":
    unittest.main()
