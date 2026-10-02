import unittest
from types import SimpleNamespace
from unittest.mock import patch

from memory_engine.cli import _cmd_bench_longmemeval


class BenchLabelSuffixTests(unittest.TestCase):
    def test_turn_and_embedding_suffixes(self) -> None:
        captured: dict = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {
                "dataset": "x",
                "samples": 0,
                "json_path": "j",
                "markdown_path": "m",
                "summary": {"label": kwargs["label"], "modes": {}},
            }

        args = SimpleNamespace(
            label="medium50",
            dataset=None,
            limit=50,
            modes="hybrid",
            top_k=10,
            output_dir=None,
            granularity="turn",
            embedding="fastembed",
        )
        with patch("memory_engine.cli.run_longmemeval_baseline", side_effect=fake_run), patch(
            "memory_engine.cli.format_longmemeval_baseline_markdown", return_value=""
        ):
            self.assertEqual(_cmd_bench_longmemeval(args), 0)
        self.assertEqual(captured["label"], "medium50-turn-fastembed")
        self.assertEqual(captured["granularity"], "turn")
        self.assertEqual(captured["embedding"], "fastembed")

    def test_ngram_does_not_suffix_embedding(self) -> None:
        captured: dict = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {
                "dataset": "x",
                "samples": 0,
                "json_path": "j",
                "markdown_path": "m",
                "summary": {"label": kwargs["label"], "modes": {}},
            }

        args = SimpleNamespace(
            label="medium50",
            dataset=None,
            limit=50,
            modes="hybrid",
            top_k=10,
            output_dir=None,
            granularity="session",
            embedding="ngram",
        )
        with patch("memory_engine.cli.run_longmemeval_baseline", side_effect=fake_run), patch(
            "memory_engine.cli.format_longmemeval_baseline_markdown", return_value=""
        ):
            self.assertEqual(_cmd_bench_longmemeval(args), 0)
        self.assertEqual(captured["label"], "medium50")


if __name__ == "__main__":
    unittest.main()
