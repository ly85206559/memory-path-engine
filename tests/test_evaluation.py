import unittest
import json
from pathlib import Path

from memory_engine.evaluation import run_evaluation_suite


class EvaluationTests(unittest.TestCase):
    def test_detailed_evaluation_suite_returns_comparison_report(self):
        contracts_dir = Path("examples/contract_pack/contracts")
        questions_path = Path("examples/contract_pack/eval/questions.json")

        report = run_evaluation_suite(contracts_dir, questions_path, detailed=True)

        self.assertIn("modes", report)
        self.assertIn("comparison", report)
        self.assertIn("weighted_graph", report["modes"])
        self.assertIn("activation_spreading_v1", report["modes"])
        self.assertIn("per_question", report["comparison"])
        self.assertTrue(report["comparison"]["per_question"])
        self.assertIn("evidence_hit_rate", report["modes"]["weighted_graph"])
        self.assertIn("avg_latency_ms", report["comparison"]["mode_summary"]["weighted_graph"])
        self.assertIn(
            "activation_trace_hit_rate",
            report["comparison"]["mode_summary"]["activation_spreading_v1"],
        )
        self.assertIn(
            "evidence_hit_rate",
            report["comparison"]["mode_summary"]["activation_spreading_v1"],
        )
        expected_question_count = len(json.loads(questions_path.read_text(encoding="utf-8")))
        self.assertEqual(
            len(report["comparison"]["per_question"]),
            expected_question_count,
        )
        first_question = report["comparison"]["per_question"][0]
        self.assertEqual(
            set(first_question["modes"].keys()),
            set(report["modes"].keys()),
        )
        self.assertEqual(
            set(first_question["best_modes"]),
            {
                mode_name
                for mode_name, mode_result in first_question["modes"].items()
                if mode_result["hit"]
            },
        )
        self.assertEqual(
            first_question["missed_by_all"],
            not any(mode_result["hit"] for mode_result in first_question["modes"].values()),
        )
        self.assertTrue(
            {
                "evidence_hit",
                "hit",
                "path_hit",
                "activation_trace_hit",
                "semantic_hit",
                "contradiction_hit",
                "matched_evidence",
                "latency_ms",
            }.issubset(first_question["modes"]["activation_spreading_v1"].keys())
        )
