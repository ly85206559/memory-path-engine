import unittest
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
        self.assertIn("avg_latency_ms", report["comparison"]["mode_summary"]["weighted_graph"])
