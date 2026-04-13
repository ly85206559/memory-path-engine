import unittest


class PublicBenchmarkMetricTests(unittest.TestCase):
    def test_compute_recall_at_k_hits_when_any_gold_item_present(self):
        from memory_engine.benchmarking.application.public_benchmarks import compute_recall_at_k

        self.assertEqual(
            compute_recall_at_k(["gold-a", "gold-b"], ["x", "gold-b", "y"], k=2),
            1.0,
        )
        self.assertEqual(
            compute_recall_at_k(["gold-a", "gold-b"], ["x", "y", "gold-b"], k=2),
            0.0,
        )

    def test_compute_ndcg_at_k_matches_single_gold_rank(self):
        from memory_engine.benchmarking.application.public_benchmarks import compute_ndcg_at_k

        expected = 1.0 / 1.584962500721156
        self.assertAlmostEqual(
            compute_ndcg_at_k(["gold"], ["x", "gold", "y"], k=10),
            expected,
            places=6,
        )

    def test_matched_ranks_returns_sorted_positions(self):
        from memory_engine.benchmarking.application.public_benchmarks import matched_ranks

        self.assertEqual(
            matched_ranks(["b", "d"], ["a", "b", "c", "d"]),
            [2, 4],
        )
