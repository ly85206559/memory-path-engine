import unittest

from memory_engine.memory_state import (
    MemoryStatePolicy,
    StaticMemoryStatePolicy,
    decay_unvisited_nodes,
    reinforce_result_paths,
)
from memory_engine.replay import path_answer
from memory_engine.schema import MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore


class MemoryStateTests(unittest.TestCase):
    def test_reinforce_result_paths_increments_usage(self):
        store = MemoryStore()
        node = MemoryNode(
            id="memory:1",
            type="clause",
            content="Buyer must pay invoices within 30 days.",
            weights=MemoryWeight(),
        )
        store.add_node(node)

        path = path_answer("payment", [(node, 0.8, "hit", None)])
        reinforce_result_paths(store, paths=[path])

        self.assertEqual(store.get_node("memory:1").weights.usage_count, 1)

    def test_decay_unvisited_nodes_reduces_decay_factor(self):
        store = MemoryStore()
        store.add_node(
            MemoryNode(
                id="memory:1",
                type="clause",
                content="Buyer must pay invoices within 30 days.",
                weights=MemoryWeight(decay_factor=1.0),
            )
        )

        decay_unvisited_nodes(store, visited_node_ids=set(), steps=2)

        self.assertLess(store.get_node("memory:1").weights.decay_factor, 1.0)

    def test_effective_weight_score_rewards_usage_but_respects_decay(self):
        policy = MemoryStatePolicy()
        weight = MemoryWeight(importance=0.6, risk=0.2, novelty=0.1, confidence=0.9, usage_count=3, decay_factor=0.8)

        score = policy.effective_weight_score(weight)

        self.assertGreater(score, weight.bounded_score() * 0.8)
        self.assertLessEqual(score, 1.0)

    def test_static_memory_state_policy_is_noop(self):
        policy = StaticMemoryStatePolicy()
        node = MemoryNode(
            id="memory:2",
            type="clause",
            content="Buyer must pay invoices within 30 days.",
            weights=MemoryWeight(usage_count=2, decay_factor=0.9),
        )

        policy.reinforce_node(node)
        policy.decay_node(node, steps=3)

        self.assertEqual(node.weights.usage_count, 2)
        self.assertEqual(node.weights.decay_factor, 0.9)
        self.assertEqual(policy.effective_weight_score(node.weights), node.weights.bounded_score())

    def test_propagation_factor_differs_for_dynamic_and_static_policies(self):
        dynamic_policy = MemoryStatePolicy()
        static_policy = StaticMemoryStatePolicy()
        node = MemoryNode(
            id="memory:3",
            type="clause",
            content="Buyer must pay invoices within 30 days.",
            weights=MemoryWeight(decay_factor=0.6),
        )

        self.assertEqual(dynamic_policy.propagation_factor(node), 0.6)
        self.assertEqual(static_policy.propagation_factor(node), 1.0)
