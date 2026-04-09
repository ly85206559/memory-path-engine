import unittest

from memory_engine.activation import ActivationSignal, DefaultPropagationPolicy
from memory_engine.schema import MemoryEdge


class ActivationPolicyTests(unittest.TestCase):
    def test_default_policy_applies_decay_and_edge_weight(self):
        policy = DefaultPropagationPolicy(activation_decay=0.5, activation_threshold=0.1)

        step = policy.propagate(
            signal=ActivationSignal(node_id="n1", activation=0.8, hop=0),
            edge=MemoryEdge(from_id="n1", to_id="n2", edge_type="depends_on", weight=0.9),
        )

        self.assertAlmostEqual(step.propagated_activation, 0.36)
        self.assertIsNone(step.stopped_reason)

    def test_default_policy_stops_below_threshold(self):
        policy = DefaultPropagationPolicy(activation_decay=0.5, activation_threshold=0.5)

        step = policy.propagate(
            signal=ActivationSignal(node_id="n1", activation=0.8, hop=0),
            edge=MemoryEdge(from_id="n1", to_id="n2", edge_type="depends_on", weight=0.9),
        )

        self.assertEqual(step.stopped_reason, "below_threshold")

    def test_default_policy_rejects_disallowed_edge_type(self):
        policy = DefaultPropagationPolicy(allowed_edge_types={"cites"})

        step = policy.propagate(
            signal=ActivationSignal(node_id="n1", activation=0.8, hop=0),
            edge=MemoryEdge(from_id="n1", to_id="n2", edge_type="depends_on", weight=1.0),
        )

        self.assertEqual(step.stopped_reason, "disallowed_edge_type")
