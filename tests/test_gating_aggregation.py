# ── tests/test_gating_aggregation.py ──
"""Tests for performance gating and the aggregation strategies.

The headline test is
:meth:`TestGateReproducesRecordedCurves.test_hard_gate_reproduces_the_recorded_curve`,
which replays the per-round signal sequence recorded in
``docs/experiment_tracker.md:39-56`` through the ported gate and checks that the
resulting global-signal trajectory matches the published one. That curve is the
project's strongest result — 0.30 (no gate) → 0.92 (hard gate) mAP50 — and it
previously lived only inside a GPU training script.
"""

import os
import sys
import unittest
from itertools import pairwise
from typing import ClassVar

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.aggregation import (
    AggregationError,
    AggregationStrategy,
    aggregate,
    aggregate_with_report,
    blend,
    fedavg,
    fedavg_unweighted,
    fedprox_penalty,
    normalise_weights,
)
from analysis.gating import GateDecision, GateStrategy, PerformanceGate, UnknownGateError


class TestGateStrategyParsing(unittest.TestCase):
    def test_accepts_names_and_instances(self):
        self.assertIs(GateStrategy.parse("hard"), GateStrategy.HARD)
        self.assertIs(GateStrategy.parse("  SOFT  "), GateStrategy.SOFT)
        self.assertIs(GateStrategy.parse(GateStrategy.LOCAL), GateStrategy.LOCAL)
        self.assertIs(GateStrategy.parse(None), GateStrategy.NONE)

    def test_unknown_strategy_raises_with_valid_options(self):
        with self.assertRaises(UnknownGateError) as ctx:
            GateStrategy.parse("nonsense")
        self.assertIn("hard", str(ctx.exception))


class TestGateReproducesRecordedCurves(unittest.TestCase):
    """Replay the recorded per-round signals through the ported policies."""

    #: ``docs/experiment_tracker.md:45-56`` — per-round mAP50 at the visited site.
    RECORDED: ClassVar[dict[str, list[float]]] = {
        "none": [0.28, 0.32, 0.33, 0.31, 0.31, 0.28, 0.28, 0.30, 0.30, 0.30],
        "soft": [0.28, 0.32, 0.35, 0.35, 0.35, 0.58, 0.65, 0.67, 0.72, 0.75],
        "hard": [0.37, 0.31, 0.74, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92],
    }

    def _replay(self, strategy):
        gate = PerformanceGate(strategy=strategy, margin=0.0, ewa_warmup_rounds=2)
        trajectory, accepted = [], []
        for rnd, signal in enumerate(self.RECORDED[strategy.value]):
            decision = gate.decide(
                signal, local_samples=8, round_index=rnd, signal_global=gate.global_signal
            )
            trajectory.append(gate.global_signal)
            accepted.append(decision.accept)
        return trajectory, accepted

    def test_none_gate_tracks_every_candidate(self):
        trajectory, accepted = self._replay(GateStrategy.NONE)
        self.assertTrue(all(accepted), "the unconditional gate must never reject")
        for recorded, observed in zip(self.RECORDED["none"], trajectory, strict=True):
            self.assertAlmostEqual(observed, recorded, places=6)

    def test_hard_gate_reproduces_the_recorded_curve(self):
        """0.37 -> 0.37 -> 0.74 -> 0.92 and then holds."""
        trajectory, accepted = self._replay(GateStrategy.HARD)

        expected = [0.37, 0.37, 0.74, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92]
        for recorded, observed in zip(expected, trajectory, strict=True):
            self.assertAlmostEqual(observed, recorded, places=6)

        # Round 1 regressed and was refused; round 4 onwards never improved.
        self.assertEqual(accepted, [True, False, True, True] + [False] * 6)
        self.assertAlmostEqual(trajectory[-1], 0.92, places=6)

    def test_hard_gate_beats_none_by_a_wide_margin(self):
        """The whole point of the change, asserted as a number."""
        none_final = self._replay(GateStrategy.NONE)[0][-1]
        hard_final = self._replay(GateStrategy.HARD)[0][-1]
        self.assertAlmostEqual(none_final, 0.30, places=6)
        self.assertAlmostEqual(hard_final, 0.92, places=6)
        self.assertGreater(hard_final / none_final, 3.0)

    def test_soft_gate_rises_monotonically(self):
        trajectory, accepted = self._replay(GateStrategy.SOFT)
        self.assertTrue(all(accepted))
        for earlier, later in pairwise(trajectory):
            self.assertLessEqual(earlier, later + 1e-9)
        self.assertGreater(trajectory[-1], 0.6)


class TestGateSemantics(unittest.TestCase):
    def test_hard_gate_initialises_from_the_first_candidate(self):
        gate = PerformanceGate(strategy="hard")
        decision = gate.decide(0.5, local_samples=10)
        self.assertTrue(decision.accept)
        self.assertEqual(decision.weight_local, 1.0)
        self.assertAlmostEqual(gate.global_signal, 0.5)

    def test_hard_gate_margin_is_strict(self):
        """An improvement of exactly ``margin`` is not an improvement."""
        gate = PerformanceGate(strategy="hard", margin=0.05)
        gate.decide(0.80, local_samples=10)
        at_margin = gate.decide(0.85, local_samples=10)
        self.assertFalse(at_margin.accept)
        above = gate.decide(0.851, local_samples=10)
        self.assertTrue(above.accept)

    def test_hard_gate_rejection_returns_unit_weight_on_global(self):
        gate = PerformanceGate(strategy="hard")
        gate.decide(0.9, local_samples=10)
        decision = gate.decide(0.1, local_samples=10)
        self.assertFalse(decision.accept)
        self.assertEqual(decision.weight_local, 0.0)
        self.assertEqual(decision.weight_global, 1.0)
        self.assertIn("keeping global", decision.reason)

    def test_min_signal_floor_refuses_updates_in_a_broken_regime(self):
        """When everything is near zero, a relative margin is not trustworthy."""
        gate = PerformanceGate(strategy="hard", min_signal=0.2)
        gate.decide(0.30, local_samples=10)
        decision = gate.decide(0.05, local_samples=10)
        self.assertFalse(decision.accept)
        self.assertIn("absolute floor", decision.reason)

    def test_none_gate_weights_by_sample_volume(self):
        gate = PerformanceGate(strategy="none")
        gate.decide(0.5, local_samples=100)
        decision = gate.decide(0.5, local_samples=300, signal_global=gate.global_signal)
        self.assertAlmostEqual(decision.weight_local, 300 / 400, places=6)
        self.assertAlmostEqual(decision.weight_global, 100 / 400, places=6)

    def test_soft_gate_uses_volume_weights_during_warmup(self):
        gate = PerformanceGate(strategy="soft", ewa_warmup_rounds=2)
        gate.decide(0.5, local_samples=100, round_index=0)
        warmup = gate.decide(0.2, local_samples=300, round_index=1)
        self.assertIn("warmup", warmup.reason)
        self.assertAlmostEqual(warmup.weight_local, 300 / 400, places=6)

    def test_soft_gate_switches_to_performance_weighting_after_warmup(self):
        gate = PerformanceGate(strategy="soft", ewa_warmup_rounds=1)
        gate.decide(0.2, local_samples=100, round_index=0)
        # Round 1 is past a one-round warmup: a stronger candidate must dominate.
        decision = gate.decide(0.8, local_samples=100, round_index=1)
        self.assertIn("performance-weighted", decision.reason)
        self.assertGreater(decision.weight_local, 0.5)

    def test_soft_gate_never_lets_a_zero_signal_zero_its_own_weight(self):
        gate = PerformanceGate(strategy="soft", ewa_warmup_rounds=0)
        gate.decide(0.5, local_samples=10, round_index=0)
        decision = gate.decide(0.0, local_samples=10, round_index=1)
        self.assertGreater(decision.weight_local, 0.0)

    def test_lower_is_better_flips_the_comparison(self):
        gate = PerformanceGate(strategy="hard", higher_is_better=False)
        gate.decide(1.0, local_samples=10)
        self.assertFalse(gate.decide(1.2, local_samples=10).accept)
        self.assertTrue(gate.decide(0.8, local_samples=10).accept)

    def test_local_gate_requires_a_node_name(self):
        gate = PerformanceGate(strategy="local")
        with self.assertRaises(UnknownGateError):
            gate.decide(0.5, local_samples=10)

    def test_local_gate_keeps_each_nodes_best_independently(self):
        gate = PerformanceGate(strategy="local")
        self.assertTrue(gate.decide(0.9, node_name="b1").accept)
        self.assertTrue(gate.decide(0.4, node_name="b2").accept)
        self.assertFalse(gate.decide(0.5, node_name="b1").accept)
        self.assertTrue(gate.decide(0.6, node_name="b2").accept)
        self.assertEqual(gate.node_best, {"b1": 0.9, "b2": 0.6})

    def test_reset_clears_state(self):
        gate = PerformanceGate(strategy="hard")
        gate.decide(0.9, local_samples=10)
        gate.reset()
        self.assertIsNone(gate.global_signal)
        self.assertFalse(gate.is_initialised)
        self.assertEqual(gate.global_sample_count, 0)

    def test_decision_is_serialisable_for_an_audit_trail(self):
        gate = PerformanceGate(strategy="hard", margin=0.02)
        gate.decide(0.5, local_samples=10, round_index=0)
        payload = gate.decide(0.4, local_samples=10, round_index=1).to_dict()
        for key in ("strategy", "accept", "weight_local", "weight_global", "reason"):
            self.assertIn(key, payload)


class TestWeightNormalisation(unittest.TestCase):
    def test_normalises_to_unit_sum(self):
        self.assertEqual(normalise_weights([1, 3]), [0.25, 0.75])

    def test_length_mismatch_raises(self):
        with self.assertRaises(AggregationError):
            normalise_weights([1, 2, 3], n=2)

    def test_negative_weight_raises(self):
        with self.assertRaises(AggregationError):
            normalise_weights([1, -1])

    def test_all_zero_raises_rather_than_silently_uniform(self):
        with self.assertRaises(AggregationError) as ctx:
            normalise_weights([0, 0, 0])
        self.assertIn("sum to zero", str(ctx.exception))


class TestAggregationPrimitives(unittest.TestCase):
    def setUp(self):
        self.clients = [
            {"w": np.full(3, float(i + 1)), "nbt": np.int64(7 + i)}
            for i in range(3)
        ]

    def test_weighted_fedavg_lets_a_large_client_dominate(self):
        result = fedavg(self.clients, [100, 700, 200])
        expected = (1 * 100 + 2 * 700 + 3 * 200) / 1000
        np.testing.assert_allclose(result["w"], np.full(3, expected), rtol=1e-6)

    def test_unweighted_fedavg_gives_equal_say(self):
        result = fedavg_unweighted(self.clients)
        np.testing.assert_allclose(result["w"], np.full(3, 2.0), rtol=1e-6)

    def test_non_float_buffers_are_not_averaged(self):
        """A fractional ``num_batches_tracked`` would fail to load back."""
        result = fedavg(self.clients, [1, 1, 1])
        self.assertEqual(int(result["nbt"]), 7)
        self.assertTrue(np.issubdtype(np.asarray(result["nbt"]).dtype, np.integer))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(AggregationError) as ctx:
            fedavg([{"w": np.zeros(2)}, {"w": np.zeros(3)}], [1, 1])
        self.assertIn("shape", str(ctx.exception))

    def test_key_mismatch_is_rejected(self):
        with self.assertRaises(AggregationError):
            fedavg([{"a": np.zeros(2)}, {"b": np.zeros(2)}], [1, 1])

    def test_empty_client_list_is_rejected(self):
        with self.assertRaises(AggregationError):
            fedavg([])

    def test_blend_interpolates(self):
        result = blend({"w": np.ones(3)}, {"w": np.zeros(3)}, 0.25)
        np.testing.assert_allclose(result["w"], np.full(3, 0.25))

    def test_blend_endpoints_short_circuit(self):
        primary = {"w": np.ones(3)}
        secondary = {"w": np.zeros(3)}
        np.testing.assert_allclose(blend(primary, secondary, 1.0)["w"], np.ones(3))
        np.testing.assert_allclose(blend(primary, secondary, 0.0)["w"], np.zeros(3))

    def test_blend_rejects_out_of_range_weight(self):
        with self.assertRaises(AggregationError):
            blend({"w": np.ones(3)}, {"w": np.zeros(3)}, 1.5)


class TestFedProx(unittest.TestCase):
    def test_zero_mu_is_plain_fedavg(self):
        local = {"w": np.ones(3)}
        glob = {"w": np.zeros(3)}
        self.assertEqual(float(fedprox_penalty(local, glob, 0.0)), 0.0)

    def test_penalty_grows_with_distance(self):
        glob = {"w": np.zeros(3)}
        near = float(fedprox_penalty({"w": np.ones(3)}, glob, 1.0))
        far = float(fedprox_penalty({"w": np.full(3, 3.0)}, glob, 1.0))
        self.assertLess(near, far)

    def test_penalty_scales_with_mu(self):
        glob = {"w": np.zeros(3)}
        local = {"w": np.ones(3)}
        self.assertAlmostEqual(
            float(fedprox_penalty(local, glob, 0.4)) / float(fedprox_penalty(local, glob, 0.2)),
            2.0, places=6,
        )

    def test_penalty_is_half_mu_squared_distance(self):
        glob = {"w": np.zeros(3)}
        local = {"w": np.full(3, 2.0)}
        # (mu/2) * ||delta||^2 = (0.5) * 3 * 4 = 6
        self.assertAlmostEqual(float(fedprox_penalty(local, glob, 1.0)), 6.0, places=6)

    def test_negative_mu_rejected(self):
        with self.assertRaises(AggregationError):
            fedprox_penalty({"w": np.ones(2)}, {"w": np.zeros(2)}, -0.1)


class TestAggregateEntryPoint(unittest.TestCase):
    def setUp(self):
        self.clients = [{"w": np.array([float(i + 1)] * 2)} for i in range(3)]
        self.previous = {"w": np.array([9.0, 9.0])}

    def test_ungated_returns_the_candidate(self):
        result = aggregate(self.clients, client_weights=[1, 1, 1])
        np.testing.assert_allclose(result["w"], np.array([2.0, 2.0]))

    def test_accept_decision_replaces_global(self):
        decision = GateDecision(
            strategy=GateStrategy.HARD, accept=True, weight_local=1.0, weight_global=0.0,
            reason="better",
        )
        report = aggregate_with_report(
            self.clients, client_weights=[1, 1, 1],
            gate_decision=decision, previous_global=self.previous,
        )
        self.assertAlmostEqual(float(report["w"][0]), 2.0, places=6)
        self.assertEqual(report.meta["applied_weights"]["candidate"], 1.0)

    def test_reject_decision_returns_the_previous_global_untouched(self):
        decision = GateDecision(
            strategy=GateStrategy.HARD, accept=False, weight_local=0.0, weight_global=1.0,
            reason="worse",
        )
        report = aggregate_with_report(
            self.clients, client_weights=[1, 1, 1],
            gate_decision=decision, previous_global=self.previous,
        )
        np.testing.assert_allclose(report["w"], self.previous["w"])
        self.assertIn("gate", report.meta)

    def test_blend_decision_interpolates(self):
        decision = GateDecision(
            strategy=GateStrategy.SOFT, accept=True, weight_local=0.5, weight_global=0.5,
            reason="ewa",
        )
        report = aggregate_with_report(
            self.clients, client_weights=[1, 1, 1],
            gate_decision=decision, previous_global=self.previous,
        )
        self.assertAlmostEqual(float(report["w"][0]), 0.5 * 2.0 + 0.5 * 9.0, places=6)

    def test_gate_needing_global_without_one_raises(self):
        decision = GateDecision(
            strategy=GateStrategy.HARD, accept=False, weight_local=0.0, weight_global=1.0,
            reason="worse",
        )
        with self.assertRaises(AggregationError):
            aggregate_with_report(self.clients, gate_decision=decision, previous_global=None)

    def test_unweighted_strategy_is_selectable(self):
        result = aggregate(self.clients, strategy=AggregationStrategy.FEDAVG_UNWEIGHTED)
        np.testing.assert_allclose(result["w"], np.array([2.0, 2.0]))

    def test_report_metadata_records_the_strategy(self):
        report = aggregate_with_report(self.clients, client_weights=[1, 2, 3])
        self.assertEqual(report.meta["strategy"], "fedavg")
        self.assertEqual(report.meta["n_clients"], 3)
        self.assertFalse(report.meta["gated"])

    def test_report_can_be_reaggregated(self):
        """``_meta`` must not leak into the parameter namespace."""
        first = aggregate_with_report(self.clients, client_weights=[1, 1, 1])
        second = aggregate_with_report([dict(first)] * 2, client_weights=[1, 1])
        self.assertNotIn("_meta", second.params)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(AggregationError):
            aggregate(self.clients, strategy="not-a-strategy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
