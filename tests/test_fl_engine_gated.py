# ── tests/test_fl_engine_gated.py ──
"""Tests for the gated / FedProx paths of the federated learning engine.

Split from ``test_fl_engine.py`` so the existing engine contract (imported and
asserted there) stays visibly untouched by this work.
"""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.aggregation import AggregationStrategy
from analysis.fl_engine import (
    FLEngine,
    OrganoidClassifier,
    fedavg_aggregate,
    get_params,
    set_params,
    train_client,
)
from analysis.gating import GateStrategy, PerformanceGate

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def separable_data(n=600, dim=32, seed=0):
    """Three mildly overlapping classes, so training does something real."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim).astype(np.float32)
    y = (
        (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
        + (X[:, 2] > 0.3).astype(np.int64)
    )
    return X, y


@unittest.skipIf(torch is None, "torch not installed")
class TestExistingContractUnchanged(unittest.TestCase):
    """The pre-existing entry points must keep behaving as before."""

    def test_fedavg_aggregate_unweighted_default(self):
        models = [OrganoidClassifier(input_dim=32, num_classes=2, hidden_dim=8) for _ in range(3)]
        params = [get_params(m) for m in models]
        avg = fedavg_aggregate(params)
        self.assertEqual(set(avg.keys()), set(params[0].keys()))
        for key in avg:
            expected = sum(p[key].data for p in params) / 3
            self.assertTrue(torch.allclose(avg[key].data, expected))

    def test_fedavg_aggregate_with_weights(self):
        models = [OrganoidClassifier(input_dim=32, num_classes=2, hidden_dim=8) for _ in range(2)]
        params = [get_params(m) for m in models]
        avg = fedavg_aggregate(params, [3, 1])
        for key in avg:
            expected = (params[0][key].data * 0.75) + (params[1][key].data * 0.25)
            self.assertTrue(torch.allclose(avg[key].data, expected, atol=1e-6))

    def test_train_client_signature_still_works_positionally(self):
        model = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        X = np.random.randn(64, 16).astype(np.float32)
        y = np.random.randint(0, 3, 64).astype(np.int64)
        params, _loss, acc = train_client(model, X, y, 0.01, 2, 16)
        self.assertEqual(set(params.keys()), set(dict(model.named_parameters()).keys()))
        self.assertGreaterEqual(acc, 0.0)

    def test_parallel_run_is_unchanged(self):
        X, y = separable_data()
        engine = FLEngine(input_dim=32, num_classes=3, hidden_dim=16, lr=0.01, local_epochs=2)
        history = engine.run(X, y, n_clients=3, rounds=3)
        self.assertEqual(len(history), 3)
        self.assertTrue(all("val_acc" in h for h in history))
        self.assertGreater(history[-1]["val_acc"], 0.1)


@unittest.skipIf(torch is None, "torch not installed")
class TestFedProx(unittest.TestCase):
    def test_proximal_term_is_applied_when_mu_is_positive(self):
        """Two runs that differ *only* in ``prox_mu`` must differ in loss.

        Both models start from identical weights and see identical batch
        shuffling, so the only asymmetry is the added penalty — which is why
        the assertion is strict rather than loose. Comparing two freely
        initialised models (as this test used to) measures initialisation noise
        as much as the proximal term.
        """
        X, y = separable_data(200, 16)
        anchor = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        anchor_params = get_params(anchor)

        def run(mu):
            model = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
            set_params(model, {k: v.clone() for k, v in anchor_params.items()})
            torch.manual_seed(0)
            _, loss, _ = train_client(
                model, X, y, lr=0.01, epochs=1,
                global_params=anchor_params, prox_mu=mu,
            )
            return loss

        loss_plain, loss_prox = run(0.0), run(0.5)
        # The penalty is added to the reported objective, so with the weight
        # trajectory held identical the proximal run cannot report less.
        self.assertGreaterEqual(loss_prox, loss_plain)
        # And it must be a real contribution, not a rounding artefact.
        self.assertGreater(loss_prox - loss_plain, 1e-4)

    def test_the_anchor_is_the_global_model_not_the_local_weights(self):
        """FedProx must pull towards ``global_params``, not towards where the
        client happens to start.

        Regression guard: the anchor used to be read off the model's own
        parameters, so passing a genuinely different global model had no effect
        and the term quietly degraded to plain FedAvg.

        The client is deliberately started far from ``global_params`` and the
        proximal coefficient is made enormous, so the penalty dominates the
        cross-entropy gradient. The only thing left for the optimiser to do is
        minimise ``||w - w_global||²``: the client must therefore end up near
        ``w_global``. Anchoring on the client's own starting weights instead
        pins it where it began, roughly ``initial_distance`` away — which is
        what makes this test discriminate between the two implementations.

        A "strong anchor means less movement" assertion would be backwards: the
        penalty gradient ``mu*(w - w_global)`` actively drags the client across
        to ``w_global``, so it can travel further than local training alone.
        """
        X, y = separable_data(200, 16)
        reference = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        far_global = get_params(reference)

        def distance_to_global(prox_mu, epochs):
            model = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
            torch.manual_seed(5)
            for p in model.parameters():
                with torch.no_grad():
                    p.add_(torch.randn_like(p) * 0.3)

            def distance(params):
                return sum(
                    float(((params[n].detach() - far_global[n].detach()) ** 2).sum())
                    for n in far_global
                )

            initial = distance(get_params(model))
            torch.manual_seed(0)
            params, _, _ = train_client(
                model, X, y, lr=0.05, epochs=epochs,
                global_params=far_global, prox_mu=prox_mu,
            )
            return initial, distance(params)

        initial, final_tight = distance_to_global(prox_mu=1e6, epochs=60)
        self.assertGreater(initial, 0.0)
        self.assertLess(
            final_tight, 0.1 * initial,
            "a dominant proximal term did not pull the client towards "
            "global_params; the anchor is being read from the model's own "
            "weights instead",
        )

    def test_positive_mu_without_global_params_raises(self):
        X, y = separable_data(50, 16)
        model = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        with self.assertRaises(ValueError):
            train_client(model, X, y, prox_mu=0.5)

    def test_strong_proximal_term_holds_the_model_near_the_global_one(self):
        X, y = separable_data(200, 16, seed=3)
        anchor = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        anchor_params = get_params(anchor)

        loose = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        set_params(loose, {k: v.clone() for k, v in anchor_params.items()})
        train_client(loose, X, y, lr=0.05, epochs=3, global_params=anchor_params, prox_mu=0.0)

        tight = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        set_params(tight, {k: v.clone() for k, v in anchor_params.items()})
        train_client(tight, X, y, lr=0.05, epochs=3, global_params=anchor_params, prox_mu=50.0)

        def drift(model):
            return sum(
                float(((p.detach() - anchor_params[n].detach()) ** 2).sum())
                for n, p in model.named_parameters()
            )

        self.assertLess(drift(tight), drift(loose))


@unittest.skipIf(torch is None, "torch not installed")
class TestSequentialGatedRun(unittest.TestCase):
    def _run(self, gate, **kw):
        X, y = separable_data()
        engine = FLEngine(
            input_dim=32, num_classes=3, hidden_dim=16, lr=0.01, local_epochs=3,
            n_clients=3, non_iid=0.6, seed=7,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return engine.run_sequential(X, y, gate=gate, rounds=4, **kw)

    def test_all_gates_return_a_trajectory(self):
        for gate in (GateStrategy.NONE, GateStrategy.SOFT, GateStrategy.HARD):
            history = self._run(gate)
            self.assertEqual(len(history), 4)
            self.assertTrue(all("val_acc" in h for h in history))
            self.assertTrue(all(h["gate_strategy"] == gate.value for h in history))

    def test_none_gate_accepts_every_client(self):
        history = self._run(GateStrategy.NONE)
        for round_metrics in history:
            self.assertEqual(round_metrics["rejected"], 0)

    def test_hard_gate_rejects_something(self):
        """A gate that never refuses is not a gate."""
        history = self._run(GateStrategy.HARD)
        self.assertGreater(sum(h["rejected"] for h in history), 0)

    def test_hard_gate_does_not_lose_to_no_gate(self):
        """The recorded phenomenon, in miniature: gating should not hurt.

        This compared two independent stochastic runs and was therefore flaky
        until ``FLEngine.seed`` was made to drive the torch RNG as well as the
        data split (see ``TestReproducibility``). Both runs now start from the
        same initialisation, so the comparison is meaningful.
        """
        none_final = self._run(GateStrategy.NONE)[-1]["val_acc"]
        hard_final = self._run(GateStrategy.HARD)[-1]["val_acc"]
        self.assertGreaterEqual(hard_final + 0.02, none_final)

    def test_client_order_is_honoured(self):
        X, y = separable_data()
        engine = FLEngine(input_dim=32, num_classes=3, hidden_dim=16, lr=0.01, local_epochs=1,
                          n_clients=3, seed=7)
        history = engine.run_sequential(X, y, gate="hard", rounds=1, client_order=[2, 0, 1])
        self.assertEqual([c["client"] for c in history[0]["client_metrics"]], [3, 1, 2])

    def test_invalid_client_order_raises(self):
        X, y = separable_data()
        engine = FLEngine(input_dim=32, num_classes=3, hidden_dim=16, lr=0.01, local_epochs=1,
                          n_clients=3, seed=7)
        with self.assertRaises(ValueError):
            engine.run_sequential(X, y, gate="hard", rounds=1, client_order=[0, 1])

    def test_an_existing_gate_object_is_used_as_is(self):
        X, y = separable_data()
        engine = FLEngine(input_dim=32, num_classes=3, hidden_dim=16, lr=0.01, local_epochs=1,
                          n_clients=3, seed=7)
        gate = PerformanceGate(strategy="hard", margin=0.4)  # deliberately strict
        history = engine.run_sequential(X, y, gate=gate, rounds=3)
        self.assertEqual(history[0]["gate_strategy"], "hard")
        # A 0.4 margin should refuse most updates.
        self.assertGreater(sum(h["rejected"] for h in history), 0)

    def test_decisions_are_recorded_for_audit(self):
        history = self._run(GateStrategy.HARD)
        for client_metrics in history[0]["client_metrics"]:
            payload = client_metrics["decision"]
            for key in ("strategy", "accept", "weight_local", "weight_global", "reason"):
                self.assertIn(key, payload)
            self.assertIn("signal", client_metrics)

    def test_per_round_totals_are_consistent(self):
        history = self._run(GateStrategy.SOFT)
        for round_metrics in history:
            total = round_metrics["accepted"] + round_metrics["rejected"]
            self.assertEqual(total, len(round_metrics["client_metrics"]))

    def test_progress_callback_fires_each_round(self):
        X, y = separable_data()
        engine = FLEngine(input_dim=32, num_classes=3, hidden_dim=16, lr=0.01, local_epochs=1,
                          n_clients=3, seed=7)
        seen = []
        engine.run_sequential(X, y, gate="hard", rounds=3, progress_callback=lambda r, m: seen.append(r))
        self.assertEqual(seen, [1, 2, 3])

    def test_global_model_is_retained_after_the_run(self):
        X, y = separable_data()
        engine = FLEngine(input_dim=32, num_classes=3, hidden_dim=16, lr=0.01, local_epochs=1,
                          n_clients=3, seed=7)
        engine.run_sequential(X, y, gate="hard", rounds=2)
        self.assertIsNotNone(engine.global_model)
        self.assertIsNotNone(engine.global_params)


@unittest.skipIf(torch is None, "torch not installed")
class TestStrategySelection(unittest.TestCase):
    def test_unweighted_strategy_warns_because_it_ignores_volume(self):
        X, y = separable_data(300, 16)
        engine = FLEngine(
            input_dim=16, num_classes=3, hidden_dim=8, lr=0.01, local_epochs=1,
            strategy=AggregationStrategy.FEDAVG_UNWEIGHTED,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.run(X, y, n_clients=3, rounds=1)
        self.assertTrue(any("equal say" in str(w.message) for w in caught))

    def test_weighted_strategy_does_not_warn(self):
        X, y = separable_data(300, 16)
        engine = FLEngine(input_dim=16, num_classes=3, hidden_dim=8, lr=0.01, local_epochs=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.run(X, y, n_clients=3, rounds=1)
        self.assertFalse(any("equal say" in str(w.message) for w in caught))

    def test_prox_mu_is_recorded_on_the_engine(self):
        engine = FLEngine(input_dim=16, num_classes=3, hidden_dim=8, prox_mu=0.1)
        self.assertEqual(engine.prox_mu, 0.1)


@unittest.skipIf(torch is None, "torch not installed")
class TestNoPandasOnTheTorchPath(unittest.TestCase):
    """Guard the import ordering that crashes the process on Windows.

    Importing pandas *after* torch loads two conflicting OpenMP runtimes and
    terminates the interpreter with no traceback. ``utils/helpers`` is reached
    from inside ``FLEngine.run()``, so a module-scope pandas import there made
    every training run die at the first step.

    ``KMP_DUPLICATE_LIB_OK=TRUE`` does **not** fix it. The only reliable
    remedy is to keep pandas out of the torch import path.
    """

    def test_importing_the_fl_engine_does_not_pull_in_pandas(self):
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys, torch;"
            "import analysis.fl_engine, utils.helpers, visualization.charts;"
            "print('PANDAS_LOADED' if 'pandas' in sys.modules else 'CLEAN')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root, capture_output=True, text=True, timeout=180,
            check=False,
        )
        self.assertEqual(
            proc.returncode, 0,
            f"importing the FL engine after torch killed the process. stderr={proc.stderr[-500:]}",
        )
        self.assertIn(
            "CLEAN", proc.stdout,
            "pandas was imported on the torch path; that ordering crashes the "
            "interpreter on Windows. Move the import inside the function that "
            "needs it, or under TYPE_CHECKING.",
        )

    def test_helpers_module_has_no_module_scope_pandas_import(self):
        import ast

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for relative in ("utils/helpers.py", "visualization/charts.py"):
            with open(os.path.join(repo_root, relative), encoding="utf-8") as fh:
                source = fh.read()
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(
                    any(n.startswith("pandas") for n in names),
                    f"{relative} imports pandas at module scope; that crashes any "
                    "process that already loaded torch",
                )

    def test_annotation_only_imports_are_deferred(self):
        """``pd.DataFrame`` must not be evaluated at import time.

        ``charts.py`` annotates ``data_distribution(df: pd.DataFrame, ...)``. Once
        the pandas import moved under ``TYPE_CHECKING`` those annotations raised
        ``NameError`` at module import unless ``from __future__ import
        annotations`` is present. Assert the guard directly rather than relying
        on the subprocess test to happen to import the module.
        """
        import ast

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for relative in ("utils/helpers.py", "visualization/charts.py"):
            with open(os.path.join(repo_root, relative), encoding="utf-8") as fh:
                source = fh.read()
            tree = ast.parse(source)
            if "pd." not in source:
                continue
            guarded = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                and any(a.name == "annotations" for a in node.names)
                for node in tree.body
            )
            # An alternatively valid strategy: every pd annotation is quoted.
            all_quoted = all(
                isinstance(node.annotation, ast.Constant)
                and isinstance(node.annotation.value, str)
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                for node in node.args.args
                if node.annotation is not None and "pd." in ast.unparse(node.annotation)
            )
            self.assertTrue(
                guarded or all_quoted,
                f"{relative} references pd.* in annotations without "
                "`from __future__ import annotations`; the annotation will be "
                "evaluated at import time and raise NameError.",
            )


@unittest.skipIf(torch is None, "torch not installed")
class TestReproducibility(unittest.TestCase):
    """``FLEngine.seed`` must make a run reproducible.

    Regression guard: the engine used to seed only the numpy data split, so
    model initialisation, ``DataLoader`` shuffling and dropout still came from
    the process-global torch RNG. Two identically-configured runs could differ
    by several accuracy points, which silently turned every comparison between
    two configurations into a coin flip.
    """

    def _trajectory(self, seed):
        X, y = separable_data()
        engine = FLEngine(input_dim=32, num_classes=3, hidden_dim=16, lr=0.01,
                          local_epochs=3, n_clients=3, non_iid=0.6, seed=seed)
        history = engine.run_sequential(X, y, gate="hard", rounds=3)
        return [round(h["val_acc"], 6) for h in history]

    def test_same_seed_gives_an_identical_trajectory(self):
        self.assertEqual(self._trajectory(7), self._trajectory(7))

    def test_run_and_run_sequential_both_honour_the_seed(self):
        X, y = separable_data()

        def go():
            engine = FLEngine(input_dim=32, num_classes=3, hidden_dim=16, lr=0.01,
                              local_epochs=2, seed=11)
            return [round(h["val_acc"], 6) for h in engine.run(X, y, n_clients=3, rounds=3)]

        self.assertEqual(go(), go())

    def test_torch_is_seeded_so_model_init_is_stable(self):
        import torch

        engine = FLEngine(input_dim=16, num_classes=3, hidden_dim=8, seed=123)
        engine._seed_torch()
        a = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        engine._seed_torch()
        b = OrganoidClassifier(input_dim=16, num_classes=3, hidden_dim=8)
        for (n1, p1), (_n2, p2) in zip(
            a.named_parameters(), b.named_parameters(), strict=True
        ):
            self.assertTrue(
                torch.equal(p1, p2),
                f"{n1} differs between two seeded initialisations",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
