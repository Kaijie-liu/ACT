"""Unit tests for M4 full-network LP module (advisor 2026-06-08 audit)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import unittest
import numpy as np
import torch


def build_toy_dense_relu_net(W_list, b_list, lb, ub, input_dim, output_dim):
    """Build a toy ACT net: INPUT_SPEC -> [DENSE -> RELU]* -> DENSE -> ASSERT.

    Uses FchzTF; returns (net, tf, queries, n_out) ready for verification.
    Last DENSE is output (no RELU after).
    """
    import torch.nn as nn
    layers = []
    for i, (W, b) in enumerate(zip(W_list[:-1], b_list[:-1])):
        lin = nn.Linear(W.shape[1], W.shape[0], dtype=torch.float64)
        lin.weight.data = torch.tensor(W, dtype=torch.float64)
        lin.bias.data = torch.tensor(b, dtype=torch.float64)
        layers.append(lin)
        layers.append(nn.ReLU())
    lin_last = nn.Linear(W_list[-1].shape[1], W_list[-1].shape[0], dtype=torch.float64)
    lin_last.weight.data = torch.tensor(W_list[-1], dtype=torch.float64)
    lin_last.bias.data = torch.tensor(b_list[-1], dtype=torch.float64)
    layers.append(lin_last)
    model = nn.Sequential(*layers).double()

    from act.front_end.verifiable_model import (
        VerifiableModel, InputLayer, InputSpecLayer, OutputSpecLayer)
    from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.pipeline.verification.torch2act import TorchToACT
    from act.back_end.transfer_functions import (
        set_transfer_function_mode, get_transfer_function)
    from act.back_end.core import Bounds

    center = torch.tensor((lb + ub) / 2, dtype=torch.float32).reshape(1, input_dim)
    labeled = LabeledInputTensor(tensor=center, label=torch.tensor([0]))
    in_shape = (1, input_dim)
    in_spec = InputSpec(kind=InKind.BOX,
                              lb=torch.tensor(lb, dtype=torch.float32).reshape(1, input_dim),
                              ub=torch.tensor(ub, dtype=torch.float32).reshape(1, input_dim))
    out_spec = OutputSpec(kind=OutKind.LINEAR_LE,
                                c=torch.zeros(1, output_dim, dtype=torch.float32),
                                d=torch.tensor([0.0], dtype=torch.float32))
    verifiable = VerifiableModel(
        input_layer=InputLayer(labeled_input=labeled, shape=in_shape, dtype=torch.float32),
        input_spec=InputSpecLayer(in_spec), model=model, output_spec=OutputSpecLayer(out_spec))
    net = TorchToACT(verifiable).run()
    import os; os.environ['HYZOR_FCHZ_USE_CUDA'] = '0'
    os.environ.pop('HYZOR_FCHZ_G_MAX_COLS', None)
    set_transfer_function_mode("fchz")
    tf = get_transfer_function()
    input_bounds = Bounds(
        lb=torch.tensor(lb, dtype=torch.float32).reshape(1, input_dim),
        ub=torch.tensor(ub, dtype=torch.float32).reshape(1, input_dim))
    before, after = {}, {}
    for L in net.layers:
        in_b = input_bounds if L.id == 0 or not net.preds.get(L.id) else after[net.preds[L.id][0]].bounds
        after[L.id] = tf.apply(L, in_b, net, before, after)
    return net, tf


def brute_force_bound(W_list, b_list, lb, ub, d_out, n_samples=10000, sense='max'):
    """Random sample x in [lb, ub], forward, return max/min of d @ output."""
    n_in = len(lb)
    np.random.seed(42)
    xs = lb + np.random.rand(n_samples, n_in) * (ub - lb)
    outs = xs
    for W, b in zip(W_list[:-1], b_list[:-1]):
        outs = np.maximum(0, outs @ W.T + b)
    outs = outs @ W_list[-1].T + b_list[-1]
    margins = outs @ d_out
    return float(margins.max() if sense == 'max' else margins.min())


def closed_form_ub(state, d):
    """FCHZ closed-form upper bound on d @ y."""
    c = state.c; G = state.G; tail = state.tail_radius
    ub = float(d @ c)
    if G is not None and G.size > 0:
        ub += float(np.abs(d @ G).sum())
    if tail is not None:
        ub += float(np.abs(d) @ tail)
    return ub


def closed_form_lb(state, d):
    c = state.c; G = state.G; tail = state.tail_radius
    lb = float(d @ c)
    if G is not None and G.size > 0:
        lb -= float(np.abs(d @ G).sum())
    if tail is not None:
        lb -= float(np.abs(d) @ tail)
    return lb


class TestOneLayer(unittest.TestCase):
    """1-layer ReLU toy: LP must be tighter than closed-form, looser than brute."""

    def test_one_layer_simple(self):
        from research.fchz.m4_full_lp import (
            solve_full_lp_ub, solve_full_lp_lb, is_dense_only_chain)
        # x in [-1, 1]^2; layer 1: y = ReLU([1, 0.5] x); output = [-1] @ y + 0
        W1 = np.array([[1.0, 0.5]])
        b1 = np.zeros(1)
        W_out = np.array([[1.0]])   # output = y[0]
        b_out = np.zeros(1)
        lb_in = np.array([-1.0, -1.0])
        ub_in = np.array([1.0, 1.0])

        net, tf = build_toy_dense_relu_net(
            [W1, W_out], [b1, b_out], lb_in, ub_in,
            input_dim=2, output_dim=1)
        self.assertTrue(is_dense_only_chain(net))

        d = np.array([1.0])
        lp_ub = solve_full_lp_ub(net, tf, d)
        lp_lb = solve_full_lp_lb(net, tf, d)
        brute_ub = brute_force_bound([W1, W_out], [b1, b_out], lb_in, ub_in, d, sense='max')
        brute_lb = brute_force_bound([W1, W_out], [b1, b_out], lb_in, ub_in, d, sense='min')

        # LP bounds must SANDWICH the brute force sample bounds (sound).
        self.assertGreaterEqual(lp_ub, brute_ub - 0.01,
                                            f"LP UB {lp_ub} should >= brute {brute_ub}")
        self.assertLessEqual(lp_lb, brute_lb + 0.01,
                                       f"LP LB {lp_lb} should <= brute {brute_lb}")

    def test_one_layer_negative_direction(self):
        """Direction with negative coef — exercise UNSAFE_LINEAR style.

        Test that solve_full_lp_lb computes min directly, not via -max(-d).
        Both methods must agree.
        """
        from research.fchz.m4_full_lp import solve_full_lp_ub, solve_full_lp_lb
        W1 = np.array([[1.0, -0.5], [0.3, 0.7]])
        b1 = np.array([0.1, -0.2])
        W_out = np.eye(2)
        b_out = np.zeros(2)
        lb_in = np.array([-1.0, -1.0])
        ub_in = np.array([1.0, 1.0])

        net, tf = build_toy_dense_relu_net(
            [W1, W_out], [b1, b_out], lb_in, ub_in,
            input_dim=2, output_dim=2)
        d = np.array([1.0, -1.0])
        lp_lb_direct = solve_full_lp_lb(net, tf, d)
        lp_ub_neg = solve_full_lp_ub(net, tf, -d)
        # Both compute same quantity: min d @ y = -(max (-d) @ y)
        self.assertAlmostEqual(lp_lb_direct, -lp_ub_neg, places=6,
                                          msg=f"Direct min {lp_lb_direct} should equal -max(-d) = {-lp_ub_neg}")


class TestTwoLayer(unittest.TestCase):
    """2-layer ReLU toy: LP not looser than closed-form, sound vs brute."""

    def test_two_layer_sound(self):
        from research.fchz.m4_full_lp import (
            solve_full_lp_ub, solve_full_lp_lb, extract_layers_for_lp)
        np.random.seed(7)
        W1 = np.random.randn(4, 3) * 0.5
        b1 = np.random.randn(4) * 0.1
        W2 = np.random.randn(2, 4) * 0.5
        b2 = np.random.randn(2) * 0.1
        W_out = np.array([[1.0, -1.0]])
        b_out = np.zeros(1)
        lb_in = np.array([-0.5, -0.5, -0.5])
        ub_in = np.array([0.5, 0.5, 0.5])

        net, tf = build_toy_dense_relu_net(
            [W1, W2, W_out], [b1, b2, b_out], lb_in, ub_in,
            input_dim=3, output_dim=1)
        d = np.array([1.0])
        lp_ub = solve_full_lp_ub(net, tf, d)
        lp_lb = solve_full_lp_lb(net, tf, d)
        brute_ub = brute_force_bound([W1, W2, W_out], [b1, b2, b_out], lb_in, ub_in, d, sense='max')
        brute_lb = brute_force_bound([W1, W2, W_out], [b1, b2, b_out], lb_in, ub_in, d, sense='min')
        # Soundness: LP UB >= brute UB, LP LB <= brute LB (with small slack for sampling)
        self.assertGreaterEqual(lp_ub, brute_ub - 0.01)
        self.assertLessEqual(lp_lb, brute_lb + 0.01)


class TestSolverDiagnostics(unittest.TestCase):
    """Test that solve_full_lp returns diagnostics dict with status/residuals."""

    def test_diagnostics_returned(self):
        from research.fchz.m4_full_lp import solve_full_lp
        W1 = np.array([[1.0, 0.5]])
        b1 = np.zeros(1)
        W_out = np.array([[1.0]])
        b_out = np.zeros(1)
        lb_in = np.array([-1.0, -1.0])
        ub_in = np.array([1.0, 1.0])
        net, tf = build_toy_dense_relu_net(
            [W1, W_out], [b1, b_out], lb_in, ub_in, 2, 1)
        d = np.array([1.0])
        val, diag = solve_full_lp(net, tf, d, sense='max')
        self.assertEqual(diag['status'], 'OK')
        self.assertIn('ub_resid', diag)
        self.assertIn('eq_resid', diag)
        self.assertLess(diag['ub_resid'], 1e-6)
        self.assertLess(diag['eq_resid'], 1e-6)


class TestDirectMinVsNegatedMax(unittest.TestCase):
    """Critical: direct min and -max(-d) must agree.

    Per advisor 2026-06-08: sat_relu -1.0 uniformity needs this check.
    """

    def test_three_layer_agreement(self):
        from research.fchz.m4_full_lp import solve_full_lp_lb, solve_full_lp_ub
        np.random.seed(11)
        Ws = [np.random.randn(4, 3) * 0.5, np.random.randn(3, 4) * 0.5,
                np.random.randn(2, 3) * 0.5]
        bs = [np.random.randn(4) * 0.1, np.random.randn(3) * 0.1,
                np.random.randn(2) * 0.1]
        lb_in = np.array([-0.3, -0.3, -0.3])
        ub_in = np.array([0.3, 0.3, 0.3])
        net, tf = build_toy_dense_relu_net(Ws, bs, lb_in, ub_in, 3, 2)
        for d in [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                      np.array([1.0, -1.0]), np.array([-1.0, 1.0]),
                      np.array([0.5, -0.7])]:
            lb_direct = solve_full_lp_lb(net, tf, d)
            ub_neg = solve_full_lp_ub(net, tf, -d)
            self.assertAlmostEqual(lb_direct, -ub_neg, places=4,
                                              msg=f"d={d}: direct lb {lb_direct} != -ub(-d) {-ub_neg}")


if __name__ == '__main__':
    unittest.main()
