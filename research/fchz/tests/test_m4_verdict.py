"""Multi-query verdict semantics tests (advisor 2026-06-08).

Critical:
  - all queries must hold (AND) for CERT
  - first-query shortcut is FORBIDDEN
  - unsupported kind in any query → UNSUPPORTED_QUERY
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import unittest
import numpy as np
import torch
import torch.nn as nn


def build_toy(W_list, b_list, lb, ub, input_dim, output_dim):
    layers = []
    for i, (W, b) in enumerate(zip(W_list[:-1], b_list[:-1])):
        lin = nn.Linear(W.shape[1], W.shape[0], dtype=torch.float64)
        lin.weight.data = torch.tensor(W, dtype=torch.float64)
        lin.bias.data = torch.tensor(b, dtype=torch.float64)
        layers.append(lin); layers.append(nn.ReLU())
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
    from act.back_end.transfer_functions import set_transfer_function_mode, get_transfer_function
    from act.back_end.core import Bounds
    import os
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
    os.environ['HYZOR_FCHZ_USE_CUDA'] = '0'
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
    pre_assert = net.preds.get(next(L for L in reversed(net.layers) if L.kind == 'ASSERT').id, [None])[0]
    state = tf._state_cache.get(pre_assert)
    return net, tf, state


class TestMultiQuery(unittest.TestCase):
    """All-query AND semantics."""

    def _build_unsafe_linear_query(self, C, d):
        """Build a fake (InputSpec, OutputSpec) tuple with UNSAFE_LINEAR kind."""
        from act.front_end.specs import OutputSpec, OutKind, InputSpec, InKind
        in_spec = InputSpec(kind=InKind.BOX,
                                  lb=torch.zeros(1, 2), ub=torch.ones(1, 2))
        out_spec = OutputSpec(kind=OutKind.UNSAFE_LINEAR,
                                    c=torch.tensor(C, dtype=torch.float32),
                                    d=torch.tensor(d, dtype=torch.float32))
        return (in_spec, out_spec)

    def test_two_queries_both_safe_should_cert(self):
        """Trivial 2-query: 2 separately-safe queries → CERT."""
        from research.fchz.m4_verdict import m4_verdict_for_queries
        W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        b1 = np.zeros(2)
        W_out = np.eye(2)
        b_out = np.zeros(2)
        lb = np.array([2.0, 2.0]); ub = np.array([3.0, 3.0])
        # Output in [2,3]^2.
        # Query 0 UNSAFE: y_0 <= 0 (impossible) → safe (LB > t)
        # Query 1 UNSAFE: y_1 <= 0 → safe
        net, tf, state = build_toy([W1, W_out], [b1, b_out], lb, ub, 2, 2)
        q0 = self._build_unsafe_linear_query([[1.0, 0.0]], [0.0])
        q1 = self._build_unsafe_linear_query([[0.0, 1.0]], [0.0])
        result = m4_verdict_for_queries(net, tf, [q0, q1], n_out=2, cf_state=state)
        self.assertEqual(result['verdict'], 'CERTIFIED',
                                 f"both safe should cert, got {result}")

    def test_two_queries_first_safe_second_unsafe_should_be_unknown(self):
        """First query safe, second NOT safe → overall UNKNOWN (no shortcut)."""
        from research.fchz.m4_verdict import m4_verdict_for_queries
        W1 = np.eye(2); b1 = np.zeros(2)
        W_out = np.eye(2); b_out = np.zeros(2)
        lb = np.array([2.0, 2.0]); ub = np.array([3.0, 3.0])
        net, tf, state = build_toy([W1, W_out], [b1, b_out], lb, ub, 2, 2)
        q0 = self._build_unsafe_linear_query([[1.0, 0.0]], [0.0])   # SAFE
        q1 = self._build_unsafe_linear_query([[1.0, 0.0]], [5.0])   # UNSAFE (y_0 <= 5)
        result = m4_verdict_for_queries(net, tf, [q0, q1], n_out=2, cf_state=state)
        self.assertEqual(result['verdict'], 'UNKNOWN',
                                 f"second query not safe → UNK, got {result}")

    def test_unsafe_first_safe_second_should_be_unknown(self):
        """Symmetric: first unsafe, second safe → UNK (no shortcut to second's safety)."""
        from research.fchz.m4_verdict import m4_verdict_for_queries
        W1 = np.eye(2); b1 = np.zeros(2)
        W_out = np.eye(2); b_out = np.zeros(2)
        lb = np.array([2.0, 2.0]); ub = np.array([3.0, 3.0])
        net, tf, state = build_toy([W1, W_out], [b1, b_out], lb, ub, 2, 2)
        q0 = self._build_unsafe_linear_query([[1.0, 0.0]], [5.0])   # UNSAFE
        q1 = self._build_unsafe_linear_query([[1.0, 0.0]], [0.0])   # SAFE
        result = m4_verdict_for_queries(net, tf, [q0, q1], n_out=2, cf_state=state)
        # MUST be UNK — must not "succeed early" or shortcut to second query.
        self.assertEqual(result['verdict'], 'UNKNOWN',
                                 f"first query unsafe → UNK regardless of others, got {result}")

    def test_unsupported_kind_marks_unsupported(self):
        """Any unsupported kind in any query → UNSUPPORTED_QUERY (not silent UNK)."""
        from research.fchz.m4_verdict import m4_verdict_for_queries
        from act.front_end.specs import OutputSpec, OutKind, InputSpec, InKind
        W1 = np.eye(2); b1 = np.zeros(2)
        W_out = np.eye(2); b_out = np.zeros(2)
        # Output range [2, 3]^2 so first query (y_0 <= 0) is provably safe
        net, tf, state = build_toy([W1, W_out], [b1, b_out],
                                              np.array([2.0, 2.0]), np.array([3.0, 3.0]), 2, 2)
        q0 = self._build_unsafe_linear_query([[1.0, 0.0]], [0.0])   # SAFE (y_0 in [2,3], can't be <=0)
        # Build a query with a fake unsupported kind by patching object after construction
        in_spec_fake = InputSpec(kind=InKind.BOX, lb=torch.zeros(1, 2), ub=torch.ones(1, 2))
        from act.front_end.specs import OutputSpec
        # construct LINEAR_LE then mutate kind to a string that is not supported
        out_spec_fake = OutputSpec(kind=OutKind.LINEAR_LE,
                                                c=torch.zeros(1, 2),
                                                d=torch.tensor([0.0]))
        # mutate kind in-place to simulate unsupported kind
        object.__setattr__(out_spec_fake, 'kind', 'EXOTIC_UNSUPPORTED_KIND')
        result = m4_verdict_for_queries(net, tf, [q0, (in_spec_fake, out_spec_fake)],
                                                          n_out=2, cf_state=state)
        # Should mark UNSUPPORTED_QUERY (no false CERT)
        self.assertEqual(result['verdict'], 'UNSUPPORTED_QUERY')


if __name__ == '__main__':
    unittest.main()
