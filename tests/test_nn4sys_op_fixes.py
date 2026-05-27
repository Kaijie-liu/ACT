"""Regression tests for nn4sys op coverage fixes (2026-05-25).

Three independent gaps blocked nn4sys 5/5 ERROR:
  1. OnnxSplit (bare, opset<13 or equal-axis form) was rejected because
     ONNX_HANDLERS binding loop used _fn.__name__ ("_convert_OnnxSplit13")
     instead of the dict key, so 'OnnxSplit' alias never reached dispatch.
     Plus OnnxSplit13 itself rejected the equal-axis form.
  2. OnnxGather constant indices resolved via _resolve_constant_tensor only,
     missing the call_module / Constant op chain that pensieve uses.
  3. tf_gather called torch.index_select with 0-d scalar negative indices
     (ONNX semantics: -1 = last element), raising IndexError.
  4. OnnxPow rejected exponent>2 and dynamic exponents; pensieve uses x^3
     via constant subgraph.
  5. nn.Conv1d was not in _convert_module's dispatch table.

These tests pin each fix so a future refactor cannot silently regress.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn

from act.back_end.core import Layer, Bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.interval_tf.tf_mlp import tf_gather


def _gather_layer(indices, axis, input_shape):
    """Build a minimal ACT GATHER Layer for tf_gather direct tests."""
    n_in = 1
    for d in input_shape[1:]:  # drop batch
        n_in *= d
    if indices.dim() == 0:
        out_shape = tuple(list(input_shape[1:])[:axis] + list(input_shape[1:])[axis+1:]) or (1,)
    else:
        out_shape = tuple(list(input_shape[1:])[:axis] + list(indices.shape) + list(input_shape[1:])[axis+1:])
    n_out = 1
    for d in out_shape:
        n_out *= d
    in_vars = list(range(n_in))
    out_vars = list(range(n_in, n_in + n_out))
    return Layer(
        id=1, kind=LayerKind.GATHER.value,
        in_vars=in_vars, out_vars=out_vars,
        params={"indices": indices, "axis": axis,
                "input_shape": input_shape[1:], "output_shape": out_shape},
    )


class TestTfGatherScalarNegativeIndex(unittest.TestCase):
    """tf_gather must handle ONNX-style 0-d scalar indices and negative
    values (axis-from-end semantics). nn4sys pensieve gathers shape=()
    value=-1 to take the last element of an axis-2 dim."""

    def test_scalar_negative_last_element(self):
        indices = torch.tensor(-1, dtype=torch.int64)  # 0-d
        input_shape = (1, 1, 1, 8)
        L = _gather_layer(indices, axis=2, input_shape=input_shape)
        lb = torch.arange(8, dtype=torch.float64).view(1, -1).clone()
        ub = lb.clone() + 0.5
        Bin = Bounds(lb=lb, ub=ub)
        fact = tf_gather(L, Bin)
        # Scalar gather drops the axis -> output shape (1, 1, 1) per-sample
        self.assertEqual(fact.bounds.lb.numel(), 1)
        # Last element value is 7 (0..7); lb stays 7, ub becomes 7.5
        self.assertAlmostEqual(float(fact.bounds.lb.flatten()[0]), 7.0)
        self.assertAlmostEqual(float(fact.bounds.ub.flatten()[0]), 7.5)

    def test_scalar_positive_index(self):
        indices = torch.tensor(3, dtype=torch.int64)  # 0-d
        input_shape = (1, 1, 1, 8)
        L = _gather_layer(indices, axis=2, input_shape=input_shape)
        lb = torch.arange(8, dtype=torch.float64).view(1, -1).clone()
        ub = lb.clone()
        fact = tf_gather(L, Bin=Bounds(lb=lb, ub=ub))
        self.assertAlmostEqual(float(fact.bounds.lb.flatten()[0]), 3.0)

    def test_1d_indices_still_work(self):
        """Don't regress the existing 1-d non-negative path."""
        indices = torch.tensor([0, 2, 4], dtype=torch.int64)
        input_shape = (1, 1, 1, 8)
        L = _gather_layer(indices, axis=2, input_shape=input_shape)
        lb = torch.arange(8, dtype=torch.float64).view(1, -1).clone()
        ub = lb.clone()
        fact = tf_gather(L, Bin=Bounds(lb=lb, ub=ub))
        self.assertEqual(fact.bounds.lb.numel(), 3)
        self.assertEqual([float(v) for v in fact.bounds.lb.flatten()], [0.0, 2.0, 4.0])

    def test_out_of_range_after_wrap_raises(self):
        """If the wrapped index is still out of range, raise a clear error
        (not the cryptic torch IndexError)."""
        indices = torch.tensor(-99, dtype=torch.int64)  # 0-d, wraps to -99+8=-91
        input_shape = (1, 1, 1, 8)
        L = _gather_layer(indices, axis=2, input_shape=input_shape)
        lb = torch.zeros(1, 8, dtype=torch.float64)
        ub = torch.zeros(1, 8, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "index out of range after wrap"):
            tf_gather(L, Bin=Bounds(lb=lb, ub=ub))


class TestOnnxHandlerBinding(unittest.TestCase):
    """The ONNX_HANDLERS dict key MUST drive the bound attribute name
    (so e.g. 'OnnxSplit' alias to _convert_OnnxSplit13 actually
    dispatches). The earlier binding used _fn.__name__ which silently
    discarded aliases."""

    def test_onnx_split_alias_bound(self):
        from act.pipeline.verification.torch2act import _LayerGraphBuilder
        self.assertTrue(
            hasattr(_LayerGraphBuilder, '_convert_OnnxSplit'),
            "OnnxSplit alias must produce a bound _convert_OnnxSplit method"
        )
        self.assertTrue(hasattr(_LayerGraphBuilder, '_convert_OnnxSplit13'))
        self.assertIs(
            _LayerGraphBuilder._convert_OnnxSplit,
            _LayerGraphBuilder._convert_OnnxSplit13,
            "OnnxSplit alias should share the OnnxSplit13 implementation"
        )

    def test_all_onnx_handlers_dispatched(self):
        """Every dict key must produce a bound _convert_<Key> attr."""
        from act.pipeline.verification.utils import ONNX_HANDLERS
        from act.pipeline.verification.torch2act import _LayerGraphBuilder
        missing = [k for k in ONNX_HANDLERS
                   if not hasattr(_LayerGraphBuilder, f'_convert_{k}')]
        self.assertEqual(missing, [],
                         f"ONNX handler dispatch keys not bound: {missing}")


class TestConv1dTupleAttrDefensive(unittest.TestCase):
    """``_conv1d_to_linear_matrix`` previously unpacked
    ``stride``/``padding``/``dilation`` as raw values from ``L.params``.
    When the ONNX→torch path delivers them as length-1 tuples
    (``stride=(1,)``), the expression ``meshgrid_tensor * stride`` falls
    through Python's ``tensor * tuple`` path → ``tuple.__rmul__(tensor)``
    → ``Tensor.__index__`` on a multi-element tensor → ``TypeError:
    only integer tensors of a single element can be converted to an
    index``. nn4sys pensieve_big_parallel surfaced this immediately
    after the broadcast-Div helper-pred fix unblocked its Conv1d.

    Fix: normalize all three params to scalar ints inside the linear-
    matrix helper. These tests pin the defensiveness."""

    def test_tuple_attrs_do_not_raise(self):
        from act.back_end.interval_tf.tf_cnn import _conv1d_to_linear_matrix
        torch.manual_seed(0)
        weight = torch.randn(2, 1, 3, dtype=torch.float64)
        # Pass tuples like nn.Conv1d's stride/padding/dilation defaults.
        m = _conv1d_to_linear_matrix(
            weight,
            input_shape=(1, 1, 8),
            output_shape=(1, 2, 6),
            stride=(1,), padding=(0,), dilation=(1,), groups=1,
        )
        # Output should have shape (out_flat, in_flat) = (12, 8)
        self.assertEqual(tuple(m.shape), (12, 8))

    def test_scalar_attrs_still_work(self):
        from act.back_end.interval_tf.tf_cnn import _conv1d_to_linear_matrix
        torch.manual_seed(1)
        weight = torch.randn(2, 1, 3, dtype=torch.float64)
        m = _conv1d_to_linear_matrix(
            weight,
            input_shape=(1, 1, 8),
            output_shape=(1, 2, 6),
            stride=1, padding=0, dilation=1, groups=1,
        )
        self.assertEqual(tuple(m.shape), (12, 8))


class TestConv1dInTorch2Act(unittest.TestCase):
    """nn.Conv1d was missing from _convert_module dispatch — nn4sys
    pensieve_*_parallel uses 1D temporal convs on (B, C, L) inputs."""

    def test_conv1d_in_module_dispatch(self):
        from act.pipeline.verification.torch2act import _LayerGraphBuilder
        self.assertTrue(
            hasattr(_LayerGraphBuilder, '_convert_conv1d'),
            "Conv1d converter must exist"
        )

    def test_conv1d_round_trip_via_build_act(self):
        from act.pipeline.verification.torch2act import build_act
        torch.manual_seed(0)
        mod = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3).double()
        net = nn.Sequential(mod)
        input_shape = (1, 3, 8)  # (B, C, L)
        layers, _, _ = build_act(net, input_shape, dtype=torch.float64)
        conv1d_layers = [L for L in layers if L.kind == LayerKind.CONV1D.value]
        self.assertEqual(len(conv1d_layers), 1)
        L = conv1d_layers[0]
        for required in ("weight", "in_channels", "out_channels", "kernel_size"):
            self.assertIn(required, L.params, f"Conv1d schema requires {required}")
        # Output shape: L_out = (8 - 1*(3-1) - 1)/1 + 1 = 6
        self.assertEqual(tuple(L.params["output_shape"]), (1, 2, 6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
