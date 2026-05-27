"""Regression tests for explicit predecessor tracking of helper layers
inserted by ONNX converters (no FX node of their own).

History
=======
Two helper-insertion patterns triggered downstream failures before this
machinery existed:

  1. ``OnnxPow`` chained MULs (e.g. x**3 = Mul(Mul(x,x), x)).
     The chained MUL layers have no FX node. The FX-based pred walk
     fell back to ``preds[i] = [i-1]`` which is WRONG if an unrelated
     layer (e.g. an OnnxConstant for the exponent literal) sits between
     the MUL and its real upstream operand. Worse, ``tf_mul`` reads
     positional ``preds[L][0]`` and ``preds[L][1]`` — Mul(x,x) needs the
     source-layer id LISTED TWICE so both index lookups resolve.

  2. ``OnnxBinaryMathOperation`` (Div/Mul/Add/Sub) with scalar-broadcast
     (one operand has size 1, the other size N). nn4sys pensieve hits
     this in the L2-norm chain ``Div(x, ||x||)``. We splice an EXPAND
     helper to grow the size-1 operand to N before the binary op, but
     EXPAND has no FX node either, so the FX walk would silently miss
     the helper edge and the downstream consumer would receive
     wrong-size bounds.

Both patterns are now handled via ``_set_explicit_preds(layer_id, list)``
on the builder, which records an explicit pred list that the FX walk's
output is overwritten with. Duplicates are intentionally preserved so
positional indexing (Mul(x,x) -> [src, src]) works. The DAG cycle check
counts UNIQUE preds for in-degree so duplicates don't trigger a false
positive.

These tests pin all three invariants.
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

from act.back_end.layer_schema import LayerKind
from act.pipeline.verification.torch2act import _LayerGraphBuilder
from act.pipeline.verification.utils import _assert_dag


class _TorchDeviceIsolation(unittest.TestCase):
    """Mixin that snapshots torch's global default device/dtype in setUp
    and restores them in tearDown. Required because some tests in this
    file load real ONNX benchmark models through ``data_model_loader``,
    which routes through the ACT DeviceManager and may leave the global
    default device set to ``cuda:0``. Downstream tests (e.g.
    test_vnnlib_parser_soundness) that build small tensors with
    ``torch.tensor([...])`` then implicitly inherit the CUDA default,
    and ``np.asarray`` on a CUDA tensor raises a TypeError.

    Subclass this and ``super().setUp() / super().tearDown()`` to inherit
    the isolation discipline."""

    def setUp(self):
        super().setUp()
        # Capture current default device/dtype so any side effects of
        # loading a CUDA-resident model can be rolled back.
        self._saved_default_device = torch.get_default_device() \
            if hasattr(torch, "get_default_device") else None
        self._saved_default_dtype = torch.get_default_dtype()
        # Force CPU + float64 for the duration of the test; sub-tests
        # that need CUDA should opt in explicitly, not by accident.
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        try:
            if self._saved_default_device is not None:
                torch.set_default_device(self._saved_default_device)
            else:
                torch.set_default_device("cpu")
        except Exception:
            pass
        torch.set_default_dtype(self._saved_default_dtype)
        super().tearDown()


def _build_inner_layers(model: nn.Module, input_shape):
    """Run the inner builder up through preds/succs construction so the
    tests can inspect explicit_preds + the resulting pred list."""
    b = _LayerGraphBuilder(model, tuple(input_shape), torch.float64)
    n_inputs = 1
    for d in input_shape:
        n_inputs *= d
    b.prev_out = b._alloc_ids(n_inputs)
    b._extract_graph()
    b._pre_register_nodes()
    b._process_fx_graph()
    preds, succs = b._build_preds_succs()
    return b, preds, succs


class TestExplicitPredsContract(_TorchDeviceIsolation):
    def test_duplicates_preserved(self):
        """`_set_explicit_preds([56, 56])` must yield exactly two entries,
        not one. tf_mul / tf_div positional indexing depends on it."""
        b = _LayerGraphBuilder(nn.Identity(), (1, 2), torch.float64)
        b._set_explicit_preds(99, [56, 56])
        self.assertEqual(b._explicit_preds[99], [56, 56])

    def test_negatives_filtered(self):
        """-1 means "placeholder predecessor" and is filtered out (the
        wrapper's INPUT_SPEC-connect-all handles the placeholder edge)."""
        b = _LayerGraphBuilder(nn.Identity(), (1, 2), torch.float64)
        b._set_explicit_preds(99, [-1, 5, -1])
        self.assertEqual(b._explicit_preds[99], [5])

    def test_self_id_filtered(self):
        """A layer must not list itself as its own predecessor."""
        b = _LayerGraphBuilder(nn.Identity(), (1, 2), torch.float64)
        b._set_explicit_preds(99, [99, 50])
        self.assertEqual(b._explicit_preds[99], [50])

    def test_var_to_producer_layer_tracked(self):
        """Every ``_add_layer`` call must register its out_vars in the
        producer map. Without this the helper-emitting handlers can't
        ask 'who feeds var v?'."""
        b = _LayerGraphBuilder(nn.Identity(), (1, 2), torch.float64)
        b._add_layer("RELU", {}, [0, 1], [2, 3])
        self.assertEqual(b._var_to_producer_layer.get(2), 0)
        self.assertEqual(b._var_to_producer_layer.get(3), 0)
        b._add_layer("RELU", {}, [2, 3], [4, 5])
        self.assertEqual(b._var_to_producer_layer.get(4), 1)


class TestAssertDagAllowsPredDuplicates(_TorchDeviceIsolation):
    """``_assert_dag`` counts UNIQUE preds for in-degree so a Mul(x, x)
    layer with preds=[56, 56] passes through Kahn's algorithm. Without
    this, in_degree=2 vs succs entry of 1 falsely reports a cycle."""

    def test_duplicate_preds_pass_dag_check(self):
        # 3 layers: 0 -> 1 (twice via duplicates) -> 2
        preds = {0: [], 1: [0, 0], 2: [1]}
        succs = {0: [1], 1: [2], 2: []}
        try:
            _assert_dag(preds, succs, n_layers=3)
        except ValueError as e:
            self.fail(f"_assert_dag falsely rejected duplicate-preds DAG: {e}")

    def test_real_cycle_still_detected(self):
        """The fix must NOT silence a real cycle. 0 <-> 1 (mutual
        predecessors) must still be rejected."""
        preds = {0: [1], 1: [0]}
        succs = {0: [1], 1: [0]}
        with self.assertRaisesRegex(ValueError, "contains a cycle"):
            _assert_dag(preds, succs, n_layers=2)


class TestOnnxPowChainPreds(_TorchDeviceIsolation):
    """The OnnxPow handler emits a chain of (k-1) MULs for ``x**k``.
    Each chained MUL is a helper (no FX node). The handler must register
    explicit preds so:
      * step 0 (Mul(x, x)): preds = [x_src, x_src] (duplicate)
      * step k (Mul(accumulator, x)): preds = [prev_mul_id, x_src]
    """

    def _build_pow_model(self, exponent: int):
        """A model that is just `x ** exponent` so the FX trace contains
        exactly one OnnxPow node (after onnx2torch conversion)."""
        # We can't easily synthesize an OnnxPow directly without going
        # through ONNX, so we exercise the handler invariant at a level
        # below: assert that `_set_explicit_preds` was called for every
        # chained MUL with a 2-element list. Easiest way is to feed a
        # tiny ONNX model through the real loader.
        return exponent  # marker

    def test_explicit_preds_registered_for_pow_chain(self):
        """Use the real nn4sys pensieve_small_parallel which contains
        x**3 in its forward pass. After conversion, every helper MUL in
        the Pow chain must have an explicit preds entry of length 2."""
        from act.front_end.vnnlib_loader.data_model_loader import load_vnnlib_pair
        try:
            pair = load_vnnlib_pair(
                category="nn4sys",
                onnx_model="onnx/pensieve_small_parallel.onnx",
                vnnlib_spec="vnnlib/pensieve_parallel_4.vnnlib",
                root_dir="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks",
                auto_download=False,
            )
        except (FileNotFoundError, Exception) as e:
            self.skipTest(f"nn4sys pensieve_small_parallel not present: {e}")
        m = pair["model"].to(torch.float64)
        x = pair["labeled_tensor"].tensor.to(torch.float64)
        b, preds, _ = _build_inner_layers(m, x.shape)
        # Find all MUL layers that were NOT FX-tracked (helpers).
        mul_helpers = [
            L.id for L in b.layers
            if L.kind == LayerKind.MUL.value and L.id in b._explicit_preds
        ]
        self.assertGreater(
            len(mul_helpers), 0,
            "expected at least one helper MUL in pensieve_small_parallel Pow chain"
        )
        for lid in mul_helpers:
            self.assertEqual(
                len(b._explicit_preds[lid]), 2,
                f"helper MUL id={lid} must have exactly 2 preds entries (positional); "
                f"got {b._explicit_preds[lid]!r}"
            )
            # And the post-override preds list also has length 2.
            self.assertEqual(
                len(preds[lid]), 2,
                f"helper MUL id={lid} post-override preds len mismatch: {preds[lid]!r}"
            )


class TestOnnxBinaryBroadcastExpandPreds(_TorchDeviceIsolation):
    """When ``_convert_OnnxBinaryMathOperation`` splices an EXPAND for
    scalar-broadcast, the downstream binary op's explicit preds must
    point at the EXPAND (not the original size-1 source). Otherwise
    tf_div/tf_mul gets the un-broadcast bounds and either errors or
    silently propagates a wrong-size Bin."""

    def test_pensieve_small_parallel_has_expand_consumed_by_div(self):
        from act.front_end.vnnlib_loader.data_model_loader import load_vnnlib_pair
        try:
            pair = load_vnnlib_pair(
                category="nn4sys",
                onnx_model="onnx/pensieve_small_parallel.onnx",
                vnnlib_spec="vnnlib/pensieve_parallel_4.vnnlib",
                root_dir="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks",
                auto_download=False,
            )
        except (FileNotFoundError, Exception) as e:
            self.skipTest(f"nn4sys pensieve_small_parallel not present: {e}")
        m = pair["model"].to(torch.float64)
        x = pair["labeled_tensor"].tensor.to(torch.float64)
        b, preds, succs = _build_inner_layers(m, x.shape)

        expand_layers = [L for L in b.layers if L.kind == LayerKind.EXPAND.value]
        div_layers = [L for L in b.layers if L.kind == LayerKind.DIV.value]
        if not expand_layers or not div_layers:
            self.skipTest("model variant did not exercise broadcast-EXPAND path")

        # At least one EXPAND should feed at least one DIV — i.e. there
        # exists a DIV whose preds list includes an EXPAND layer id.
        expand_ids = {L.id for L in expand_layers}
        consumer_links = [
            (div.id, [p for p in preds.get(div.id, []) if p in expand_ids])
            for div in div_layers
        ]
        any_linked = any(linked for _, linked in consumer_links)
        self.assertTrue(
            any_linked,
            f"no DIV layer is wired to an EXPAND helper; "
            f"DIV preds: {consumer_links}; expand ids: {sorted(expand_ids)}"
        )


class TestVarConstBroadcastExpandPreds(_TorchDeviceIsolation):
    """ml4acopf creates var-constant outer broadcasts such as
    ``(1, 24, 1) + (54,) -> (1, 24, 54)``. The converter inserts an
    EXPAND for the variable operand; downstream BIAS must consume that
    EXPAND rather than the pre-broadcast UNSQUEEZE."""

    def test_ml4acopf_large_bias_consumes_expand_helper(self):
        from act.front_end.vnnlib_loader.data_model_loader import load_vnnlib_pair
        try:
            pair = load_vnnlib_pair(
                category="ml4acopf_2024",
                onnx_model="./onnx/14_ieee_ml4acopf-linear-residual.onnx",
                vnnlib_spec="./vnnlib/14_ieee_prop9.vnnlib",
                root_dir="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks",
                auto_download=False,
            )
        except Exception as e:
            self.skipTest(f"ml4acopf benchmark not present: {e}")
        model = pair["model"].to(torch.float64)
        x = pair["labeled_tensor"].tensor.to(torch.float64)
        builder, preds, _ = _build_inner_layers(model, x.shape)
        by_id = {layer.id: layer for layer in builder.layers}

        broadcast_biases = [
            layer for layer in builder.layers
            if layer.kind == LayerKind.BIAS.value and len(layer.out_vars) > 100
        ]
        self.assertGreater(len(broadcast_biases), 0)
        for bias in broadcast_biases:
            bias_preds = preds.get(bias.id, [])
            self.assertEqual(
                len(bias_preds), 1,
                f"broadcast BIAS id={bias.id} expected one EXPAND pred; got {bias_preds}",
            )
            pred = by_id[bias_preds[0]]
            self.assertEqual(
                pred.kind, LayerKind.EXPAND.value,
                f"broadcast BIAS id={bias.id} bypasses EXPAND via pred "
                f"id={pred.id} kind={pred.kind}",
            )
            self.assertEqual(len(pred.out_vars), len(bias.out_vars))


if __name__ == "__main__":
    unittest.main(verbosity=2)
