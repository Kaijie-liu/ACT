"""Schema + round-trip regression test for CONVTRANSPOSE2D conversion.

History: cgan_2023 5-instance smoke (2026-05-25) failed 5/5 with
``Layer(kind=CONVTRANSPOSE2D) schema violation: Missing required PARAMS:
['in_channels', 'out_channels', 'kernel_size']``. Root cause:
torch2act._convert_conv_transpose2d populated weight/stride/padding/etc.
but NOT the three constructor-arity params that REGISTRY[CONVTRANSPOSE2D]
requires (parallel CONV2D path populated them correctly). A secondary
issue was act2torch._build_from_schema's kwarg whitelist not including
output_padding, so round-trip reconstruction would silently drop it
and shape-shift outputs on asymmetric models (e.g. cGAN_*_padding_1).

These tests pin both fixes so a future refactor cannot silently regress
to the broken schema or drop output_padding.
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

from act.back_end.layer_schema import LayerKind, REGISTRY
from act.back_end.layer_util import validate_layer
from act.pipeline.verification.torch2act import build_act
from act.pipeline.verification.act2torch import ACTToTorch


def _rebuild_layer(act_layer):
    """Invoke ACTToTorch._build_from_schema without paying the Net-wrapper
    validation cost. The method only touches REGISTRY / _ACT_TO_TORCH /
    act_layer.params, never self.act_net, so __new__ is safe here."""
    builder = ACTToTorch.__new__(ACTToTorch)
    return builder._build_from_schema(act_layer)


class TestConvTransposeSchemaPopulated(unittest.TestCase):
    """torch2act must populate every key REGISTRY lists in params_required
    for CONVTRANSPOSE2D — currently {weight, in_channels, out_channels,
    kernel_size}. Missing any key trips validate_layer at solve time and
    surfaces as ERROR_ValueError to the CLI (the cgan smoke symptom)."""

    def _build_conv_t_layer(self, **mod_kwargs):
        mod = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3, **mod_kwargs)
        net = nn.Sequential(mod)
        input_shape = (1, 4, 5, 5)
        layers, _, _ = build_act(net, input_shape, dtype=torch.float64)
        ct_layers = [L for L in layers if L.kind == LayerKind.CONVTRANSPOSE2D.value]
        self.assertEqual(len(ct_layers), 1, "exactly one CONVTRANSPOSE2D layer expected")
        return ct_layers[0]

    def test_all_required_params_present_default_kernel(self):
        L = self._build_conv_t_layer()
        required = REGISTRY[LayerKind.CONVTRANSPOSE2D.value]["params_required"]
        missing = [k for k in required if k not in L.params]
        self.assertEqual(missing, [], f"required params missing from torch2act output: {missing}")

    def test_in_channels_out_channels_kernel_size_values(self):
        L = self._build_conv_t_layer()
        self.assertEqual(int(L.params["in_channels"]), 4)
        self.assertEqual(int(L.params["out_channels"]), 2)
        ks = L.params["kernel_size"]
        ks_first = int(ks) if isinstance(ks, int) else int(ks[0])
        self.assertEqual(ks_first, 3)

    def test_validate_layer_does_not_raise(self):
        L = self._build_conv_t_layer(stride=2, padding=1, output_padding=1)
        # If params_required were missing keys, this would raise the same
        # ValueError the cgan smoke surfaced as ERROR_ValueError. Soundness
        # gate: solve-time path must not blow up at schema check.
        try:
            validate_layer(L)
        except Exception as e:
            self.fail(f"validate_layer raised on a correctly-converted CONVTRANSPOSE2D: {e}")

    def test_output_padding_preserved_in_params(self):
        L = self._build_conv_t_layer(stride=2, output_padding=1)
        # output_padding must survive conversion — it changes the spatial
        # output shape on asymmetric models (cgan_2023 cGAN_*_padding_1).
        self.assertIn("output_padding", L.params)
        op = L.params["output_padding"]
        op_first = int(op) if isinstance(op, int) else int(op[0])
        self.assertEqual(op_first, 1)


class TestConvTransposeRoundTripEquivalence(unittest.TestCase):
    """torch -> ACT -> torch round-trip must reproduce the original
    ConvTranspose2d's behaviour bit-for-bit on a concrete input.
    Asserts both shape and value equivalence."""

    def _round_trip(self, mod: nn.ConvTranspose2d, input_shape):
        net = nn.Sequential(mod)
        layers, _, _ = build_act(net, input_shape, dtype=torch.float64)
        ct_layers = [L for L in layers if L.kind == LayerKind.CONVTRANSPOSE2D.value]
        self.assertEqual(len(ct_layers), 1)
        return _rebuild_layer(ct_layers[0])

    def test_default_stride_padding(self):
        torch.manual_seed(0)
        mod = nn.ConvTranspose2d(3, 2, kernel_size=3).double()
        x = torch.randn(1, 3, 5, 5, dtype=torch.float64)
        y_ref = mod(x)
        rebuilt = self._round_trip(mod, x.shape).double()
        y_got = rebuilt(x)
        self.assertEqual(tuple(y_got.shape), tuple(y_ref.shape))
        self.assertTrue(
            torch.allclose(y_got, y_ref, atol=1e-12, rtol=0),
            f"round-trip output diverged: max|diff|={float((y_got - y_ref).abs().max())}",
        )

    def test_stride_and_output_padding(self):
        torch.manual_seed(1)
        mod = nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1).double()
        x = torch.randn(1, 4, 5, 5, dtype=torch.float64)
        y_ref = mod(x)
        rebuilt = self._round_trip(mod, x.shape).double()
        y_got = rebuilt(x)
        # The output_padding fix is load-bearing here — without it the
        # rebuilt module produces a smaller spatial shape.
        self.assertEqual(tuple(y_got.shape), tuple(y_ref.shape))
        self.assertTrue(
            torch.allclose(y_got, y_ref, atol=1e-12, rtol=0),
            f"output_padding round-trip diverged: shapes ref={tuple(y_ref.shape)} got={tuple(y_got.shape)}",
        )

    def test_no_bias_path(self):
        torch.manual_seed(2)
        mod = nn.ConvTranspose2d(3, 2, kernel_size=3, bias=False).double()
        x = torch.randn(1, 3, 5, 5, dtype=torch.float64)
        y_ref = mod(x)
        rebuilt = self._round_trip(mod, x.shape).double()
        y_got = rebuilt(x)
        self.assertTrue(torch.allclose(y_got, y_ref, atol=1e-12, rtol=0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
