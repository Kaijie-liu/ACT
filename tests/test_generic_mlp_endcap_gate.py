"""Unit tests for the structural generic MLP end-cap gate.

Run with ``PYTHONPATH=. python tests/test_generic_mlp_endcap_gate.py``
or as part of the full soundness suite.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from typing import Sequence

# Make the test runnable from the repo root without installing.
sys.path.insert(0, ".")

from act.back_end.profiles.generic_mlp_endcap_gate import (  # noqa: E402
    supports_generic_mlp_endcap,
)


@dataclass
class _FakeLayer:
    kind: str
    out_vars: Sequence[int]


def _net(*kinds_and_dims):
    return [_FakeLayer(k, list(range(d))) for (k, d) in kinds_and_dims]


def _top1_pair():
    """Minimal vnnlib pair with the top-1 robust signal."""
    return {"labeled_tensor": object()}


def _no_labeled_tensor():
    return {"labeled_tensor": None}


class GenericMlpEndcapGateTests(unittest.TestCase):
    """Each test names ONE acceptance criterion from the advisor."""

    # ── Positive cases ────────────────────────────────────────────────

    def test_tiny_style_tail_triggers(self):
        """Tiny: ... -> ADD -> FLATTEN -> DENSE -> RELU -> DENSE (out=200)."""
        layers = _net(
            ("CONV2D", 4096),
            ("RELU", 4096),
            ("ADD", 4096),
            ("FLATTEN", 4096),
            ("DENSE", 512),
            ("RELU", 512),
            ("DENSE", 200),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertTrue(diag.enabled, f"diag={diag}")
        self.assertEqual(diag.tail_kinds, ("DENSE", "RELU", "DENSE"))
        self.assertEqual(diag.final_out_dim, 200)
        self.assertTrue(diag.tail_supported)

    def test_two_dense_tail_triggers(self):
        """Degenerate: ... -> FLATTEN -> DENSE -> DENSE (no hidden RELU)."""
        layers = _net(
            ("CONV2D", 256),
            ("FLATTEN", 256),
            ("DENSE", 128),
            ("DENSE", 10),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertTrue(diag.enabled, f"diag={diag}")
        self.assertEqual(diag.tail_kinds, ("DENSE", "DENSE"))

    def test_matmul_synonyms_accepted(self):
        """MatMul/Gemm synonyms should also count as Dense."""
        layers = _net(
            ("CONV2D", 128),
            ("FLATTEN", 128),
            ("MATMUL", 64),
            ("RELU", 64),
            ("GEMM", 10),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertTrue(diag.enabled, f"diag={diag}")

    def test_trailing_assert_ignored_for_tail_kinds(self):
        """ASSERT noise after the tail must not block the gate."""
        layers = _net(
            ("FLATTEN", 64),
            ("DENSE", 32),
            ("RELU", 32),
            ("DENSE", 10),
            ("ASSERT", 1),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertTrue(diag.enabled)

    def test_final_out_dim_uses_last_non_assert(self):
        """Advisor's correction: out_dim should be the last NON-ASSERT layer."""
        layers = _net(
            ("FLATTEN", 64),
            ("DENSE", 32),
            ("RELU", 32),
            ("DENSE", 10),
            ("ASSERT", 1),  # ASSERT often has out_vars of size 1
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertEqual(
            diag.final_out_dim, 10,
            "ASSERT's out_vars=1 must not be reported as the final dim",
        )
        self.assertTrue(diag.enabled)

    # ── Negative cases ────────────────────────────────────────────────

    def test_cifar_narrow_profile_exclusion(self):
        """CIFAR has its own profile; this gate must defer."""
        layers = _net(
            ("FLATTEN", 256),
            ("DENSE", 128),
            ("RELU", 128),
            ("DENSE", 100),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=True,
        )
        self.assertFalse(diag.enabled)
        self.assertTrue(diag.cifar_endcap_active)

    def test_yolo_style_no_dense_after_flatten_refused(self):
        """YOLO ends in ...Conv -> Flatten (no Dense tail)."""
        layers = _net(
            ("CONV2D", 21125),
            ("FLATTEN", 21125),
            # No further Dense
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertFalse(diag.enabled)
        self.assertFalse(diag.tail_supported)

    def test_relusplitter_gate_enabled_structurally(self):
        """Relusplitter (oval21-style) has ...FLATTEN -> DENSE -> RELU ->
        DENSE (out=10). The structural gate must accept it; downstream
        sidecar may still refuse on snapshot shape, but the gate's
        verdict here is independent of that.
        """
        layers = _net(
            ("CONV2D", 2048),
            ("RELU", 2048),
            ("CONV2D", 1024),
            ("RELU", 1024),
            ("FLATTEN", 1024),
            ("DENSE", 100),
            ("RELU", 100),
            ("DENSE", 10),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertTrue(diag.enabled, f"diag={diag}")

    def test_single_dense_tail_accepted(self):
        """P2: ...FLATTEN -> DENSE (single layer affine head) is now
        supported by the LP-only research script via the
        ``_set_objective_single_dense`` path. Covers malbeware /
        soundnessbench / vgg-style heads where the snapshot at FLATTEN
        already includes any hidden Dense+ReLU above the final logit.
        """
        layers = _net(
            ("FLATTEN", 1024),
            ("DENSE", 10),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertTrue(diag.enabled)
        self.assertTrue(diag.tail_supported)
        self.assertEqual(diag.tail_kinds, ("DENSE",))

    def test_single_matmul_tail_accepted(self):
        """Same single-Dense rule should accept MatMul as the synonym."""
        layers = _net(
            ("FLATTEN", 512),
            ("MATMUL", 25),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertTrue(diag.enabled)
        self.assertTrue(diag.tail_supported)

    def test_single_dense_too_huge_out_dim_refused(self):
        """Single-Dense still must obey the out_dim cap."""
        layers = _net(
            ("FLATTEN", 4096),
            ("DENSE", 2048),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertFalse(diag.enabled)
        self.assertEqual(diag.final_out_dim, 2048)

    def test_no_flatten_refused(self):
        """ACASXu/MLP style: no Flatten layer at all."""
        layers = _net(
            ("DENSE", 64),
            ("RELU", 64),
            ("DENSE", 32),
            ("RELU", 32),
            ("DENSE", 5),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertFalse(diag.enabled)
        self.assertFalse(diag.tail_supported)
        self.assertIsNone(diag.tail_kinds)

    def test_huge_out_dim_refused(self):
        """Pixel/grid output rejects the gate."""
        layers = _net(
            ("FLATTEN", 4096),
            ("DENSE", 2048),
            ("RELU", 2048),
            ("DENSE", 2048),  # 2048 > 1024 cap
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertFalse(diag.enabled)
        self.assertEqual(diag.final_out_dim, 2048)

    def test_soundnessbench_style_yi_op_const_refused(self):
        """Soundnessbench: model has Flatten->Dense tail and labeled_tensor
        is set, but the vnnlib uses (Y_i op const) constraints instead of
        (>= Y_a Y_b) rival pairs. The LP-endcap can't decode that form,
        so the gate must refuse even though the structural tail matches.
        """
        import tempfile, os
        soundness_vnn = tempfile.NamedTemporaryFile(
            "w", suffix=".vnnlib", delete=False)
        soundness_vnn.write(
            "(declare-const Y_0 Real)\n"
            "(declare-const Y_1 Real)\n"
            "(assert (or\n"
            "  (and (<= Y_107 -0.034) (>= Y_175 -0.2))\n"
            "))\n"
        )
        soundness_vnn.close()
        try:
            layers = _net(
                ("FLATTEN", 384),
                ("DENSE", 384),
            )
            pair = {
                "labeled_tensor": object(),
                "vnnlib_path": soundness_vnn.name,
            }
            diag = supports_generic_mlp_endcap(
                layers=layers,
                pair=pair,
                cifar_endcap_active=False,
            )
            self.assertFalse(diag.enabled)
            self.assertFalse(diag.is_top1_robust)
        finally:
            os.unlink(soundness_vnn.name)

    def test_top1_yy_pair_in_vnnlib_accepted(self):
        """When the vnnlib does contain (>= Y_a Y_b) the gate accepts."""
        import tempfile, os
        top1_vnn = tempfile.NamedTemporaryFile(
            "w", suffix=".vnnlib", delete=False)
        top1_vnn.write(
            "(declare-const Y_0 Real)\n"
            "(declare-const Y_1 Real)\n"
            "(assert (or\n"
            "  (and (>= Y_2 Y_1))\n"
            "  (and (>= Y_3 Y_1))\n"
            "))\n"
        )
        top1_vnn.close()
        try:
            layers = _net(
                ("FLATTEN", 384),
                ("DENSE", 10),
            )
            pair = {
                "labeled_tensor": object(),
                "vnnlib_path": top1_vnn.name,
            }
            diag = supports_generic_mlp_endcap(
                layers=layers,
                pair=pair,
                cifar_endcap_active=False,
            )
            self.assertTrue(diag.enabled, f"diag={diag}")
            self.assertTrue(diag.is_top1_robust)
        finally:
            os.unlink(top1_vnn.name)

    def test_no_labeled_tensor_refused(self):
        """Non-top-1-robust vnnlib should not trigger the sidecar."""
        layers = _net(
            ("FLATTEN", 64),
            ("DENSE", 32),
            ("RELU", 32),
            ("DENSE", 10),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_no_labeled_tensor(),
            cifar_endcap_active=False,
        )
        self.assertFalse(diag.enabled)
        self.assertFalse(diag.is_top1_robust)

    def test_env_off_refused(self):
        """Explicit env opt-out should disable the gate even when structure
        matches."""
        layers = _net(
            ("FLATTEN", 64),
            ("DENSE", 32),
            ("RELU", 32),
            ("DENSE", 10),
        )
        for off_val in ("0", "false", "FALSE", "off", "no"):
            diag = supports_generic_mlp_endcap(
                layers=layers,
                pair=_top1_pair(),
                cifar_endcap_active=False,
                env={"ACT_HZ_MLP_ENDCAP_PROFILE": off_val},
            )
            self.assertFalse(diag.enabled, f"off_val={off_val!r}")
            self.assertTrue(diag.env_off)

    def test_three_dense_tail_refused(self):
        """3+ dense layers after Flatten not yet supported (LP script
        hardcodes 2-layer head). When extended, update both gate and test
        together."""
        layers = _net(
            ("FLATTEN", 64),
            ("DENSE", 32),
            ("RELU", 32),
            ("DENSE", 16),
            ("RELU", 16),
            ("DENSE", 10),
        )
        diag = supports_generic_mlp_endcap(
            layers=layers,
            pair=_top1_pair(),
            cifar_endcap_active=False,
        )
        self.assertFalse(diag.enabled)
        self.assertEqual(
            diag.tail_kinds,
            ("DENSE", "RELU", "DENSE", "RELU", "DENSE"),
        )


def _run() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(
        GenericMlpEndcapGateTests
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    n_pass = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\nResult: {n_pass}/{result.testsRun} passed")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_run())
