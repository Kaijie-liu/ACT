"""Integration regression test: LSNC-ReLU runs end-to-end without the
"shape '[1, 1, 6]' is invalid for input of size 2" tf_gather failure.

History: LSNC-ReLU single-instance smoke (2026-05-25) failed because
``TorchToACT._build_layer_graph`` only attached INPUT_SPEC as
predecessor of the FIRST model layer. Every other layer that read the
model input directly (lsnc has 2 Slice + 1 Gather on the input
alongside the main Gemm chain) kept ``preds=[]``, and analyze.py fell
through to the +/-inf default sized ``(B, len(out_vars))``, giving
tf_gather a wrong-sized Bin.

Fix: in ``_build_layer_graph``, additionally attach INPUT_SPEC as
predecessor of every model layer whose in_vars overlap INPUT_SPEC's
out_vars and whose preds are otherwise empty.

This test runs the actual CLI on the actual lsnc_relu instance the
original bug surfaced under, and asserts the run exits cleanly with no
RuntimeError shape failure. Subprocess-style because reconstructing the
full VerifiableModel wrap surface in-test is brittle (constructor and
LabeledInputTensor signatures evolve)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ACT_REPO = Path(__file__).resolve().parents[1]
LSNC_ONNX = Path(
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/lsnc_relu/onnx/"
    "relu_quadrotor2d_state.onnx"
)
LSNC_VNNLIB_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
PYTHON = "/data1/Kane/miniconda3/envs/act-py312/bin/python"


class TestLsncReluSmokeNoShapeFailure(unittest.TestCase):
    def setUp(self):
        if not LSNC_ONNX.exists():
            self.skipTest("lsnc_relu benchmark files not present")
        if not Path(PYTHON).exists():
            self.skipTest("act-py312 conda env not present")

    def test_lsnc_relu_1instance_runs_without_shape_error(self):
        """End-to-end CLI smoke: 1 instance, 25s budget, CPU. The original
        bug surfaced as ERROR_RuntimeError immediately at 0.1s. The fix
        produces 1/1 UNKNOWN within budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "ACT_VNNLIB_ROOT": str(LSNC_VNNLIB_ROOT),
                "ACT_FORMAL_RESULTS_DIR": tmpdir,
                "PYTHONPATH": str(ACT_REPO),
            })
            r = subprocess.run(
                [PYTHON, "-m", "act.pipeline",
                 "--verify", "vnnlib",
                 "--category", "lsnc_relu",
                 "--max-instances", "1", "--timeout", "25",
                 "--device", "cpu", "--dtype", "float64",
                 "--solvers", "hybridz"],
                cwd=str(ACT_REPO), env=env, capture_output=True, text=True,
                timeout=120,
            )
            self.assertEqual(
                r.returncode, 0,
                f"CLI exited rc={r.returncode}; stderr tail:\n{r.stderr[-2000:]}"
            )
            # The original bug error string MUST NOT appear
            self.assertNotIn(
                "is invalid for input of size 2",
                r.stdout + r.stderr,
                "lsnc_relu still hits the size-2 view error in tf_gather; "
                "wrapper INPUT_SPEC-connect-all fix has regressed"
            )
            # And the per-instance status must NOT be ERROR
            self.assertNotIn("ERROR_RuntimeError", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
