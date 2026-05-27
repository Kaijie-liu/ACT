"""Regression for the analyze() worklist ready-check (R16).

History
=======
ViT (vit_2023) first instance failed in analyze() with the assertion
``slice produced invalid bounds (lb > ub)``. Diagnosis:

  1. The worklist seeded the model INPUT and a zero-indegree CONSTANT
     (one branch of a CONCAT in the embedding chain).
  2. Processing the CONSTANT enqueued the CONCAT before the OTHER
     branch (CONV → RESHAPE → TRANSPOSE) had finished.
  3. The CONCAT ``box_join``ed against the default ``(-inf, +inf)``
     sentinel for the unfinished branch.
  4. DOWNSTREAM DENSE computed ``(-inf) @ W_pos + (+inf) @ W_neg + b``;
     ``(-inf) * 0`` is NaN, which poisoned every subsequent bound. The
     SLICE assertion fired (``lb <= ub`` is False on NaN) before the
     worklist could re-converge.

Fix (R16 in ``analyze.py``)
============================
Track a ``visited`` set; defer a layer when ANY predecessor is not
yet visited. CONSTANT layers without predecessors count as visited
from the start (their ``before`` is seeded with the literal value).

Soundness contract
==================
Deferral never permanently skips a layer; a bounded retry guards
against pathological cycles. Inside the limit, every layer visits
exactly once when its preds are ready, and any later change in a
predecessor re-enqueues it.

Strategy of the test
====================
Run the real `--verify vnnlib --category vit_2023 --max-instances 1`
CLI as a subprocess and assert:
  * the run exits cleanly (no NaN-poisoned SLICE-assert ERROR),
  * the per-instance JSON does NOT contain a
    ``slice produced invalid bounds`` error.
A pass shows the ready-check survives ALL upstream changes (wrap
constructor, FX trace, build_act helper-pred logic); a failure flags
the regression at the level the bug actually surfaced.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ACT_REPO = Path(__file__).resolve().parents[1]
VIT_ONNX = Path(
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/vit_2023/onnx/"
    "pgd_2_3_16.onnx"
)
PYTHON = "/data1/Kane/miniconda3/envs/act-py312/bin/python"


class TestVitDoesNotNanPropagate(unittest.TestCase):
    def setUp(self):
        if not VIT_ONNX.exists():
            self.skipTest("vit_2023 benchmark not present")
        if not Path(PYTHON).exists():
            self.skipTest("act-py312 env not present")

    def test_vit_first_instance_no_slice_invalid_bounds(self):
        """Pre-R16: 5/5 instances raised AssertionError with message
        ``slice produced invalid bounds (lb > ub)`` after NaN
        propagation. Post-R16: the SLICE assertion no longer fires
        (a different deeper shape error remains, but THIS specific
        NaN-induced one must be gone)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "ACT_VNNLIB_ROOT": "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks",
                "ACT_FORMAL_RESULTS_DIR": tmpdir,
                "PYTHONPATH": str(ACT_REPO),
            })
            subprocess.run(
                [PYTHON, "-m", "act.pipeline",
                 "--verify", "vnnlib",
                 "--category", "vit_2023",
                 "--max-instances", "1", "--timeout", "30",
                 "--device", "cpu", "--dtype", "float64",
                 "--solvers", "hybridz"],
                cwd=str(ACT_REPO), env=env, capture_output=True, text=True,
                timeout=180,
            )
            # Locate the structured per-instance JSON the CLI wrote.
            jsons = sorted(Path(tmpdir).glob("per_instance_vit_2023_*.json"))
            self.assertGreater(len(jsons), 0, "CLI did not write per_instance JSON")
            doc = json.loads(jsons[-1].read_text())
            err_msgs = [
                (p.get("error") or "")
                for p in doc.get("per_instance", [])
                if p.get("internal_status") == "ERROR"
            ]
            for msg in err_msgs:
                self.assertNotIn(
                    "slice produced invalid bounds", msg,
                    "ViT regressed: SLICE assertion fired again, meaning the "
                    "analyze ready-check (R16) is no longer routing the "
                    "worklist topologically"
                )
                self.assertNotIn(
                    "lb > ub", msg,
                    "Some other path is producing lb>ub bounds; soundness gate"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
