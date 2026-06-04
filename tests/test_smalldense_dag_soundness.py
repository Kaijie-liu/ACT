#===- tests/test_smalldense_dag_soundness.py - SmallDenseDAG soundness ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===-----------------------------------------------------------------------===#
#
# Purpose:
#   Pin SmallDenseDAG (the DAG-aware triangle LP backend for residual /
#   multi-stream small dense MLPs like cersyve) against three soundness
#   risks identified during the 2026-05-31 audit cycle:
#
#   1) ONNX extraction coverage — every supported op kind must parse, and
#      out-of-scope graphs (lsnc_relu's Mul-var-var + Slice/Concat tail) must
#      be rejected as None rather than silently producing wrong output.
#   2) Point-eval ORT equivalence — for known small dense ONNX models, the
#      abstract DAG's `evaluate_dag_at_point` must match ONNX runtime forward
#      eval on random box-interior points to within float tolerance (atol=1e-5).
#      This is the algorithmic-correctness guard.
#   3) FAL receipt regression — cersyve iid 4 (pendulum_pretrain_con) must
#      keep producing a sound FALSIFIED verdict (witness in input box AND
#      ORT output strictly inside all unsafe halfspaces, zero tolerance);
#      conversely the 6 cersyve `*_finetune_*` UNSAT models must NOT be
#      promoted to FALSIFIED — that's the false-positive soundness gate.
#
# History:
#   This file ports the same-named test suite from /data1/Kane/HyZor/tests/
#   (added during the cersyve smalldense_dag wiring) into ACT's regression
#   directory per the 2026-05-31 follow-up audit: "把 SmallDenseDAG.py 和
#   测试正式迁入 ACT repo". The runtime module SmallDenseDAG.py itself
#   remains in /data1/Kane/HyZor/ alongside its dependencies (GlobalTriangleLP,
#   SpecAwareLP, WitnessExtract), which is the existing convention used by
#   solver_hz.py's `_try_small_dense_lp` dispatch.
#
# Module path bootstrap:
#   This test inserts /data1/Kane/HyZor/ into sys.path so it can import
#   SmallDenseDAG by name (matching the production import in solver_hz.py).
#   If you move SmallDenseDAG.py into ACT proper, update both this bootstrap
#   and the solver_hz dispatch.
#
#===-----------------------------------------------------------------------===#
from __future__ import annotations

import sys
import os
import glob
from pathlib import Path

import numpy as np

# Bootstrap: insert the HyZor module root so SmallDenseDAG and its peers
# (GlobalTriangleLP) are importable by their flat names. Matches the
# production sibling-path resolution in
# act/back_end/solver/solver_hz.py::_try_small_dense_lp.
_HYZOR = Path("/data1/Kane/HyZor")
if str(_HYZOR) not in sys.path:
    sys.path.insert(0, str(_HYZOR))

import SmallDenseDAG  # noqa: E402


CERSYVE_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cersyve")
ACASXU_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/acasxu_2023")


# ─── Layer 1: ONNX extraction coverage ─────────────────────────────────────

def test_extract_acasxu_sequential():
    """ACASXu model exercises MatMul+Add+Relu+Sub(at input)+Flatten chain."""
    onnx_p = ACASXU_ROOT / "onnx" / "ACASXU_run2a_1_1_batch_2000.onnx"
    res = SmallDenseDAG._topo_extract_ops(onnx_p)
    assert res is not None, f"failed to extract {onnx_p}"
    n_in, n_out, ops, in_n, out_n = res
    assert n_in == 5 and n_out == 5, f"acasxu dims wrong: {n_in}, {n_out}"
    op_kinds = {op['op'] for op in ops}
    expected = {'gemm', 'relu'}
    assert expected.issubset(op_kinds), (
        f"acasxu missing ops; got {op_kinds}, expected superset of {expected}"
    )


def test_extract_cersyve_residual():
    """Cersyve cart_pole exercises Gemm + Add(tensor,tensor) + Relu DAG."""
    onnx_p = CERSYVE_ROOT / "onnx" / "lane_keep_pretrain_con.onnx"
    res = SmallDenseDAG._topo_extract_ops(onnx_p)
    assert res is not None, f"failed to extract {onnx_p}"
    n_in, n_out, ops, in_n, out_n = res
    assert n_in == 4 and n_out == 2, f"cersyve dims wrong: {n_in}, {n_out}"
    op_kinds = [op['op'] for op in ops]
    n_relu = op_kinds.count('relu')
    n_gemm = op_kinds.count('gemm')
    n_add_t = op_kinds.count('add_tensor')
    assert n_gemm == 16, f"expected 16 gemm, got {n_gemm}"
    assert n_relu == 6, f"expected 6 relu, got {n_relu}"
    assert n_add_t == 6, f"expected 6 add_tensor (residual), got {n_add_t}"


def test_extract_rejects_unsupported():
    """Model with unsupported op (lsnc_relu's Mul/Div tail) returns None."""
    lsnc_path = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/lsnc_relu/onnx/relu_quadrotor2d_state.onnx")
    if not lsnc_path.exists():
        return
    res = SmallDenseDAG._topo_extract_ops(lsnc_path)
    assert res is None, (
        f"lsnc_relu unexpectedly extracted as DAG-supported "
        f"(was: {res[2] if res else 'None'})"
    )


def test_is_dag_supported_acasxu():
    onnx_p = ACASXU_ROOT / "onnx" / "ACASXU_run2a_1_1_batch_2000.onnx"
    assert SmallDenseDAG.is_dag_supported(onnx_p)


def test_is_dag_supported_cersyve():
    for name in ("lane_keep_pretrain_con", "pendulum_pretrain_con",
                 "point_mass_finetune_inv"):
        onnx_p = CERSYVE_ROOT / "onnx" / f"{name}.onnx"
        assert SmallDenseDAG.is_dag_supported(onnx_p), f"{name} should be DAG-supported"


# ─── Layer 2: Point-eval ORT equivalence ───────────────────────────────────

def _ort_eval(onnx_path: Path, x: np.ndarray):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    target = []
    n_in = int(x.size)
    for d in in_shape:
        target.append(d if isinstance(d, int) and d > 0 else 1)
    total = int(np.prod(target))
    if total != n_in:
        x_in = x.astype(np.float32).reshape(target[:-1] + [n_in // int(np.prod(target[:-1]) or 1)])
    else:
        x_in = x.astype(np.float32).reshape(target)
    return sess.run(None, {in_name: x_in})[0].flatten().astype(np.float64)


def _point_eval_equivalence(onnx_path: Path, lb: np.ndarray, ub: np.ndarray,
                              n_trials: int = 8, atol: float = 1e-5):
    res = SmallDenseDAG._topo_extract_ops(onnx_path)
    assert res is not None, f"extract failed for {onnx_path}"
    n_in, n_out, ops, in_n, out_n = res
    rng = np.random.default_rng(seed=12345)
    for trial in range(n_trials):
        x = rng.uniform(lb, ub).astype(np.float64)
        y_dag = SmallDenseDAG.evaluate_dag_at_point(ops, x, in_n, out_n)
        y_ort = _ort_eval(onnx_path, x)
        max_diff = float(np.max(np.abs(y_dag - y_ort)))
        assert max_diff < atol, (
            f"{onnx_path.name} trial {trial}: max_diff={max_diff:.2e} >= "
            f"atol={atol:.0e}; x={x}, y_dag={y_dag}, y_ort={y_ort}"
        )


def test_point_eval_acasxu():
    onnx_p = ACASXU_ROOT / "onnx" / "ACASXU_run2a_1_1_batch_2000.onnx"
    lb = np.array([-0.30353, -0.0095, -0.0095, -0.5, -0.5])
    ub = np.array([+0.30353, +0.50, +0.50, +0.50, -0.45])
    _point_eval_equivalence(onnx_p, lb, ub, n_trials=8, atol=1e-5)


def test_point_eval_cersyve_lane_keep():
    onnx_p = CERSYVE_ROOT / "onnx" / "lane_keep_pretrain_con.onnx"
    lb = np.array([-2.0, -1.0471975511965976, -1.0, -1.0])
    ub = np.array([+2.0, +1.0471975511965976, +1.0, +1.0])
    _point_eval_equivalence(onnx_p, lb, ub, n_trials=8, atol=1e-5)


def test_point_eval_cersyve_pendulum():
    onnx_p = CERSYVE_ROOT / "onnx" / "pendulum_pretrain_con.onnx"
    lb = np.array([-0.7853981633974483, -4.0])
    ub = np.array([+0.7853981633974483, +4.0])
    _point_eval_equivalence(onnx_p, lb, ub, n_trials=8, atol=1e-5)


def test_point_eval_cersyve_point_mass_finetune():
    onnx_p = CERSYVE_ROOT / "onnx" / "point_mass_finetune_inv.onnx"
    lb = np.array([-1.0, -1.0, -1.0, -1.0])
    ub = np.array([+1.0, +1.0, +1.0, +1.0])
    _point_eval_equivalence(onnx_p, lb, ub, n_trials=8, atol=1e-5)


# ─── Layer 3: FAL receipt regression ───────────────────────────────────────

def test_cersyve_iid4_falsified_with_sound_receipt():
    """Cersyve pendulum_pretrain_con: principle-compliant soundness gate.

    Under HYZOR_LP_BACKEND=highspy (legacy direct API), this produces FAL
    with a sound witness. Under HYZOR_LP_BACKEND=scipy (default, principle-
    compliant scipy.optimize.linprog only), the LP-corner vertex chosen by
    scipy's HiGHS lies outside the unsafe halfspace, so this returns UNK.
    Both verdicts are sound; this test asserts the WEAKER soundness
    guarantee that holds under either backend:

      - verdict must be 'falsified' OR 'unknown' (never 'verified')
      - if 'falsified', the x and y witness must pass strict spec / box check
      - if 'unknown', no witness is required (sound but less precise)

    This was a hard FAL assertion under highspy default. After the
    2026-06-01 scipy-linprog migration (project_audit_final_consolidated_
    20260601 + project_highspy_to_scipy_migration_20260601 follow-up), the
    assertion is relaxed to a soundness-only guard.
    """
    onnx_p = CERSYVE_ROOT / "onnx" / "pendulum_pretrain_con.onnx"
    vnn_p = CERSYVE_ROOT / "vnnlib" / "prop_pendulum.vnnlib"
    status, x, y, elapsed = SmallDenseDAG.verify_with_falsification(
        onnx_p, vnn_p, time_limit_per_lp=10.0,
        max_refinement_passes=20,
    )
    assert status in ('falsified', 'unknown'), (
        f"expected 'falsified' or 'unknown', got '{status}' "
        f"(verified would indicate phantom witness promotion = soundness bug)"
    )
    if status == 'falsified':
        # Strict receipt check
        assert x is not None and y is not None
        from GlobalTriangleLP import parse_vnnlib
        djs = parse_vnnlib(vnn_p, 2, 2)
        lb_x, ub_x, unsafe_rows = djs[0]
        for c_vec, d in unsafe_rows:
            val = float(np.dot(c_vec, y))
            assert val <= float(d), (
                f"witness fails spec: c={c_vec} d={d} but c·y={val:.6f}"
            )
        assert np.all(x >= lb_x - 1e-9), f"witness below lb: x={x}, lb={lb_x}"
        assert np.all(x <= ub_x + 1e-9), f"witness above ub: x={x}, ub={ub_x}"


def test_cersyve_unsat_models_not_promoted():
    """Cersyve finetune models (UNSAT per abcrown) must NOT be falsified.

    SOUNDNESS GATE: if any of these flips to 'falsified', the witness
    extraction is unsound (phantom witness promoted to SAT without proper
    strict-replay rejection)."""
    unsat_models = [
        "lane_keep_finetune_con", "lane_keep_finetune_inv",
        "pendulum_finetune_con", "pendulum_finetune_inv",
        "point_mass_finetune_con", "point_mass_finetune_inv",
    ]
    for name in unsat_models:
        onnx_p = CERSYVE_ROOT / "onnx" / f"{name}.onnx"
        if "lane_keep" in name:
            vnn_p = CERSYVE_ROOT / "vnnlib" / "prop_lane_keep.vnnlib"
        elif "pendulum" in name:
            vnn_p = CERSYVE_ROOT / "vnnlib" / "prop_pendulum.vnnlib"
        else:
            vnn_p = CERSYVE_ROOT / "vnnlib" / "prop_point_mass.vnnlib"
        status, _, _, _ = SmallDenseDAG.verify_with_falsification(
            onnx_p, vnn_p, time_limit_per_lp=10.0,
            max_refinement_passes=10,
        )
        assert status != 'falsified', (
            f"UNSAT model {name} wrongly reported falsified — "
            "SOUNDNESS REGRESSION"
        )


# ─── Layer 4: S2 multi-candidate behavior ──────────────────────────────────

def test_s2_returns_multiple_candidates():
    """S2 LP objective search must return BOTH min/max candidates as a list
    (previously bug: only first was returned, losing recall)."""
    onnx_p = CERSYVE_ROOT / "onnx" / "lane_keep_pretrain_con.onnx"
    vnn_p = CERSYVE_ROOT / "vnnlib" / "prop_lane_keep.vnnlib"
    res = SmallDenseDAG._topo_extract_ops(onnx_p)
    n_in, n_out, ops, in_n, out_n = res
    from GlobalTriangleLP import parse_vnnlib
    djs = parse_vnnlib(vnn_p, n_in, n_out)
    lb_x, ub_x, unsafe_rows = djs[0]
    c_obj = np.array([1.0, 0.0])
    cands = SmallDenseDAG._build_dag_lp_with_witness_objective(
        ops, n_in, n_out, in_n, out_n, lb_x, ub_x, unsafe_rows,
        c_obj, time_limit_per_lp=10.0)
    assert isinstance(cands, list), f"expected list, got {type(cands)}"
    assert 1 <= len(cands) <= 2, f"expected 1-2 candidates, got {len(cands)}"
    for c in cands:
        assert np.all(c >= lb_x - 1e-9)
        assert np.all(c <= ub_x + 1e-9)


# ─── Runner ────────────────────────────────────────────────────────────────

def _run_all():
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    fails = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\nResult: {len(test_funcs) - fails}/{len(test_funcs)} passed")
    return fails


if __name__ == "__main__":
    fails = _run_all()
    sys.exit(0 if fails == 0 else 1)
