"""Backend guard + capability smoke for the small-dense LP path.

P3 ("No Gurobi / no commercial MILP") allows continuous LP via either
scipy.optimize.linprog(method='highs') or the official HiGHS Python API
(highspy). This test file pins the runtime contract:

  1. Production default is highspy: `HYZOR_LP_BACKEND` unset → highspy.
  2. Audit-strict scipy profile is reachable: `HYZOR_LP_BACKEND=scipy`
     → scipy_lp_backend wrapper around scipy.linprog.
  3. Neither backend introduces integer variables (no MILP).
  4. `HYZOR_SCIPY_LOOP_HIGHSPY` default is 1 (matches production highspy).

Plus a capability smoke that fails LOUDLY if the production default
silently loses the cersyve iid 4 FAL — soundness is necessary but not
sufficient; we also need to keep verifier capability.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np

# pytest is optional; we only use pytest.skip(). Fall back to a no-op
# class so `python test_lp_backend_guard.py` works in environments
# (like the soundness gate) without pytest installed.
try:
    import pytest  # type: ignore
except Exception:
    class _PytestStub:
        @staticmethod
        def skip(reason):
            raise RuntimeError(f"SKIP: {reason}")
    pytest = _PytestStub()  # type: ignore


HYZOR_ROOT = Path("/data1/Kane/HyZor")
CERSYVE_ROOT = Path(
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cersyve"
)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _force_backend(name: str | None):
    """Set HYZOR_LP_BACKEND and force re-import of GTL + downstream modules
    so the change takes effect within a single pytest session."""
    if name is None:
        os.environ.pop("HYZOR_LP_BACKEND", None)
    else:
        os.environ["HYZOR_LP_BACKEND"] = name
    for m in [
        "SmallDenseDAG",
        "GlobalTriangleLP",
        "scipy_lp_backend",
        "SpecAwareLP",
        "WitnessExtract",
    ]:
        if m in sys.modules:
            del sys.modules[m]
    if str(HYZOR_ROOT) not in sys.path:
        sys.path.insert(0, str(HYZOR_ROOT))


# ─── Guard 1: production default is highspy ───────────────────────────────


def test_default_backend_is_highspy_module():
    """No HYZOR_LP_BACKEND env → `_hp` resolves to the highspy module,
    not scipy_lp_backend. This is the production contract."""
    _force_backend(None)
    GTL = importlib.import_module("GlobalTriangleLP")
    GTL._lazy_imports()
    assert GTL._hp is not None
    mod_name = getattr(GTL._hp, "__name__", "<anon>")
    assert mod_name == "highspy", (
        f"production default broke: expected highspy, got {mod_name!r}"
    )


def test_default_backend_is_highspy_has_no_milp_call():
    """Production highspy backend must not expose any MILP / integrality
    entry point that HyZor calls. We check that GTL.verify never reaches
    `passModel` with an integer-variable LP — the highspy API doesn't
    expose integer-variable LPs by default and GTL has no `integrality=`
    plumbing anywhere."""
    _force_backend(None)
    GTL = importlib.import_module("GlobalTriangleLP")
    src = Path(GTL.__file__).read_text()
    forbidden = [
        "integrality",
        "HighsMipSolver",
        "MipSolver",
        "milp",
        "binary_vars",
        "addBinary",
    ]
    found = [s for s in forbidden if s in src]
    assert not found, (
        f"GlobalTriangleLP contains MILP-related symbols: {found}"
    )


# ─── Guard 2: audit-strict scipy profile reachable ────────────────────────


def test_scipy_backend_opt_in_works():
    """HYZOR_LP_BACKEND=scipy → `_hp` resolves to scipy_lp_backend wrapper."""
    _force_backend("scipy")
    GTL = importlib.import_module("GlobalTriangleLP")
    GTL._lazy_imports()
    mod_name = getattr(GTL._hp, "__name__", "<anon>")
    assert mod_name == "scipy_lp_backend", (
        f"audit-strict scipy profile broke: expected scipy_lp_backend, "
        f"got {mod_name!r}"
    )
    # The scipy wrapper must implement the API surface we rely on.
    for attr in [
        "Highs", "HighsLp", "HighsModelStatus", "MatrixFormat", "kHighsInf"
    ]:
        assert hasattr(GTL._hp, attr), (
            f"scipy backend missing attr {attr!r}"
        )


def test_scipy_backend_uses_continuous_lp_only():
    """The scipy wrapper must call scipy.optimize.linprog without
    `integrality=` (which would be a MILP request)."""
    _force_backend("scipy")
    SCB = importlib.import_module("scipy_lp_backend")
    src = Path(SCB.__file__).read_text()
    assert "integrality=" not in src, (
        "scipy_lp_backend appears to request MILP via integrality= — "
        "this would violate P3"
    )


# ─── Guard 3: bounds_tighten flag default ─────────────────────────────────


def test_bounds_tighten_highspy_default_is_on():
    """HYZOR_SCIPY_LOOP_HIGHSPY default must be '1' (highspy warm-start ON)
    so the production default matches `HYZOR_LP_BACKEND=highspy`. If a
    future commit flips this to '0' silently, the production V/A would
    regress invisibly — this test catches that."""
    bt_path = Path(
        "/data1/Kane/ACT/act/back_end/hybridz_tf/algorithms/bounds_tighten.py"
    )
    src = bt_path.read_text()
    # The exact default-bake line:
    #   _highspy_on = _os.environ.get("HYZOR_SCIPY_LOOP_HIGHSPY", "1") == "1"
    assert (
        '"HYZOR_SCIPY_LOOP_HIGHSPY", "1"' in src
        or "'HYZOR_SCIPY_LOOP_HIGHSPY', '1'" in src
    ), "HYZOR_SCIPY_LOOP_HIGHSPY default is no longer '1'"


# ─── Capability smoke: highspy default must recover cersyve iid 4 FAL ─────


def test_cersyve_iid4_capability_under_production_default():
    """Capability guard: under the production default (no env override),
    cersyve iid 4 (pendulum_pretrain_con) MUST return 'falsified' with a
    receipt that passes strict input-box + zero-tol checks.

    The soundness test in test_smalldense_dag_soundness.py allows
    {FAL, UNK} under either backend (sound but possibly degraded). This
    capability test pins the production-grade outcome separately so a
    silent capability regression cannot ship."""
    _force_backend(None)  # production default = highspy
    SmallDenseDAG = importlib.import_module("SmallDenseDAG")
    onnx_p = CERSYVE_ROOT / "onnx" / "pendulum_pretrain_con.onnx"
    vnn_p = CERSYVE_ROOT / "vnnlib" / "prop_pendulum.vnnlib"
    if not (onnx_p.exists() and vnn_p.exists()):
        pytest.skip("cersyve benchmark not available locally")
    status, x, y, _ = SmallDenseDAG.verify_with_falsification(
        onnx_p, vnn_p, time_limit_per_lp=10.0,
        max_refinement_passes=20,
    )
    assert status == "falsified", (
        f"capability regression under production default: cersyve iid 4 "
        f"expected 'falsified', got {status!r}. The audit profile may "
        f"tolerate UNK; production must not."
    )
    assert x is not None and y is not None
    GTL = importlib.import_module("GlobalTriangleLP")
    djs = GTL.parse_vnnlib(vnn_p, 2, 2)
    lb_x, ub_x, unsafe_rows = djs[0]
    # Strict box check
    assert np.all(x >= lb_x - 1e-9), f"witness below lb: x={x}"
    assert np.all(x <= ub_x + 1e-9), f"witness above ub: x={x}"
    # Strict zero-tol unsafe-spec check
    for c_vec, d in unsafe_rows:
        val = float(np.dot(c_vec, y))
        assert val <= float(d), (
            f"witness fails unsafe spec: c·y={val:.6e} > d={d}"
        )


if __name__ == "__main__":
    # Allow `python test_lp_backend_guard.py` to run as a smoke check.
    for fn in [
        test_default_backend_is_highspy_module,
        test_default_backend_is_highspy_has_no_milp_call,
        test_scipy_backend_opt_in_works,
        test_scipy_backend_uses_continuous_lp_only,
        test_bounds_tighten_highspy_default_is_on,
        test_cersyve_iid4_capability_under_production_default,
    ]:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERR   {fn.__name__}: {type(e).__name__}: {e}")
