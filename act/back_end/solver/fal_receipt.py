"""FAL receipt layer for ACT (added 2026-05-24 per advisor Route C).

For every formally-emitted SAT verdict, produce a portable receipt:

    JSON record (atomic):
        x_star_sha256, model_sha256, spec_sha256
        ort_y_recomputed (saved as .npy alongside)
        spec_zero_tol_holds   (strict zero-tolerance verdict)
        spec_small_tol_holds  (slack 1e-6 verdict)
        timestamp_utc, git_head, schema_version
        witness_id (benchmark, instance_id, source, attempt)

The receipt is the audit substrate that lets reviewers re-derive the SAT
claim end-to-end without trusting any tool's internal verdict. It NEVER
changes ACT's verdict — it just records provenance.

This module is intentionally minimal and self-contained:
    - no torch dependency (uses raw onnxruntime)
    - no SATSidecar dependency (the exploratory work in HyZor stays separate)
    - no proof-side changes (the freeze rule applies)

Activation: receipts are written only when the env var
``ACT_FAL_RECEIPT_DIR`` is set to an existing directory. If unset or the
write fails, the verdict semantics are unaffected.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


RECEIPT_SCHEMA_VERSION = 1
ZERO_TOL: float = 0.0
SMALL_TOL: float = 1e-6
ENV_RECEIPT_DIR = "ACT_FAL_RECEIPT_DIR"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _eval_unsafe_with_tol(y: np.ndarray, assert_layer, *, tol: float) -> bool:
    """Generalization of ``solver_hz._eval_unsafe_strict`` with a tolerance
    parameter that GROWS the unsafe set (i.e. small_tol is MORE accepting
    of witnesses than zero_tol). For tol=0 this matches the strict version
    exactly.

    Convention (matches SATSidecar.replay_evaluator.atom_holds):
        - condition ``≤ d``: holds iff lhs ≤ d + tol
        - condition ``≥ d``: holds iff lhs ≥ d - tol
    """
    # Local helpers mirroring solver_hz internals (kept inline to avoid
    # an import cycle through solver_hz):
    from act.back_end.solver.solver_hz import (
        _unwrap_int, _unwrap_float, _to_np,
    )
    kind = assert_layer.params.get("kind")
    kstr = str(kind).split(".")[-1] if kind is not None else ""
    if kstr == "TOP1_ROBUST":
        t = int(_unwrap_int(assert_layer.params["y_true"]))
        return any(y[j] >= y[t] - tol for j in range(len(y)) if j != t)
    if kstr == "MARGIN_ROBUST":
        t = int(_unwrap_int(assert_layer.params["y_true"]))
        m = float(_unwrap_float(assert_layer.params["margin"]))
        return any(y[j] >= y[t] - m - tol for j in range(len(y)) if j != t)
    if kstr == "LINEAR_LE":
        coef = _to_np(assert_layer.params["c"]).reshape(-1)
        d = float(_unwrap_float(assert_layer.params["d"]))
        return float(coef @ y) > d - tol
    if kstr == "UNSAFE_LINEAR":
        C = _to_np(assert_layer.params["c"])
        d_vec = _to_np(assert_layer.params["d"]).reshape(-1)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        return bool(np.all(C @ y <= d_vec + tol))
    if kstr == "RANGE":
        lb_t = assert_layer.params.get("lb")
        ub_t = assert_layer.params.get("ub")
        if lb_t is not None and np.any(y < _to_np(lb_t).reshape(-1) + tol):
            return True
        if ub_t is not None and np.any(y > _to_np(ub_t).reshape(-1) - tol):
            return True
        return False
    return False


def compute_ort_y(model_path: str | Path, x_star: np.ndarray) -> np.ndarray:
    """Run ONNX runtime on x_star, return flat y (deterministic CPU)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    in_shape = [d if isinstance(d, int) and d > 0 else 1 for d in in_meta.shape]
    x_in = np.asarray(x_star, dtype=np.float32).reshape(in_shape)
    y = sess.run(None, {in_meta.name: x_in})[0]
    return np.asarray(y, dtype=np.float64).reshape(-1)


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(str(tmp), arr, allow_pickle=False)
    # np.save appends .npy if missing; normalize
    real_tmp = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npy")
    os.replace(real_tmp, path)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True).encode())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ReceiptCollisionError(RuntimeError):
    """Raised in formal-run mode when a receipt would overwrite a prior one
    in the same run directory. Filename collision indicates the caller is
    not propagating a unique (benchmark, instance_id, query_index) triple.

    Recovery: caller should fix the metadata or use a fresh run dir; the
    audit ledger MUST be uniquely keyed for any paper-grade claim.
    """


def write_receipt(
    *,
    x_star: np.ndarray,
    model_path: str | Path,
    spec_path: str | Path,
    assert_layer,
    benchmark: str,
    instance_id: int,
    query_index: int = 0,
    source: str = "act_hz_walker",
    attempt: int = 0,
    receipt_dir: Optional[str | Path] = None,
    formal_mode: Optional[bool] = None,
    input_box_holds: Optional[bool] = None,
    input_box_reason: Optional[str] = None,
) -> Optional[Path]:
    """Write a portable FAL receipt for one SAT witness.

    Activation:
        receipt_dir (explicit) > ACT_FAL_RECEIPT_DIR env > no-op.

    formal_mode:
        When True the caller declares this is a paper-grade run; the
        function will raise on:
          - missing receipt_dir
          - instance_id == -1 (sentinel)
          - filename collision with a prior receipt in the same dir
        The default (None) inspects env ``ACT_FAL_RECEIPT_FORMAL``: a
        truthy value (1/true/yes/on) activates formal-mode globally.

    Filename stem:
        ``{benchmark}_{instance_id}_q{query_index}_{source}_{attempt}``.
        The query_index segment is REQUIRED for multi-OR specs (one
        VNNLIB → N Cartesian queries → up to N receipts per instance).
        Without query_index every Cartesian receipt would collide on
        the same model path.

    Returns:
        JSON path on success, None on best-effort failure (only in
        non-formal mode). Raises in formal-mode on any precondition or
        write failure so audits cannot silently lose receipts.
    """
    if formal_mode is None:
        formal_mode = os.environ.get("ACT_FAL_RECEIPT_FORMAL", "").strip().lower() \
            in ("1", "true", "yes", "on")
    base = receipt_dir or os.environ.get(ENV_RECEIPT_DIR)
    if not base:
        if formal_mode:
            raise ReceiptCollisionError(
                "formal-mode receipt requires receipt_dir or ACT_FAL_RECEIPT_DIR; "
                "no SAT verdict may be emitted without a written receipt"
            )
        return None
    if formal_mode and int(instance_id) < 0:
        raise ReceiptCollisionError(
            f"formal-mode receipt requires real instance_id; got sentinel {instance_id}. "
            "Caller must propagate the official benchmark instance_id from instances.csv"
        )
    base_dir = Path(base)
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        if formal_mode:
            raise
        return None

    stem = f"{benchmark}_{instance_id}_q{query_index}_{source}_{attempt}"
    json_path = base_dir / f"{stem}.json"
    x_path = base_dir / f"{stem}.x_star.npy"
    y_path = base_dir / f"{stem}.y_ort.npy"
    if formal_mode and json_path.exists():
        raise ReceiptCollisionError(
            f"receipt collision: {json_path} already exists. The caller is "
            f"emitting a duplicate (benchmark={benchmark!r}, instance_id="
            f"{instance_id}, query_index={query_index}). Use a fresh run dir "
            f"or fix metadata propagation."
        )

    try:
        x_arr = np.asarray(x_star, dtype=np.float64).reshape(-1)
        y_ort = compute_ort_y(model_path, x_arr)
        zero_holds = _eval_unsafe_with_tol(y_ort, assert_layer, tol=ZERO_TOL)
        small_holds = _eval_unsafe_with_tol(y_ort, assert_layer, tol=SMALL_TOL)
        _atomic_write_npy(x_path, x_arr)
        _atomic_write_npy(y_path, y_ort)
        record = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "timestamp_utc": _now_iso(),
            "act_git_head": _git_head(),
            "witness_id": {
                "benchmark": benchmark,
                "instance_id": int(instance_id),
                "query_index": int(query_index),
                "source": source,
                "attempt": int(attempt),
            },
            "model_path": str(model_path),
            "spec_path": str(spec_path),
            "model_sha256": sha256_file(model_path),
            "spec_sha256": sha256_file(spec_path),
            "x_star_sha256": sha256_array(x_arr),
            "x_star_npy": x_path.name,
            "y_ort_npy": y_path.name,
            "spec_zero_tol_holds": bool(zero_holds),
            "spec_small_tol_holds": bool(small_holds),
            "tol_zero": ZERO_TOL,
            "tol_small": SMALL_TOL,
            # R9.3: record input-box gate result (None if caller didn't
            # check; True/False+reason if checked). Auditors must treat
            # a SAT receipt with input_box_holds != True as INVALID.
            "input_box_holds": (
                None if input_box_holds is None else bool(input_box_holds)
            ),
            "input_box_reason": input_box_reason or "not_checked",
        }
        _atomic_write_json(json_path, record)
        manifest_csv = base_dir / "MANIFEST.csv"
        header = ("benchmark,instance_id,query_index,source,attempt,"
                  "zero_tol,small_tol,artifact\n")
        line = (f"{benchmark},{instance_id},{query_index},{source},{attempt},"
                f"{int(zero_holds)},{int(small_holds)},{json_path.name}\n")
        write_header = not manifest_csv.exists()
        with open(manifest_csv, "a") as f:
            if write_header:
                f.write(header)
            f.write(line)
        return json_path
    except Exception:
        if formal_mode:
            raise
        return None


def load_receipt(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return json.loads(f.read().decode())
