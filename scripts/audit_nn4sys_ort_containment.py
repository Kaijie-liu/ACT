"""ORT-sampled soundness audit for nn4sys (and any benchmark) CERT/UNK
instances.

Why
====
The R11/R12 helper-emitting handlers (broadcast Div via EXPAND, OnnxPow
chained MULs, ReduceSum routing) closed the conversion-side blockers
for nn4sys pensieve_*_parallel; structural interval-containment tests
on synthetic boxes pass. That is necessary, not sufficient, evidence.

This audit is a falsification probe: for each real benchmark instance,
sample ``n_samples`` points from the input box, forward each via the
canonical ORT session, and evaluate the VNNLIB output spec on the
result. The audit then cross-checks against ACT's verdict:

  * If ACT returned **CERTIFIED**, every sampled output must NOT hold
    the unsafe spec (the safety property must be observed for all
    sampled x). Any single violation = unsound CERT = critical bug.

  * If ACT returned **FALSIFIED**, the formal-mode receipt remains the
    authoritative witness check. Random samples are reported only as
    diagnostic context; they do not validate or invalidate that receipt.

  * If ACT returned **UNKNOWN**, no claim is being verified. The audit
    reports whether the sampled inputs lean safe or unsafe — useful
    information for prioritising precision work, not a soundness check.

Output: JSON summary + per-query violation list. Exits non-zero iff
ANY CERT verdict shows even one sampled unsafe, or if a requested ACT
verdict/input shape/input dtype cannot be resolved exactly. Absence of
a sampled counterexample is evidence only, never a proof of CERT.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path("/data1/Kane/ACT")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from act.front_end.specs import OutKind, OutputSpec
from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries


def _to_np(x) -> np.ndarray:
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unsafe_holds(out_spec: OutputSpec, y: np.ndarray, *, tol: float = 0.0) -> bool:
    """Mirror ``fal_receipt._eval_unsafe_with_tol`` but evaluate against
    a ``front_end.specs.OutputSpec`` directly (no ACT assert_layer
    construction needed). Returns True iff the UNSAFE region contains
    y under the given tolerance.

    The signal we audit on is **CERT → unsafe must never hold at any
    sampled y**. If this returns True for any sample on a CERT instance,
    the CERT is unsound.
    """
    kind = out_spec.kind
    if kind == OutKind.UNSAFE_LINEAR:
        C = _to_np(out_spec.c)
        d = _to_np(out_spec.d).reshape(-1)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        return bool(np.all(C @ y <= d + tol))
    if kind == OutKind.LINEAR_LE:
        coef = _to_np(out_spec.c).reshape(-1)
        d_scalar = float(_to_np(out_spec.d).reshape(-1)[0])
        return float(coef @ y) > d_scalar - tol
    if kind == OutKind.TOP1_ROBUST:
        t = int(getattr(out_spec, "y_true", 0))
        return any(y[j] >= y[t] - tol for j in range(len(y)) if j != t)
    if kind == OutKind.MARGIN_ROBUST:
        t = int(getattr(out_spec, "y_true", 0))
        m = float(getattr(out_spec, "margin", 0.0))
        return any(y[j] >= y[t] - m - tol for j in range(len(y)) if j != t)
    if kind == OutKind.RANGE:
        lb_t = getattr(out_spec, "lb", None)
        ub_t = getattr(out_spec, "ub", None)
        if lb_t is not None and np.any(y < _to_np(lb_t).reshape(-1) + tol):
            return True
        if ub_t is not None and np.any(y > _to_np(ub_t).reshape(-1) - tol):
            return True
        return False
    raise NotImplementedError(f"audit_nn4sys_ort_containment: unhandled OutputSpec.kind={kind}")


def _ort_session(onnx_path: Path):
    """Deterministic single-threaded CPU ORT session, matching the
    R9.3 receipt replay convention so audit ORT outputs are bit-equal
    to receipt-time ORT outputs."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    return sess


def _ort_numpy_dtype(type_name: str) -> np.dtype:
    """Map an ORT input type to the only NumPy dtype submitted to ORT.

    Do not retry with a different dtype after an execution error: an ORT
    shape or operator failure is evidence that the audit setup is wrong,
    not an invitation to guess a new input contract.
    """
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(double)": np.dtype(np.float64),
    }
    if type_name not in mapping:
        raise NotImplementedError(
            f"audit cannot feed ORT input type {type_name!r} without an "
            "explicit dtype policy"
        )
    return mapping[type_name]


def _resolve_ort_input_shape(in_shape_decl: Tuple[Any, ...], input_numel: int) -> Tuple[int, ...]:
    """Resolve an ORT input shape without changing its native rank.

    Symbolic/unknown dimensions may be inferred from the VNNLIB numel when
    exactly one is present. Static mismatches fail closed. In particular a
    native rank-1 model input ``(12296,)`` remains rank-1; it is never
    guessed into ``(1, 12296)``.
    """
    resolved: List[Optional[int]] = []
    unknown: List[int] = []
    known_prod = 1
    for index, dim in enumerate(in_shape_decl):
        if isinstance(dim, str) or dim is None or (
            isinstance(dim, int) and dim <= 0
        ):
            resolved.append(None)
            unknown.append(index)
            continue
        d = int(dim)
        resolved.append(d)
        known_prod *= d
    if len(unknown) > 1:
        raise ValueError(
            f"cannot infer ORT input shape {in_shape_decl!r} from "
            f"numel={input_numel}: multiple unknown dimensions"
        )
    if unknown:
        if known_prod == 0 or input_numel % known_prod != 0:
            raise ValueError(
                f"ORT input shape {in_shape_decl!r} is incompatible with "
                f"VNNLIB numel={input_numel}"
            )
        resolved[unknown[0]] = input_numel // known_prod
    elif known_prod != input_numel:
        raise ValueError(
            f"ORT native input shape {in_shape_decl!r} has numel={known_prod}, "
            f"but VNNLIB describes numel={input_numel}; refusing reshape guess"
        )
    return tuple(int(d) for d in resolved)


def _audit_one_instance(
    onnx_path: Path, vnnlib_path: Path, *,
    n_samples: int = 200,
    rng_seed: int = 0,
    act_verdict: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the ORT audit on one (onnx, vnnlib) pair.

    Returns a dict that records:
      * per-query unsafe sample counts at zero and small tolerance,
      * total samples that triggered the unsafe region,
      * a soundness verdict ("OK" or "CERT_UNSOUND") which is non-OK
        iff act_verdict was CERTIFIED and any sample landed in unsafe.
    """
    sess = _ort_session(onnx_path)
    in_meta = sess.get_inputs()[0]
    in_name = in_meta.name
    in_shape_decl = tuple(in_meta.shape)  # may contain symbolic / -1 dims
    input_dtype = _ort_numpy_dtype(in_meta.type)

    queries = parse_vnnlib_queries(vnnlib_path)
    rng = np.random.default_rng(rng_seed)

    per_query: List[Dict[str, Any]] = []
    any_cert_unsafe = False

    for q_idx, (in_spec, out_spec) in enumerate(queries):
        lb = _to_np(in_spec.lb).reshape(-1).astype(np.float64)
        ub = _to_np(in_spec.ub).reshape(-1).astype(np.float64)
        if lb.shape != ub.shape:
            raise ValueError(f"query {q_idx}: lb/ub shape mismatch {lb.shape} vs {ub.shape}")
        ort_shape = _resolve_ort_input_shape(in_shape_decl, lb.size)

        unsafe_zero = 0
        unsafe_small = 0
        # Pin one corner of the box for stress on inequality boundaries,
        # then mix in uniformly random samples.
        corner = ub.copy()
        for s_idx in range(n_samples):
            if s_idx == 0:
                x = corner
            elif s_idx == 1:
                x = lb.copy()
            else:
                x = lb + rng.random(lb.shape) * (ub - lb)
            x_ort = np.ascontiguousarray(x.astype(input_dtype)).reshape(ort_shape)
            y = sess.run(None, {in_name: x_ort})[0]
            y_flat = np.asarray(y).reshape(-1).astype(np.float64)
            if _unsafe_holds(out_spec, y_flat, tol=0.0):
                unsafe_zero += 1
            if _unsafe_holds(out_spec, y_flat, tol=1e-6):
                unsafe_small += 1

        q_record = {
            "q_idx": q_idx,
            "n_samples": int(n_samples),
            "unsafe_zero_tol": int(unsafe_zero),
            "unsafe_small_tol": int(unsafe_small),
        }
        per_query.append(q_record)
        # The soundness contract: a CERTIFIED verdict claims the unsafe
        # region is EMPTY over the input box. A single sampled x in the
        # unsafe region = unsound CERT.
        if act_verdict == "CERTIFIED" and unsafe_zero > 0:
            any_cert_unsafe = True

    return {
        "onnx_path": str(onnx_path),
        "vnnlib_path": str(vnnlib_path),
        "act_verdict": act_verdict,
        "n_samples_per_query": int(n_samples),
        "n_queries": len(per_query),
        "per_query": per_query,
        "soundness_verdict": (
            "CERT_COUNTEREXAMPLE_FOUND" if any_cert_unsafe
            else "NO_SAMPLED_COUNTEREXAMPLE"
        ),
    }


def _load_per_instance(per_instance_json: Path) -> Dict[int, Dict[str, Any]]:
    """Map ``official_instance_id -> per-instance record`` from one or
    more structured CLI run JSONs.

    The watchdog runner spawns one subprocess per instance and each
    writes its own ``per_instance_<bench>_<UTC>.json``. If the caller
    points at a directory we union across all JSONs in it; if it points
    at a single file we read only that. Empty dict on missing path."""
    if per_instance_json.is_dir():
        merged: Dict[int, Dict[str, Any]] = {}
        for f in sorted(per_instance_json.glob("per_instance_*.json")):
            if "watchdog" in f.name:
                continue  # synthetic watchdog-termination records have own verdict
            try:
                doc = json.loads(f.read_text())
            except Exception:
                continue
            for p in doc.get("per_instance", []):
                merged[int(p["official_instance_id"])] = p
        return merged
    if not per_instance_json.exists():
        return {}
    doc = json.loads(per_instance_json.read_text())
    return {int(p["official_instance_id"]): p for p in doc.get("per_instance", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, help="benchmark category dir name")
    ap.add_argument("--instance-ids", required=True,
                    help="comma-separated official instance ids to audit")
    ap.add_argument("--per-instance-json",
                    help="optional structured CLI run JSON to read ACT verdicts from")
    ap.add_argument("--canonical-root",
                    default="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks",
                    type=Path)
    ap.add_argument("--manifest-csv",
                    help="optional instances.csv; defaults to <canonical_root>/<bench>/instances.csv")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    bench_dir = Path(args.canonical_root) / args.benchmark
    manifest = Path(args.manifest_csv) if args.manifest_csv else bench_dir / "instances.csv"
    if not manifest.exists():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 2
    rows = list(csv.reader(manifest.open()))
    if rows and not rows[0][0].endswith(".onnx") and "/" not in rows[0][0]:
        rows = rows[1:]  # skip header if present
    wanted = [int(s) for s in args.instance_ids.split(",") if s.strip()]

    per_inst = _load_per_instance(Path(args.per_instance_json)) if args.per_instance_json else {}

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "ort_containment_summary.json"
    results: List[Dict[str, Any]] = []
    any_unsound = False

    for iid in wanted:
        if iid >= len(rows):
            print(f"iid {iid} out of range (manifest has {len(rows)} rows)", file=sys.stderr)
            return 2
        row = rows[iid]
        onnx_rel, vnnlib_rel = row[0], row[1]
        onnx_path = bench_dir / onnx_rel
        vnnlib_path = bench_dir / vnnlib_rel
        verdict = per_inst.get(iid, {}).get("cli_normalized")
        if args.per_instance_json and verdict is None:
            print(
                f"iid {iid}: ACT verdict not found in {args.per_instance_json}; "
                "refusing to label an unlabeled ORT probe as a verdict audit",
                file=sys.stderr,
            )
            return 2
        print(f"[audit] iid={iid} onnx={onnx_rel} vnnlib={vnnlib_rel} act_verdict={verdict}")
        t0 = time.monotonic()
        rec = _audit_one_instance(
            onnx_path, vnnlib_path,
            n_samples=args.n_samples, rng_seed=args.seed + iid,
            act_verdict=verdict,
        )
        rec["instance_id"] = iid
        rec["wall_s"] = time.monotonic() - t0
        results.append(rec)
        print(f"  -> verdict={rec['soundness_verdict']} "
              f"(per_query={[q['unsafe_zero_tol'] for q in rec['per_query']]})")
        if rec["soundness_verdict"] == "CERT_COUNTEREXAMPLE_FOUND":
            any_unsound = True

    summary = {
        "benchmark": args.benchmark,
        "n_samples_per_query": args.n_samples,
        "any_cert_unsound": any_unsound,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[audit] wrote {summary_path}")
    return 1 if any_unsound else 0


if __name__ == "__main__":
    sys.exit(main())
