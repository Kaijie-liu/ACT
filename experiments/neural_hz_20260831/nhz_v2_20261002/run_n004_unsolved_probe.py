"""N004: run the N003 GPU engine probe over the formal 543 unsolved rows.

Diagnostic (no rounding control, no verdict authority).  Selection by the
frozen manifest manifests/formal_unsolved_structure_manifest_v1.json; model and
spec SHA-256 are checked before use.  Unsupported operators are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import Engine, certify_disjuncts, parse_vnnlib  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
MANIFEST = "/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/manifests/formal_unsolved_structure_manifest_v1.json"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--cohorts", default="")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--final-iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    inst = json.load(open(MANIFEST))["instances"]
    fams = set(args.families.split(",")) if args.families else None
    cohs = set(args.cohorts.split(",")) if args.cohorts else None
    sel = [r for r in inst if (fams is None or r["family"] in fams) and (cohs is None or r["cohort"] in cohs)]
    engines = {}
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for r in sel:
            mp = os.path.join(ROOT, r["model"]["relative_path"]); sp = os.path.join(ROOT, r["spec"]["relative_path"])
            rec = {"row_identity": r["row_identity"], "family": r["family"], "cohort": r["cohort"],
                   "baseline": r["verdict"], "model": r["model"]["relative_path"], "spec": r["spec"]["relative_path"],
                   "timeout": r["source"]["timeout_seconds"]}
            try:
                if mp not in engines:
                    if sha256(mp) != r["model"]["sha256"]:
                        raise RuntimeError("model hash mismatch")
                    engines[mp] = Engine(mp, "cuda", torch.float32)
                if sha256(sp) != r["spec"]["sha256"]:
                    raise RuntimeError("spec hash mismatch")
                e = engines[mp]
                n_in = int(np.prod(e.input_shape))
                z = torch.zeros(e.input_shape, device="cuda")
                o0, _, _, _ = e.propagate(z, z, 0)
                spec = parse_vnnlib(sp, n_in, int(o0.c.numel()))
                torch.cuda.synchronize(); t0 = time.time()
                worst = []; unst = []; K = None
                for lb, ub in spec.boxes:
                    out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), args.iters, args.lr)
                    worst.append(max(certify_disjuncts(out, rows, K, spec.disjuncts, args.final_iters, args.lr)))
                    unst.append(sum(int(p.idx.numel()) for p in phases))
                torch.cuda.synchronize()
                rec.update({"n_boxes": len(spec.boxes), "n_disjuncts": len(spec.disjuncts), "K_last": K,
                            "unstable_per_box": unst, "worst_disjunct_ub": max(worst),
                            "lp_cert_probe": bool(max(worst) < 0), "wall_s": time.time() - t0})
            except Exception as ex:
                rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(r["row_identity"], r["verdict"], rec.get("worst_disjunct_ub"), rec.get("lp_cert_probe"),
                  round(rec.get("wall_s", 0), 2), rec.get("error", ""), flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
