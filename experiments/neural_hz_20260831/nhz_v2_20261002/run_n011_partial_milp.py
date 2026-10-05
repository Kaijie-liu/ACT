"""N011: terminal partial MILP on E0 rows (diagnostic, float engine n003.3).

Fixed configuration for every row: aligned propagation + LP tightening
(300 it); terminal LP for all disjuncts (1000 it); for every disjunct whose LP
bound is >= 0, HiGHS MILP with exact binaries on the last `--exact-layers` ReLU
layers (structural choice) and LP relaxation elsewhere, time limit per
disjunct.  Outcome MILP_CERT_PROBE if every disjunct is excluded (LP bound < 0
or MILP dual bound < 0).  Rigorous re-certification is a separate step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import ENGINE_VERSION, Engine, lp_upper, parse_vnnlib  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
EVID = "/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/evidence"


def build_partial(A, b, phases, K, exact_layers):
    sel = [p for p in phases if int(p.idx.numel())]
    sel = sel[-exact_layers:] if exact_layers > 0 else []
    nb = sum(int(p.idx.numel()) for p in sel)
    An = sp.csr_matrix(A.double().cpu().numpy()); bn = b.double().cpu().numpy()
    blocks = [sp.hstack([An, sp.csr_matrix((An.shape[0], nb))])]; rhs = [bn]
    col = 0
    for p in sel:
        m = int(p.idx.numel())
        gx = np.zeros((m, K)); g0 = p.gx.double().cpu().numpy(); gx[:, :g0.shape[1]] = g0
        gy = np.zeros((m, K)); g1 = p.gy.double().cpu().numpy(); gy[:, :g1.shape[1]] = g1
        cx = p.cx.double().cpu().numpy(); cy = p.cy.double().cpu().numpy()
        l = p.l.double().cpu().numpy(); u = p.u.double().cpu().numpy()
        D1 = sp.csr_matrix((-u, (np.arange(m), col + np.arange(m))), shape=(m, nb))
        D2 = sp.csr_matrix((-l, (np.arange(m), col + np.arange(m))), shape=(m, nb))
        blocks.append(sp.hstack([sp.csr_matrix(gy), D1])); rhs.append(-cy)              # y <= u d
        blocks.append(sp.hstack([sp.csr_matrix(gy - gx), D2])); rhs.append(cx - cy - l)  # y <= x - l(1-d)
        col += m
    return sp.vstack(blocks).tocsr(), np.concatenate(rhs), nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--exact-layers", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.2)
    ledger = json.load(open(f"{EVID}/{args.family}_evidence_baseline_v2.json"))
    inst = list(csv.reader(open(f"{ROOT}/{args.family}/instances.csv")))
    byrow = {r["row_index"]: r for r in ledger["rows"]}
    engines = {}
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for ri in [int(x) for x in args.rows.split(",")]:
            r = byrow[ri]
            onnx_rel, spec_rel, _ = inst[ri]
            mp = f"{ROOT}/{args.family}/{onnx_rel}"; spp = f"{ROOT}/{args.family}/{spec_rel}"
            if mp not in engines:
                engines[mp] = Engine(mp, "cuda", torch.float32)
            e = engines[mp]
            z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
            spec = parse_vnnlib(spp, int(np.prod(e.input_shape)), int(o0.c.numel()))
            lb, ub = spec.boxes[0]
            t0 = time.time()
            out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
            c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
            gg = torch.stack([-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts])
            cc = torch.as_tensor([d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32))
                                  for d in spec.disjuncts], device="cuda")
            lpb = lp_upper(gg, cc, A, b, 1000, 0.05).cpu().numpy()
            open_ids = [int(k) for k in np.argsort(-lpb) if lpb[k] >= 0]
            rec = {"engine": ENGINE_VERSION, "family": args.family, "row_index": ri, "baseline": r["baseline_verdict"],
                   "exact_layers": args.exact_layers, "time_limit": args.time_limit,
                   "lp_open": {str(k): float(lpb[k]) for k in open_ids}, "milp": {}}
            if open_ids:
                AA, bb, nb = build_partial(A, b, phases, K, args.exact_layers)
                rec["n_binaries"] = nb
                integ = np.concatenate([np.zeros(K), np.ones(nb)])
                bnds = Bounds(np.concatenate([-np.ones(K), np.zeros(nb)]), np.ones(K + nb))
                for k in open_ids:
                    t1 = time.time()
                    obj = np.concatenate([-gg[k].double().cpu().numpy(), np.zeros(nb)])
                    res = milp(obj, constraints=LinearConstraint(AA, -np.inf, bb), integrality=integ, bounds=bnds,
                               options={"time_limit": args.time_limit, "disp": False})
                    db = getattr(res, "mip_dual_bound", None)
                    ub_k = (-db + float(cc[k])) if db is not None and np.isfinite(db) else None
                    inc = (-res.fun + float(cc[k])) if res.fun is not None else None
                    rec["milp"][str(k)] = {"dual_upper": ub_k, "incumbent": inc, "status": int(res.status),
                                           "wall_s": time.time() - t1}
                    if ub_k is None or ub_k >= 0:
                        break      # row cannot be certified by this configuration; stop early
            excluded = all((v["dual_upper"] is not None and v["dual_upper"] < 0) for v in rec["milp"].values()) \
                and len(rec["milp"]) == len(open_ids)
            rec["outcome"] = "MILP_CERT_PROBE" if excluded else "OPEN"
            rec["wall_s"] = time.time() - t0
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(ri, rec["outcome"], {k: (round(v["dual_upper"], 4) if v["dual_upper"] is not None else None, v["status"])
                                        for k, v in rec["milp"].items()}, round(rec["wall_s"], 1), flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
