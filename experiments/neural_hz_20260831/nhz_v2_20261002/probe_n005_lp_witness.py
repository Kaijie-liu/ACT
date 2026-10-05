"""N005 probe: witness decoding from the terminal LP of the aligned HZ.

For every unsafe disjunct, solve the terminal LP (max violation over the LP
relaxation) and decode the input factors of its optimum into a concrete input
x = center + radius * xi.  The candidate is then checked by ONNX Runtime
against the original VNNLIB property (independent validator).  One decode per
disjunct per box; no iterative input search, no gradients of the concrete
network.  Diagnostic only.

Two decoders:
  * 'dual'  : box maximiser sign(g - A^T nu) at the final GPU multipliers;
  * 'highs' : exact LP optimum from HiGHS (CPU), when the LP is small enough.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import Engine, lp_upper, parse_vnnlib  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"


def check_property(sess, iname, x, ish, disjuncts):
    y = sess.run(None, {iname: x.reshape(ish).astype(np.float32)})[0].reshape(-1).astype(np.float64)
    for atoms in disjuncts:
        if all(float(a @ y) <= b for a, b in atoms):
            return True, y
    return False, y


def dual_decode(g, c, A, b, iters, lr):
    """Return the box maximiser at the optimised multipliers (input-part decoder)."""
    R = A.shape[0]
    nu = g.new_zeros((1, R)); m1 = torch.zeros_like(nu); m2 = torch.zeros_like(nu)
    for it in range(1, iters + 1):
        resid = g - nu @ A
        grad = b.unsqueeze(0) - torch.sign(resid) @ A.t()
        m1.mul_(0.9).add_(grad, alpha=0.1); m2.mul_(0.999).addcmul_(grad, grad, value=0.001)
        nu = (nu - lr * (m1 / (1 - 0.9 ** it)) / ((m2 / (1 - 0.999 ** it)).sqrt() + 1e-8)).clamp_(min=0)
    return torch.sign(g - nu @ A).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--decoder", default="dual", choices=["dual", "highs"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import onnxruntime as ort
    inst = list(csv.reader(open(f"{ROOT}/{args.family}/instances.csv")))
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    engines, sessions = {}, {}
    with os.fdopen(fd, "w") as fout:
        for ri in [int(x) for x in args.rows.split(",")]:
            onnx_rel, spec_rel = inst[ri][:2]
            mp = os.path.normpath(f"{ROOT}/{args.family}/{onnx_rel}"); sp = os.path.normpath(f"{ROOT}/{args.family}/{spec_rel}")
            if mp not in engines:
                engines[mp] = Engine(mp, "cuda", torch.float32)
                sessions[mp] = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
            e = engines[mp]; sess = sessions[mp]; iname = sess.get_inputs()[0].name
            z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
            spec = parse_vnnlib(sp, int(np.prod(e.input_shape)), int(o0.c.numel()))
            t0 = time.time(); found = None; tried = 0
            for lb, ub in spec.boxes:
                out, rows, K, _ = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), args.iters, 0.05)
                nzi = e.input_factor_index.cpu().numpy()
                c = out.c.reshape(-1); G = out.G.reshape(K, c.numel())
                A, bb = rows.dense(K)
                center = (lb + ub) / 2; rad = (ub - lb) / 2
                for atoms in spec.disjuncts:
                    a0, b0 = atoms[0]
                    a0t = torch.as_tensor(a0, device="cuda", dtype=torch.float32)
                    g = -(G @ a0t).unsqueeze(0); cc = (b0 - c @ a0t).reshape(1)
                    AA, BB = A, bb
                    if len(atoms) > 1:
                        ex = torch.stack([G @ torch.as_tensor(ak, device="cuda", dtype=torch.float32) for ak, _ in atoms[1:]])
                        exb = torch.as_tensor([bk for _, bk in atoms[1:]], device="cuda", dtype=torch.float32) - torch.stack(
                            [c @ torch.as_tensor(ak, device="cuda", dtype=torch.float32) for ak, _ in atoms[1:]])
                        AA = torch.cat([A, ex]); BB = torch.cat([bb, exb])
                    if args.decoder == "dual":
                        w = dual_decode(g, cc, AA, BB, 600, 0.05).double().cpu().numpy()
                    else:
                        from scipy.optimize import linprog
                        res = linprog(-g.reshape(-1).double().cpu().numpy(), A_ub=AA.double().cpu().numpy(),
                                      b_ub=BB.double().cpu().numpy(), bounds=(-1, 1), method="highs")
                        if res.x is None:
                            continue
                        w = res.x
                    xi = np.zeros(lb.size); xi[nzi] = w[: nzi.size]
                    x = np.clip(center + rad * xi, lb, ub)
                    tried += 1
                    ok, y = check_property(sess, iname, x, e.input_shape, spec.disjuncts)
                    if ok:
                        found = x; break
                if found is not None:
                    break
            rec = {"family": args.family, "row": ri, "onnx": onnx_rel, "decoder": args.decoder, "tried": tried,
                   "witness_valid_ort": found is not None, "wall_s": time.time() - t0}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(ri, onnx_rel, rec["witness_valid_ort"], tried, round(rec["wall_s"], 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
