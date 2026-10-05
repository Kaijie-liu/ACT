"""N016: N015's rigorous pipeline over formal unsolved rows (capability probe).

Same frozen modules and per-box logic as N015 (sound engine n004.2, rigorous
batched terminal LP, box-maximiser and MILP-incumbent witness candidates,
gamma=last-layer / lambda=all-layers HiGHS cutoff MILP).  Rows come from the
frozen 543-row manifest; the budget is the row's official timeout.  A spec with
several input boxes is CERT iff every box is excluded, ADV iff any box yields an
ORT-validated witness.  An ORT self-check of the engine's centre value map is
recorded per model.  Not a formal replay: formal promotion still needs all 2413
rows on one path.
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
from nhz_engine import parse_vnnlib  # noqa: E402
from nhz_sound import SOUND_VERSION, SoundEngine  # noqa: E402
from nhz_terminal import plan_milp_highs, sound_lp_batch  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
MANIFEST = "/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/manifests/formal_unsolved_structure_manifest_v1.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def solve_box(e, sess, iname, spec, lb, ub, deadline, iters=300, final_iters=1000):
    n_in = int(np.prod(e.input_shape))
    info = {}
    t = time.time()
    out, rset, K, phases = e.propagate(lb, ub, iters, 0.05)
    torch.cuda.synchronize(); info["propagate_s"] = time.time() - t
    t = time.time()
    lp = sound_lp_batch(out, rset, K, spec.disjuncts, final_iters, 0.05)
    torch.cuda.synchronize(); info["terminal_lp_s"] = time.time() - t
    ups = [d["upper"] for d in lp]
    open_ids = sorted([i for i, v in enumerate(ups) if v >= -1e-4], key=lambda i: -ups[i])
    info.update({"K": K, "unstable": sum(int(p["idx"].numel()) for p in phases), "lp_worst": max(ups), "lp_open": len(open_ids)})
    nzi = e.input_factor_index.cpu().numpy(); center = (lb + ub) / 2; rad = (ub - lb) / 2

    def check(w):
        xi = np.zeros(n_in); xi[nzi] = np.asarray(w)[: nzi.size]
        x = np.clip(center + rad * xi, lb, ub).astype(np.float32)
        y = sess.run(None, {iname: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
        hit = next((k for k, atoms in enumerate(spec.disjuncts) if all(float(a @ y) <= b for a, b in atoms)), None)
        return hit, x

    if not open_ids:
        return "CERT", info, None
    for i in open_ids:
        hit, x = check(lp[i]["cand"])
        if hit is not None:
            return "ADV", info, ("lp_box_maximiser", i, hit, x)
    milp_log = {}
    info["milp"] = milp_log
    for i in open_ids:
        remaining = deadline - time.time()
        if remaining <= 1:
            return "TIMEOUT", info, None
        d = lp[i]
        A_, b_ = rset.dense(K)
        m = plan_milp_highs(A_, b_, phases, K, d["g"], d["cc"], d["pad"], d["extra"], 1, 99, remaining)
        milp_log[str(i)] = {k: v for k, v in m.items() if k != "incumbent_w"}
        if m["incumbent_w"] is not None:
            hit, x = check(m["incumbent_w"])
            if hit is not None:
                return "ADV", info, ("milp_incumbent", i, hit, x)
        if not m["excluded"]:
            return ("TIMEOUT" if time.time() >= deadline - 1 else "UNKNOWN"), info, None
    return "CERT", info, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=""); ap.add_argument("--rows", default="")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--mem-fraction", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    inst = json.load(open(MANIFEST))["instances"]
    if a.families:
        fams = set(a.families.split(",")); inst = [r for r in inst if r["family"] in fams]
    if a.rows:
        ids = set(a.rows.split(",")); inst = [r for r in inst if r["row_identity"] in ids]
    engines, sessions = {}, {}
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for r in inst:
            mp = os.path.join(ROOT, r["model"]["relative_path"]); sp_ = os.path.join(ROOT, r["spec"]["relative_path"])
            budget = a.budget or float(r["source"]["timeout_seconds"])
            rec = {"engine": SOUND_VERSION, "row_identity": r["row_identity"], "family": r["family"], "baseline": r["verdict"],
                   "model": r["model"]["relative_path"], "budget_s": budget, "boxes": []}
            t0 = time.time()
            try:
                if mp not in engines:
                    assert sha256(mp) == r["model"]["sha256"], "model hash"
                    e = SoundEngine(mp, "cuda", torch.float64)
                    s = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                    x = np.random.default_rng(0).uniform(0, 1, size=e.input_shape).astype(np.float32)
                    ref = s.run(None, {s.get_inputs()[0].name: x})[0].reshape(-1)
                    o, *_ = e.propagate(x.reshape(-1).astype(np.float64), x.reshape(-1).astype(np.float64), 0)
                    engines[mp] = e; sessions[mp] = s
                    rec["ort_center_err"] = float(np.abs(o.c.reshape(-1).cpu().numpy() - ref).max())
                assert sha256(sp_) == r["spec"]["sha256"], "spec hash"
                e = engines[mp]; s = sessions[mp]; iname = s.get_inputs()[0].name
                n_in = int(np.prod(e.input_shape))
                o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
                spec = parse_vnnlib(sp_, n_in, int(o0.c.numel()))
                deadline = t0 + budget
                outcome = "CERT"
                for lb, ub in spec.boxes:
                    oc, info, wit = solve_box(e, s, iname, spec, lb, ub, deadline)
                    rec["boxes"].append(info)
                    if oc == "ADV":
                        src, i, hit, x = wit
                        rec.update({"witness_source": src, "witness_hits_disjunct": hit,
                                    "witness_x_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()})
                        np.save(f"{os.path.splitext(a.out)[0]}_x_{r['row_identity'].replace(':', '_')}.npy", x)
                        outcome = "ADV"; break
                    if oc != "CERT":
                        outcome = oc; break
                rec["outcome"] = outcome
            except Exception as ex:
                rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
            rec["wall_s"] = time.time() - t0
            if rec["outcome"] in ("CERT", "ADV") and rec["wall_s"] > budget:
                rec["over_budget_outcome"] = rec["outcome"]; rec["outcome"] = "TIMEOUT"
            fout.write(json.dumps(rec, default=float) + "\n"); fout.flush()
            print(r["row_identity"], r["verdict"], rec["outcome"], round(rec["wall_s"], 1),
                  [round(b.get("lp_worst", float("nan")), 3) for b in rec["boxes"]], rec.get("error", ""), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
