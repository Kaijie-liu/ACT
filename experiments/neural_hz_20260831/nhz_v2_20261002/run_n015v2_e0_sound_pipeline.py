"""N015 v2: identical decision logic to N015 v1; additionally saves the float64 witness point (x64) next to the float32 ORT input, for the S1/S2 audit (N022 v3).

N015: rigorous single-path Neural-HZ pipeline over E0 rows (candidate path v1).

Fixed configuration for every row:
  1. sound engine n004.2 (float64, rounding radii) with 300-iteration LP tightening;
  2. rigorous batched terminal LP for every unsafe disjunct;
  3. for open disjuncts (worst first): level-plan MILP gamma=last ReLU layer,
     lambda=all layers, HiGHS with objective cutoff, within the remaining budget;
  4. witness candidates: box maximiser at the LP multipliers and the MILP
     incumbent; validated by ONNX Runtime against the original VNNLIB property.
Outcome: CERT (every disjunct excluded), ADV (ORT-validated witness), TIMEOUT
(budget exhausted) or UNKNOWN.  Witness decoding is a terminal primal heuristic,
reported in its own field.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound import SOUND_VERSION, SoundEngine
from nhz_engine import parse_vnnlib
from nhz_terminal import plan_milp_highs, sound_lp_batch
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
EVID = "/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/evidence"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True); ap.add_argument("--rows", default="all")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--final-iters", type=int, default=1000)
    ap.add_argument("--mem-fraction", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    ledger = json.load(open(f"{EVID}/{a.family}_evidence_baseline_v2.json"))
    inst = list(csv.reader(open(f"{ROOT}/{a.family}/instances.csv")))
    rows = ledger["rows"]
    if a.rows not in ("all",):
        want = {int(x) for x in a.rows.split(",")}; rows = [r for r in rows if r["row_index"] in want]
    engines, sess = {}, {}
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for r in rows:
            onnx_rel, spec_rel, _ = inst[r["row_index"]]
            mp = f"{ROOT}/{a.family}/{onnx_rel}"; sp_ = f"{ROOT}/{a.family}/{spec_rel}"
            rec = {"engine": SOUND_VERSION, "family": a.family, "row_index": r["row_index"], "baseline": r["baseline_verdict"],
                   "budget_s": a.budget, "stages": {}}
            t0 = time.time()
            try:
                if mp not in engines:
                    assert sha256(mp) == r["model_sha256"], "model hash"
                    engines[mp] = SoundEngine(mp, "cuda", torch.float64)
                    sess[mp] = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                assert sha256(sp_) == r["spec_sha256"], "spec hash"
                e = engines[mp]; s = sess[mp]; iname = s.get_inputs()[0].name
                n_in = int(np.prod(e.input_shape))
                o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
                spec = parse_vnnlib(sp_, n_in, int(o0.c.numel()))
                lb, ub = spec.boxes[0]
                t_start = time.time()
                out, rset, K, phases = e.propagate(lb, ub, a.iters, 0.05)
                torch.cuda.synchronize(); rec["stages"]["propagate_s"] = time.time() - t_start
                t1 = time.time()
                lp = sound_lp_batch(out, rset, K, spec.disjuncts, a.final_iters, 0.05)
                torch.cuda.synchronize(); rec["stages"]["terminal_lp_s"] = time.time() - t1
                ups = [d["upper"] for d in lp]
                open_ids = sorted([i for i, v in enumerate(ups) if v >= -1e-4], key=lambda i: -ups[i])
                rec.update({"K": K, "unstable": sum(int(p["idx"].numel()) for p in phases), "lp_worst": max(ups),
                            "lp_open": len(open_ids)})
                nzi = e.input_factor_index.cpu().numpy(); center = (lb + ub) / 2; rad = (ub - lb) / 2

                def check(w):
                    xi = np.zeros(n_in); xi[nzi] = np.asarray(w)[: nzi.size]
                    x64 = np.clip(center + rad * xi, lb, ub)
                    x = x64.astype(np.float32)
                    y = s.run(None, {iname: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
                    hit = next((k for k, atoms in enumerate(spec.disjuncts) if all(float(aa @ y) <= bb for aa, bb in atoms)), None)
                    return hit, (x, x64)
                outcome = "CERT" if not open_ids else "UNKNOWN"
                witness = None
                t2 = time.time()
                for i in open_ids:                       # LP box-maximiser candidates (cheap)
                    hit, x = check(lp[i]["cand"])
                    if hit is not None:
                        witness = ("lp_box_maximiser", i, hit, x); break
                rec["stages"]["decode_s"] = time.time() - t2
                milp_log = {}
                if witness is None and open_ids:
                    certified_all = True
                    for i in open_ids:
                        remaining = a.budget - (time.time() - t0)
                        if remaining <= 1:
                            certified_all = False; outcome = "TIMEOUT"; break
                        d = lp[i]
                        A_, b_ = rset.dense(K)
                        mres = plan_milp_highs(A_, b_, phases, K, d["g"], d["cc"], d["pad"], d["extra"], 1, 99, remaining)
                        milp_log[str(i)] = {k: v for k, v in mres.items() if k != "incumbent_w"}
                        if mres["incumbent_w"] is not None:
                            hit, x = check(mres["incumbent_w"])
                            if hit is not None:
                                witness = ("milp_incumbent", i, hit, x); break
                        if not mres["excluded"]:
                            certified_all = False
                            if time.time() - t0 >= a.budget - 1:
                                outcome = "TIMEOUT"
                            break
                    if witness is None and certified_all:
                        outcome = "CERT"
                rec["milp"] = milp_log
                if witness is not None:
                    outcome = "ADV"
                    src, i, hit, (x, x64) = witness
                    rec.update({"witness_source": src, "witness_from_disjunct": i, "witness_hits_disjunct": hit,
                                "witness_x_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()})
                    np.save(f"{os.path.splitext(a.out)[0]}_x_{r['row_index']}.npy", x)
                    np.save(f"{os.path.splitext(a.out)[0]}_x64_{r['row_index']}.npy", x64)
                rec["outcome"] = outcome
            except Exception as ex:
                rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
            rec["wall_s"] = time.time() - t0
            if rec["outcome"] in ("CERT", "ADV") and rec["wall_s"] > a.budget:
                rec["over_budget_outcome"] = rec["outcome"]; rec["outcome"] = "TIMEOUT"
            fout.write(json.dumps(rec, default=float) + "\n"); fout.flush()
            print(r["row_index"], r["baseline_verdict"], rec["outcome"], round(rec.get("lp_worst", float("nan")), 4),
                  {k: round(v, 1) for k, v in rec["stages"].items()}, round(rec["wall_s"], 1), rec.get("error", ""), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
