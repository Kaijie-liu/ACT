"""N061: E0 single-path replay with the formal candidate (engine n011.2 + path v5.5), all 400 rows.

Universe, hashes and budgets as N015 (evidence ledgers v2, 100 s per row).  Per row: one
call of solve_box_v5 per input box (E0 specs have one box), the same code the formal
replay N039 v6 runs.  ADV acceptance: strict float32 point after inward rounding (S2) first,
baseline semantics S1 second; both recorded.  Witnesses saved for the N022 audit."""
import argparse, csv, hashlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import parse_vnnlib
from nhz_path_v5 import PATH_VERSION, solve_box_v5
from nhz_sound_v11b import V11_VERSION, SoundEngineV11b
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
    ap.add_argument("--families", default="cifar100_2024,tinyimagenet_2024"); ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--mem-fraction", type=float, default=0.3); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    wdir = os.path.splitext(a.out)[0] + "_witness"; os.makedirs(wdir, exist_ok=True)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for fam in a.families.split(","):
            ledger = json.load(open(f"{EVID}/{fam}_evidence_baseline_v2.json"))
            inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
            engines, sess = {}, {}
            for r in ledger["rows"]:
                onnx_rel, spec_rel, _ = inst[r["row_index"]]
                mp = f"{ROOT}/{fam}/{onnx_rel}"; sp_ = f"{ROOT}/{fam}/{spec_rel}"
                rec = {"engine": V11_VERSION, "path": PATH_VERSION, "family": fam, "row_index": r["row_index"],
                       "baseline": r["baseline_verdict"], "budget_s": a.budget, "boxes": []}
                t0 = time.time()
                try:
                    if mp not in engines:
                        assert sha256(mp) == r["model_sha256"], "model hash"
                        engines.clear(); sess.clear(); torch.cuda.empty_cache()
                        engines[mp] = SoundEngineV11b(mp, "cuda")
                        sess[mp] = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                    assert sha256(sp_) == r["spec_sha256"], "spec hash"
                    e = engines[mp]; s = sess[mp]; iname = s.get_inputs()[0].name
                    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
                    spec = parse_vnnlib(sp_, n_in, int(o0.c.numel()))
                    outcome = "CERT"
                    for lb, ub in spec.boxes:
                        oc, info, wit = solve_box_v5(e, s, iname, spec, lb, ub, t0 + a.budget)
                        rec["boxes"].append(info)
                        if oc == "ADV":
                            outcome = "ADV"; rec["witness_source"] = wit[0]
                            tag = f"{fam}_{r['row_index']}"
                            np.save(f"{wdir}/x_{tag}.npy", wit[3][0]); np.save(f"{wdir}/x64_{tag}.npy", wit[3][1])
                            break
                        if oc != "CERT":
                            outcome = oc; break
                    rec["outcome"] = outcome
                except Exception as ex:
                    rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
                rec["wall_s"] = time.time() - t0
                if rec["outcome"] in ("CERT", "ADV") and rec["wall_s"] > a.budget:
                    rec["over_budget_outcome"] = rec["outcome"]; rec["outcome"] = "TIMEOUT"
                rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                fout.write(json.dumps(rec, default=float) + "\n"); fout.flush()
                print(fam, r["row_index"], r["baseline_verdict"], "->", rec["outcome"], round(rec["wall_s"], 1), rec.get("error", "")[:100], flush=True)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
