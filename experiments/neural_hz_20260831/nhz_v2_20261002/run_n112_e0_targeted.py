"""N112 E0 targeted runner (path v7.2, --rows). Derived from N100: E0 single-path replay with the N039 v13 candidate (path v7.0 domain-only, engine n020). Derived from
N092: E0 single-path replay with the N039 v12 candidate (engine n019, path v6.6). Derived from
N068: E0 single-path replay with the formal candidate of N039 v7 (engine n017 + path v6.1), all 400 rows.
(derived from N061)

Universe, hashes and budgets as N015 (evidence ledgers v2, 100 s per row).  Per row: one
call of solve_box_v5 per input box (E0 specs have one box), the same code the formal
replay N039 v6 runs.  ADV acceptance: strict float32 point after inward rounding (S2) first,
baseline semantics S1 second; both recorded.  Witnesses saved for the N022 audit."""
import argparse, csv, gc, hashlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import parse_vnnlib
from nhz_path_v7_2 import PATH_VERSION, solve_box_v7 as solve_box_v5
from nhz_sound_v20 import V20_VERSION as V11_VERSION, SoundEngineV20 as SoundEngineV11b
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
    ap.add_argument("--rows", default="")
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
            want = {int(x) for x in a.rows.split(",")} if a.rows else None
            for r in ledger["rows"]:
                if want is not None and r["row_index"] not in want:
                    continue
                onnx_rel, spec_rel, _ = inst[r["row_index"]]
                mp = f"{ROOT}/{fam}/{onnx_rel}"; sp_ = f"{ROOT}/{fam}/{spec_rel}"
                rec = {"engine": V11_VERSION, "path": PATH_VERSION, "family": fam, "row_index": r["row_index"],
                       "baseline": r["baseline_verdict"], "budget_s": a.budget, "boxes": []}
                t0 = time.time()
                for attempt in (0, 1):
                    rec["boxes"] = []
                    try:
                        if mp not in engines:
                            assert sha256(mp) == r["model_sha256"], "model hash"
                            engines.clear(); sess.clear(); gc.collect(); torch.cuda.empty_cache()
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
                        rec["outcome"] = outcome; rec.pop("error", None)
                        break
                    except torch.OutOfMemoryError as ex:
                        rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
                        engines.clear(); sess.clear(); gc.collect(); torch.cuda.empty_cache(); rec["oom_retry"] = attempt + 1
                        free, total = torch.cuda.mem_get_info(); frac2 = min(0.5, a.mem_fraction + 0.8 * free / total)
                        if frac2 > a.mem_fraction:
                            torch.cuda.set_per_process_memory_fraction(frac2); rec["oom_retry_fraction"] = frac2
                        if time.time() - t0 > a.budget - 1:
                            break
                    except Exception as ex:
                        rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
                        break
                rec["wall_s"] = time.time() - t0
                if rec["outcome"] in ("CERT", "ADV") and rec["wall_s"] > a.budget:
                    rec["over_budget_outcome"] = rec["outcome"]; rec["outcome"] = "TIMEOUT"
                rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                fout.write(json.dumps(rec, default=float) + "\n"); fout.flush()
                print(fam, r["row_index"], r["baseline_verdict"], "->", rec["outcome"], round(rec["wall_s"], 1), rec.get("error", "")[:100], flush=True)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.set_per_process_memory_fraction(a.mem_fraction)


if __name__ == "__main__":
    main()
