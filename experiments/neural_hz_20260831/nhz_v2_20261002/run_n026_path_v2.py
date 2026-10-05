"""N026: candidate path v2 (nhz_path_v2) on selected formal rows (overlay iids or manifest ids)."""
import argparse, csv, json, os, sys, time, hashlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import parse_vnnlib
from nhz_sound_mp import MP_VERSION, SoundEngineMP
from nhz_path_v2 import PATH_VERSION, solve_box_v2
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
OVERLAY = "/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="comma list family:iid (overlay names)")
    ap.add_argument("--mem-fraction", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    csv.field_size_limit(10 ** 9)
    ov = {(r["benchmark"], r["iid"]): r for r in csv.DictReader(open(OVERLAY))}
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    engines, sess = {}, {}
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for item in a.items.split(","):
            fam, iid = item.rsplit(":", 1); r = ov[(fam, iid)]
            mp = os.path.normpath(os.path.join(ROOT, fam, r["onnx"])); sp = os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"]))
            budget = float(r["csv_timeout"])
            rec = {"engine": MP_VERSION, "path": PATH_VERSION, "family": fam, "iid": iid, "baseline": r["raw_verdict"], "budget_s": budget, "boxes": []}
            t0 = time.time()
            try:
                if mp not in engines:
                    engines[mp] = SoundEngineMP(mp, "cuda"); sess[mp] = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                e = engines[mp]; s = sess[mp]; iname = s.get_inputs()[0].name
                n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
                spec = parse_vnnlib(sp, n_in, int(o0.c.numel()))
                outcome = "CERT"
                for lb, ub in spec.boxes:
                    oc, info, wit = solve_box_v2(e, s, iname, spec, lb, ub, t0 + budget)
                    rec["boxes"].append(info)
                    if oc == "ADV":
                        outcome = "ADV"; rec["witness_source"] = wit[0]
                        np.save(f"{os.path.splitext(a.out)[0]}_x64_{fam}_{iid}.npy", wit[3][1]); break
                    if oc != "CERT":
                        outcome = oc; break
                rec["outcome"] = outcome
            except Exception as ex:
                rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
            rec["wall_s"] = time.time() - t0
            if rec["outcome"] in ("CERT", "ADV") and rec["wall_s"] > budget:
                rec["over_budget_outcome"] = rec["outcome"]; rec["outcome"] = "TIMEOUT"
            rec["conflict"] = {rec["outcome"], rec["baseline"]} == {"CERT", "ADV"}
            fout.write(json.dumps(rec, default=float) + "\n"); fout.flush()
            print(fam, iid, rec["baseline"], "->", rec["outcome"], round(rec["wall_s"], 1), {k: (v.get("status"), round(v["upper"], 4) if v.get("upper") is not None else None, round(v["wall_s"], 1)) for b in rec["boxes"] for k, v in b.get("milp", {}).items()}, rec.get("error", "")[:100], flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
