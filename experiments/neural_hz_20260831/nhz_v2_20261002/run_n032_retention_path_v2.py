"""N032: retention sample with candidate path v2 (nhz_path_v2) and engine n007 (nhz_sound_mp2).

Derived from N021 v2: retention sample of the N016 rigorous path on baseline-SOLVED formal rows.

Rows: a fixed random sample (seed 20261002) of rows that the frozen 1870 overlay
lists as CERT or ADV, per family.  Uses N016's solve_box unchanged.  Reports for
each row baseline verdict versus new outcome; CERT-vs-ADV disagreements are
flagged as soundness conflicts (one of the two results is wrong).
"""
import argparse, csv, json, os, random, sys, time, hashlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_n016v2_formal_sound_pipeline import ROOT
from nhz_path_v2 import solve_box_v2 as solve_box, PATH_VERSION
from nhz_engine import parse_vnnlib
from nhz_sound_mp2 import MP_VERSION as SOUND_VERSION, SoundEngineMP2
OVERLAY = "/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True); ap.add_argument("--per-family", type=int, default=20); ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--out", required=True); ap.add_argument("--mem-fraction", type=float, default=0.25)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    csv.field_size_limit(10 ** 9)
    rows = [r for r in csv.DictReader(open(OVERLAY)) if r["raw_verdict"] in ("CERT", "ADV")]
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    rng = random.Random(a.seed)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    engines, sess = {}, {}
    with os.fdopen(fd, "w") as fout:
        for fam in a.families.split(","):
            fr = [r for r in rows if r["benchmark"] == fam]
            sample = rng.sample(fr, min(a.per_family, len(fr)))
            for r in sample:
                mp = os.path.normpath(os.path.join(ROOT, fam, r["onnx"])); sp = os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"]))
                budget = float(r["csv_timeout"]); rec = {"engine": SOUND_VERSION, "path": PATH_VERSION, "family": fam, "iid": r["iid"], "baseline": r["raw_verdict"],
                                                         "budget_s": budget, "boxes": []}
                t0 = time.time()
                try:
                    if mp not in engines:
                        engines.clear(); sess.clear(); torch.cuda.empty_cache()
                        engines[mp] = SoundEngineMP2(mp, "cuda")
                        sess[mp] = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                    e = engines[mp]; s = sess[mp]; iname = s.get_inputs()[0].name
                    n_in = int(np.prod(e.input_shape))
                    o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
                    spec = parse_vnnlib(sp, n_in, int(o0.c.numel()))
                    outcome = "CERT"; deadline = t0 + budget
                    for lb, ub in spec.boxes:
                        oc, info, wit = solve_box(e, s, iname, spec, lb, ub, deadline)
                        rec["boxes"].append(info)
                        if oc == "ADV":
                            outcome = "ADV"; rec["witness_source"] = wit[0]; break
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
                print(fam, r["iid"], r["raw_verdict"], "->", rec["outcome"], round(rec["wall_s"], 1), "CONFLICT" if rec["conflict"] else "", rec.get("error", ""), flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
