"""N116 targeted runner (path v7.2 + engine n021); copy of the N099 targeted runner = N039 v13 runner with --rows.
N039 v13 runner: path v7.0 (domain-only verdicts: no LP box maximiser, no centre point, no seeded MIP
start, no stage W), engine n020 (LP-tightening time share).  Otherwise as the v12 runner.
N039 v12 runner: engine n019 (pooling), path v6.6 (MIP start); an OOM retry raises the per-process
memory fraction up to 0.5 when the device has room (resource handling only), then restores it.
N039 v7 runner: path v6.1 (terminal v8 smooth rows, stage F, exact-plan target rule), engine n017.
N039 v4 runner: path v5 (terminal v7, inward witness rounding), engine n009.2.
N039 v3: as N039 v2 but path v4.3 and engine n009.2 (bounded polishing).
N039 v2: as N039 v1 but with candidate path v4.2 (seed portfolio + early stop + callback interrupt).

N039: single-path replay of the formal 13-family universe (2413 rows) with candidate
path v3 (nhz_path_v3) and rigorous engine n009 (nhz_sound_v9).

Rows and budgets come from the composite baseline authority recorded in BASELINE_LOCK.md:
the 2,213-row 12-family overlay (csv_timeout, raw_verdict) and the 200-row strict ViT CSV
(timeout_sec, strict_status).  One configuration for every row; workers only partition
the row set by family.  Each output line records the baseline verdict next to the new
outcome.  ADV acceptance uses S1 (baseline semantics); witnesses are saved for audit.
"""
import argparse
import gc
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import parse_vnnlib  # noqa: E402
from nhz_path_v7_2 import PATH_VERSION, solve_box_v7 as solve_box_v3  # noqa: E402
from nhz_sound_v21 import V21_VERSION as V9_VERSION, SoundEngineV21 as SoundEngineV9  # noqa: E402  (engine n021)

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
OVERLAY = "/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv"
VIT = "/data1/Kane/HyZor/vit_hz_legacy1_100s_20260826/consolidated_strict_100s.csv"
VIT_MAP = {"CERTIFIED": "CERT", "UNKNOWN": "UNKNOWN", "TIMEOUT": "TIMEOUT"}


def universe():
    csv.field_size_limit(10 ** 9)
    rows = []
    for r in csv.DictReader(open(OVERLAY)):
        rows.append({"family": r["benchmark"], "iid": int(r["iid"]), "onnx": r["onnx"], "vnnlib": r["vnnlib"],
                     "timeout": float(r["csv_timeout"]), "baseline": r["raw_verdict"]})
    for r in csv.DictReader(open(VIT)):
        rows.append({"family": "vit_2023", "iid": int(r["iid"]), "onnx": r["onnx"], "vnnlib": r["vnnlib"],
                     "timeout": float(r["timeout_sec"]), "baseline": VIT_MAP.get(r["strict_status"], r["strict_status"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--mem-fraction", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    fams = a.families.split(",")
    rows = [r for r in universe() if r["family"] in fams]
    if a.rows:
        want = set(tuple(x.split(":")) for x in a.rows.split(","))
        rows = [r for r in rows if (r["family"], str(r["iid"])) in want]
    rows.sort(key=lambda r: (fams.index(r["family"]), r["iid"]))
    engines, sess = {}, {}
    wdir = os.path.splitext(a.out)[0] + "_witness"
    os.makedirs(wdir, exist_ok=True)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for r in rows:
            mp = os.path.normpath(os.path.join(ROOT, r["family"], r["onnx"]))
            sp = os.path.normpath(os.path.join(ROOT, r["family"], r["vnnlib"]))
            rec = {"engine": V9_VERSION, "path": PATH_VERSION, **r, "boxes": []}
            t0 = time.time()
            for attempt in (0, 1):
                rec["boxes"] = []
                try:
                    if mp not in engines:
                        engines.clear(); sess.clear(); gc.collect(); torch.cuda.empty_cache()
                        engines[mp] = SoundEngineV9(mp, "cuda")
                        sess[mp] = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                    e = engines[mp]; s = sess[mp]; iname = s.get_inputs()[0].name
                    n_in = int(np.prod(e.input_shape))
                    o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
                    spec = parse_vnnlib(sp, n_in, int(o0.c.numel()))
                    outcome = "CERT"
                    for lb, ub in spec.boxes:
                        oc, info, wit = solve_box_v3(e, s, iname, spec, lb, ub, t0 + r["timeout"])
                        rec["boxes"].append(info)
                        if oc == "ADV":
                            outcome = "ADV"; rec["witness_source"] = wit[0]
                            tag = f"{r['family']}_{r['iid']}"
                            np.save(f"{wdir}/x_{tag}.npy", wit[3][0]); np.save(f"{wdir}/x64_{tag}.npy", wit[3][1])
                            rec["witness_x_sha256"] = hashlib.sha256(np.ascontiguousarray(wit[3][0]).tobytes()).hexdigest()
                            break
                        if oc != "CERT":
                            outcome = oc; break
                    rec["outcome"] = outcome
                    rec.pop("error", None)
                    break
                except torch.OutOfMemoryError as ex:
                    # resource failure (not a verdict): release everything, retry once within the same deadline
                    rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
                    engines.clear(); sess.clear(); gc.collect(); torch.cuda.empty_cache()
                    rec["oom_retry"] = attempt + 1
                    free, total = torch.cuda.mem_get_info()
                    frac2 = min(0.5, a.mem_fraction + 0.8 * free / total)
                    if frac2 > a.mem_fraction:
                        torch.cuda.set_per_process_memory_fraction(frac2); rec["oom_retry_fraction"] = frac2
                    if time.time() - t0 > r["timeout"] - 1:
                        break
                except Exception as ex:
                    rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
                    break
            rec["wall_s"] = time.time() - t0
            if rec["outcome"] in ("CERT", "ADV") and rec["wall_s"] > r["timeout"]:
                rec["over_budget_outcome"] = rec["outcome"]; rec["outcome"] = "TIMEOUT"
            rec["conflict"] = {rec["outcome"], rec["baseline"]} == {"CERT", "ADV"}
            fout.write(json.dumps(rec, default=float) + "\n"); fout.flush()
            print(r["family"], r["iid"], r["baseline"], "->", rec["outcome"], round(rec["wall_s"], 1),
                  "CONFLICT" if rec["conflict"] else "", rec.get("error", "")[:100], flush=True)
            gc.collect(); torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(a.mem_fraction)      # restore the co-scheduling cap


if __name__ == "__main__":
    main()
