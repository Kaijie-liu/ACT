"""N029: candidate path v2 (fixed cascade, mixed-precision sound engine) over the formal
543-row unsolved manifest, family order given on the command line.  Capability probe;
saves float32 and float64 witness points; budget = official timeout."""
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
from nhz_path_v2 import PATH_VERSION, solve_box_v2  # noqa: E402
from nhz_sound_mp import MP_VERSION, SoundEngineMP  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
MANIFEST = "/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/manifests/formal_unsolved_structure_manifest_v1.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--mem-fraction", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    allrows = json.load(open(MANIFEST))["instances"]
    engines, sess = {}, {}
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for fam in a.families.split(","):
            for r in [x for x in allrows if x["family"] == fam]:
                mp = os.path.join(ROOT, r["model"]["relative_path"]); sp = os.path.join(ROOT, r["spec"]["relative_path"])
                budget = float(r["source"]["timeout_seconds"])
                rec = {"engine": MP_VERSION, "path": PATH_VERSION, "row_identity": r["row_identity"], "family": fam,
                       "baseline": r["verdict"], "budget_s": budget, "boxes": []}
                t0 = time.time()
                try:
                    if mp not in engines:
                        assert sha256(mp) == r["model"]["sha256"], "model hash"
                        engines.clear(); sess.clear(); torch.cuda.empty_cache()
                        engines[mp] = SoundEngineMP(mp, "cuda")
                        sess[mp] = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                    assert sha256(sp) == r["spec"]["sha256"], "spec hash"
                    e = engines[mp]; s = sess[mp]; iname = s.get_inputs()[0].name
                    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
                    spec = parse_vnnlib(sp, n_in, int(o0.c.numel()))
                    outcome = "CERT"
                    for lb, ub in spec.boxes:
                        oc, info, wit = solve_box_v2(e, s, iname, spec, lb, ub, t0 + budget)
                        rec["boxes"].append(info)
                        if oc == "ADV":
                            outcome = "ADV"; rec["witness_source"] = wit[0]
                            tag = r["row_identity"].replace(":", "_")
                            np.save(f"{os.path.splitext(a.out)[0]}_x_{tag}.npy", wit[3][0])
                            np.save(f"{os.path.splitext(a.out)[0]}_x64_{tag}.npy", wit[3][1])
                            break
                        if oc != "CERT":
                            outcome = oc; break
                    rec["outcome"] = outcome
                except Exception as ex:
                    rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
                rec["wall_s"] = time.time() - t0
                if rec["outcome"] in ("CERT", "ADV") and rec["wall_s"] > budget:
                    rec["over_budget_outcome"] = rec["outcome"]; rec["outcome"] = "TIMEOUT"
                fout.write(json.dumps(rec, default=float) + "\n"); fout.flush()
                print(r["row_identity"], r["verdict"], "->", rec["outcome"], round(rec["wall_s"], 1),
                      rec.get("witness_source", ""), rec.get("error", "")[:120], flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
