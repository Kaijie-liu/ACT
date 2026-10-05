"""N017 independent audit of CERT outcomes (falsification test of the implementation).

For every CERT in the given result files, draw `--samples` uniform inputs from
each input box plus all 2^min(n,10) corner-pattern-free probes (centre and the
box corners of the first coordinates are NOT used; only uniform samples), run
ONNX Runtime and check that no sample satisfies the unsafe VNNLIB disjunction.
A violation proves the CERT wrong (an implementation defect).  Passing is only
weak evidence; it never creates a verdict.  The parser used here is the same
module as the pipeline, so spec-parsing defects are checked separately by a
re-parse with the independent reader below (numeric atoms only).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import parse_vnnlib  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"


def independent_box(path, n_in):
    """Independent regex reader of simple input bounds (single-box specs only)."""
    txt = open(path).read()
    lb = np.full(n_in, -np.inf); ub = np.full(n_in, np.inf)
    for op, k, v in re.findall(r"\(\s*(<=|>=)\s+X_(\d+)\s+([-+0-9.eE]+)\s*\)", txt):
        k = int(k); v = float(v)
        if op == "<=":
            ub[k] = min(ub[k], v)
        else:
            lb[k] = max(lb[k], v)
    return lb, ub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import onnxruntime as ort
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for f in a.files:
            for line in open(f):
                r = json.loads(line)
                if r.get("outcome") != "CERT":
                    continue
                if "model" in r:
                    mp = os.path.join(ROOT, r["model"]); key = r["row_identity"]
                    import json as _j
                    man = _j.load(open("/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/manifests/formal_unsolved_structure_manifest_v1.json"))["instances"]
                    sp = os.path.join(ROOT, next(x for x in man if x["row_identity"] == key)["spec"]["relative_path"])
                else:
                    import csv
                    fam = r["family"]; inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
                    o, s = inst[r["row_index"]][:2]; mp = f"{ROOT}/{fam}/{o}"; sp = f"{ROOT}/{fam}/{s}"; key = f"{fam}:{r['row_index']}"
                sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
                inp = sess.get_inputs()[0]; shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                n_in = int(np.prod(shape))
                y0 = sess.run(None, {inp.name: np.zeros(shape, np.float32)})[0].reshape(-1)
                spec = parse_vnnlib(sp, n_in, y0.size)
                rng = np.random.default_rng(12345)
                viol = 0; min_slack = np.inf
                box_agree = True
                if len(spec.boxes) == 1:
                    il, iu = independent_box(sp, n_in)
                    box_agree = bool(np.array_equal(il, spec.boxes[0][0]) and np.array_equal(iu, spec.boxes[0][1]))
                for lb, ub in spec.boxes:
                    X = rng.uniform(lb, ub, size=(a.samples, n_in)).astype(np.float32)
                    bs = 256 if not (isinstance(inp.shape[0], int) and inp.shape[0] == 1) else 1
                    for s0 in range(0, a.samples, bs):
                        Y = sess.run(None, {inp.name: X[s0:s0 + bs].reshape([-1] + shape[1:])})[0].reshape(-1, y0.size).astype(np.float64)
                        for atoms in spec.disjuncts:
                            sl = np.max(np.stack([Y @ aa - bb for aa, bb in atoms]), 0)   # <=0 means unsafe
                            viol += int((sl <= 0).sum()); min_slack = min(min_slack, float(sl.min()))
                rec = {"row": key, "samples": a.samples * len(spec.boxes), "violations": viol,
                       "min_unsafe_slack": min_slack, "independent_box_agrees": box_agree}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(rec, flush=True)


if __name__ == "__main__":
    main()
