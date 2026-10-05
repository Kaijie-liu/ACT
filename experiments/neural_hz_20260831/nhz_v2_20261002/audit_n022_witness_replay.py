"""N022 independent witness replay (v3: reports two input semantics).

S1 (baseline ADV gate, hz_full_worker._is_cex): the witness is a real point x64
inside the box (here x64 = clip(float64(x32), lb, ub) in exact rationals); the
network is ONNX Runtime evaluated on float32(x64).  Accepted iff float32(x64)
equals the stored x32 and the unsafe output assertions hold exactly.
S2 (strict): the float32 point x32 itself satisfies every input assertion.
Output assertions are evaluated in exact rationals on ORT float32 outputs.

For every saved witness x (*.npy written by N015/N016), recompute the network
output with ONNX Runtime (CPU, single thread) and evaluate the ORIGINAL VNNLIB
file by direct recursive evaluation of its S-expressions (assert/and/or/<=/>=/
</>/+/-/*), at literal zero tolerance:
  * every input assertion must hold for x (box membership), and
  * the conjunction of all output assertions must hold for y (unsafe reached).
Shares no code with the pipeline's parser.
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from fractions import Fraction

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
MANIFEST = "/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/manifests/formal_unsolved_structure_manifest_v1.json"


def parse(text):
    text = re.sub(r";[^\n]*", "", text)
    toks = re.findall(r"\(|\)|[^\s()]+", text)
    pos = 0

    def rd():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t != "(":
            return t
        lst = []
        while toks[pos] != ")":
            lst.append(rd())
        pos += 1
        return lst
    out = []
    while pos < len(toks):
        out.append(rd())
    return out


def ev(e, env):
    if isinstance(e, str):
        if e in env:
            return env[e]
        return Fraction(e)
    h = e[0]
    if h == "and":
        return all(ev(x, env) for x in e[1:])
    if h == "or":
        return any(ev(x, env) for x in e[1:])
    if h in ("<=", ">=", "<", ">"):
        a, b = ev(e[1], env), ev(e[2], env)
        return {"<=": a <= b, ">=": a >= b, "<": a < b, ">": a > b}[h]
    if h == "+":
        return sum(ev(x, env) for x in e[1:])
    if h == "-":
        v = [ev(x, env) for x in e[1:]]
        return -v[0] if len(v) == 1 else v[0] - sum(v[1:])
    if h == "*":
        p = Fraction(1)
        for x in e[1:]:
            p *= ev(x, env)
        return p
    raise ValueError(h)


def uses(e, prefix):
    if isinstance(e, str):
        return e.startswith(prefix)
    return any(uses(x, prefix) for x in e)


def check(onnx_path, spec_path, x, semantics="S2"):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    y = s.run(None, {inp.name: x.reshape(shape).astype(np.float32)})[0].reshape(-1).astype(np.float64)
    env = {f"X_{i}": Fraction(float(v)) for i, v in enumerate(x.reshape(-1).astype(np.float32))}
    env.update({f"Y_{i}": Fraction(float(v)) for i, v in enumerate(y.astype(np.float32))})
    forms = [f[1] for f in parse(open(spec_path).read()) if isinstance(f, list) and f and f[0] == "assert"]
    in_forms = [f for f in forms if uses(f, "X_") and not uses(f, "Y_")]
    out_forms = [f for f in forms if uses(f, "Y_")]
    in_ok = all(ev(f, env) for f in in_forms)
    out_ok = all(ev(f, env) for f in out_forms)
    if semantics == "S1":
        # S1: does a real point in the box round to the same float32 vector?
        # Exact per-coordinate test for box atoms of the form (<= X_i c) / (>= X_i c).
        lo = {}; hi = {}
        for f in in_forms:
            if isinstance(f, list) and len(f) == 3 and isinstance(f[1], str) and f[1].startswith("X_"):
                k = f[1]; v = Fraction(f[2])
                if f[0] == "<=":
                    hi[k] = min(hi.get(k, v), v)
                elif f[0] == ">=":
                    lo[k] = max(lo.get(k, v), v)
        x32 = x.reshape(-1).astype(np.float32)
        s1 = True
        for i, v in enumerate(x32):
            k = f"X_{i}"; fv = Fraction(float(v))
            cl = min(max(fv, lo.get(k, fv)), hi.get(k, fv))      # exact clip into the box
            # the real point cl must round (nearest float32) to v
            if np.float32(float(cl)) != v and cl != fv:
                # conservative: float(cl) may itself round; require |cl - v| < half ulp
                ulp = Fraction(float(np.spacing(np.float32(abs(v)))))
                if abs(cl - fv) * 2 >= ulp:
                    s1 = False; break
        return s1, out_ok
    return in_ok, out_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = {r["row_identity"]: r for r in json.load(open(MANIFEST))["instances"]}
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for p in sorted(glob.glob(a.glob)):
            base = os.path.basename(p)
            m = re.search(r"_x_(.+)\.npy$", base)
            key = m.group(1)
            if key.isdigit():                      # E0 file: family from the file name
                fam = "cifar100_2024" if "cifar" in p else "tinyimagenet_2024"
                inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
                o, s = inst[int(key)][:2]
                mp, sp = f"{ROOT}/{fam}/{o}", f"{ROOT}/{fam}/{s}"; rid = f"{fam}:{key}"
            else:
                rid = key.replace("_", ":", 1) if ":" not in key else key
                fam_key = rid.rsplit("_", 1)
                r = man.get(rid) or man.get(key.rsplit("_", 1)[0] + ":" + key.rsplit("_", 1)[1])
                rid = r["row_identity"]
                mp, sp = os.path.join(ROOT, r["model"]["relative_path"]), os.path.join(ROOT, r["spec"]["relative_path"])
            x = np.load(p)
            in_ok, out_ok = check(mp, sp, x, "S2")
            s1_in, _ = check(mp, sp, x, "S1")
            rec = {"witness_file": base, "row": rid, "unsafe_reached": out_ok,
                   "S1_valid": bool(s1_in and out_ok), "S2_valid": bool(in_ok and out_ok)}
            fout.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
