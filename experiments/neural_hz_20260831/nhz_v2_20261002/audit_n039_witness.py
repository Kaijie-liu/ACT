"""Witness audit for N039 replays: reuses the independent S-expression reader and the S1/S2
checks of audit_n022_witness_replay.py (exact rationals, ORT CPU single thread), with rows
located through the composite baseline universe (overlay CSV + strict ViT CSV) by family/iid."""
import argparse, csv, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_n022_witness_replay import check, ROOT
from run_n039_full_replay import universe


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--glob", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    U = {(r["family"], r["iid"]): r for r in universe()}
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for p in sorted(glob.glob(a.glob)):
            m = re.search(r"/x_(.+)_(\d+)\.npy$", p)
            if not m:
                continue
            fam, iid = m.group(1), int(m.group(2)); r = U[(fam, iid)]
            mp = os.path.normpath(os.path.join(ROOT, fam, r["onnx"])); sp = os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"]))
            x = np.load(p)
            in_ok, out_ok = check(mp, sp, x, "S2"); s1_in, _ = check(mp, sp, x, "S1")
            rec = {"witness_file": os.path.basename(p), "family": fam, "iid": iid, "baseline": r["baseline"],
                   "unsafe_reached": out_ok, "S1_valid": bool(s1_in and out_ok), "S2_valid": bool(in_ok and out_ok)}
            fout.write(json.dumps(rec) + "\n"); print(rec, flush=True)


if __name__ == "__main__":
    main()
