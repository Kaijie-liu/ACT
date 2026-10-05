"""E0 witness audit for N061-style runners (witness files x_<family>_<row>.npy): reuses the
independent S-expression reader and the S1/S2 checks of audit_n022_witness_replay.py; rows are
located through the family's instances.csv (model/spec paths) and the E0 evidence ledger."""
import argparse, csv, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_n022_witness_replay import check, ROOT
EVID = "/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/evidence"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--glob", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inst = {}; base = {}
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for p in sorted(glob.glob(a.glob)):
            m = re.search(r"/x_(cifar100_2024|tinyimagenet_2024)_(\d+)\.npy$", p)
            if not m:
                continue
            fam, row = m.group(1), int(m.group(2))
            if fam not in inst:
                inst[fam] = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
                base[fam] = {r["row_index"]: r["baseline_verdict"] for r in json.load(open(f"{EVID}/{fam}_evidence_baseline_v2.json"))["rows"]}
            o, s = inst[fam][row][:2]
            mp, sp = f"{ROOT}/{fam}/{o}", f"{ROOT}/{fam}/{s}"
            x = np.load(p)
            in_ok, out_ok = check(mp, sp, x, "S2"); s1_in, _ = check(mp, sp, x, "S1")
            rec = {"witness_file": os.path.basename(p), "family": fam, "row_index": row, "baseline": base[fam].get(row),
                   "unsafe_reached": out_ok, "S1_valid": bool(s1_in and out_ok), "S2_valid": bool(in_ok and out_ok)}
            fout.write(json.dumps(rec) + "\n"); print(rec, flush=True)


if __name__ == "__main__":
    main()
