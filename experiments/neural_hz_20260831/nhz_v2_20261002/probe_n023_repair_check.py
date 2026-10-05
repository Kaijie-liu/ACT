import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from witness_util import to_box_f32
from nhz_engine import parse_vnnlib
import audit_n022_witness_replay as A
ROOT = A.ROOT
man = {r["row_identity"]: r for r in json.load(open(A.MANIFEST))["instances"]}
for p in sorted(glob.glob("results/n016_formal_v1_x_*.npy")):
    key = p.split("_x_")[1][:-4]; rid = key.rsplit("_", 1)[0] + ":" + key.rsplit("_", 1)[1]
    r = man[rid]; mp = os.path.join(ROOT, r["model"]["relative_path"]); sp = os.path.join(ROOT, r["spec"]["relative_path"])
    x = np.load(p).astype(np.float64)
    import onnxruntime as ort
    n = x.size; spec = parse_vnnlib(sp, n, 1000)
    lb, ub = spec.boxes[0]
    out_lo = int((x < lb).sum()); out_hi = int((x > ub).sum())
    gap = float(max((lb - x).max(), (x - ub).max()))
    x2 = to_box_f32(x, lb, ub)
    if x2 is None:
        print(rid, "no float32 point inside box"); continue
    in_ok, out_ok = A.check(mp, sp, x2)
    print(rid, "coords below lb", out_lo, "above ub", out_hi, "max excess", gap, "-> repaired in_box", in_ok, "unsafe", out_ok)
