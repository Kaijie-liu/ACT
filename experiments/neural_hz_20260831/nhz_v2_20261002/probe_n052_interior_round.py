"""N052: interior float32 rounding of S1-only witnesses.  For each saved x64 (real point in the
box), take x32 = float32 rounding toward the box interior (nextafter inward when the nearest
float32 lies outside), then run the independent S2 check of audit_n022 (exact rationals)."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_n022_witness_replay import check, ROOT
from nhz_engine import parse_vnnlib
from run_n039_full_replay import universe
U = {(r["family"], r["iid"]): r for r in universe()}
out = open(sys.argv[2], "a")
for p in sorted(glob.glob(sys.argv[1])):
    m = re.search(r"/x64_(.+)_(\d+)\.npy$", p)
    if not m:
        continue
    fam, iid = m.group(1), int(m.group(2)); r = U[(fam, iid)]
    mp = os.path.normpath(os.path.join(ROOT, fam, r["onnx"])); sp = os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"]))
    x64 = np.load(p).astype(np.float64).reshape(-1)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s_ = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
    ishape = [d if isinstance(d, int) and d > 0 else 1 for d in s_.get_inputs()[0].shape]
    n_out = int(np.prod(s_.run(None, {s_.get_inputs()[0].name: x64.astype(np.float32).reshape(ishape)})[0].shape))
    sp_ = parse_vnnlib(sp, x64.size, n_out)   # pipeline reader used for the box only; the check is independent
    lb, ub = sp_.boxes[0]
    x32 = x64.astype(np.float32)
    if lb is not None:
        lo32 = lb.astype(np.float32); hi32 = ub.astype(np.float32)
        lo32 = np.where(lo32.astype(np.float64) < lb, np.nextafter(lo32, np.float32(np.inf)), lo32)
        hi32 = np.where(hi32.astype(np.float64) > ub, np.nextafter(hi32, np.float32(-np.inf)), hi32)
        # one more float32 step inward where the box is wide enough: covers the gap between the
        # float64 parse and the exact decimal literal (far below one float32 ulp)
        wide = hi32 > lo32
        lo32 = np.where(wide, np.nextafter(lo32, np.float32(np.inf)), lo32)
        hi32 = np.where(wide, np.nextafter(hi32, np.float32(-np.inf)), hi32)
        x32 = np.minimum(np.maximum(x32, lo32), hi32)
    in_ok, out_ok = check(mp, sp, x32, "S2")
    rec = {"family": fam, "iid": iid, "baseline": r["baseline"], "S2_valid_interior": bool(in_ok and out_ok),
           "in_ok": bool(in_ok), "out_ok": bool(out_ok), "max_shift": float(np.abs(x32.astype(np.float64) - x64).max())}
    out.write(json.dumps(rec) + "\n"); out.flush(); print(rec, flush=True)
