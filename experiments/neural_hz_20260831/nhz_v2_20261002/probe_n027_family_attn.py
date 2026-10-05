"""N027: float probe with engine n005.3 (attention + smooth) over a family's instances."""
import sys, os, csv, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_attn import AttnEngine, ATTN_VERSION
from nhz_engine import parse_vnnlib, terminal_query
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam, rows_arg, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
torch.cuda.set_per_process_memory_fraction(0.15)
import onnxruntime as ort
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
sel = range(len(inst)) if rows_arg == "all" else [int(x) for x in rows_arg.split(",")]
eng = {}
fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, "w") as fout:
    for ri in sel:
        o, s = inst[ri][:2]; mp = os.path.normpath(f"{ROOT}/{fam}/{o}"); sp = os.path.normpath(f"{ROOT}/{fam}/{s}")
        rec = {"engine": ATTN_VERSION, "row": ri, "onnx": o}
        try:
            if mp not in eng:
                e = AttnEngine(mp, "cuda", torch.float32)
                sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
                x = np.random.default_rng(0).uniform(0, 1, size=e.input_shape).astype(np.float32)
                ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
                t = torch.as_tensor(x, device="cuda"); o0, *_ = e.propagate(t, t, 0)
                rec["ort_center_err"] = float(np.abs(o0.c.reshape(-1).cpu().numpy() - ref).max()); eng[mp] = e
            e = eng[mp]
            z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
            spec = parse_vnnlib(sp, int(np.prod(e.input_shape)), int(o0.c.numel()))
            t0 = time.time(); worst = []
            for lb, ub in spec.boxes:
                out, rws, K, ph = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
                bnd, _ = terminal_query(out, rws, K, spec.disjuncts, 1000, 0.05)
                worst.append(max(bnd))
            rec.update({"K": K, "lp_worst": max(worst), "lp_cert_probe": max(worst) < 0, "wall_s": time.time() - t0,
                        "groups": [(g[0], g[2]) for g in e.factor_groups]})
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {ex}"[:200]
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(ri, rec.get("lp_worst"), rec.get("lp_cert_probe"), rec.get("ort_center_err", ""), round(rec.get("wall_s", 0), 1), rec.get("error", ""), flush=True)
