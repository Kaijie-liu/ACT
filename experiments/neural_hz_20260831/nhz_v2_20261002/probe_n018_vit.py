"""N018 probe: attention engine n005 on vit_2023 rows (float, diagnostic)."""
import sys, os, csv, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_attn import AttnEngine, ATTN_VERSION
from nhz_engine import parse_vnnlib, terminal_query
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/vit_2023"
rows = [int(x) for x in sys.argv[1].split(",")]; iters = int(sys.argv[2]); out_path = sys.argv[3]
torch.cuda.set_per_process_memory_fraction(0.15)
import onnxruntime as ort
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
eng = {}
fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, "w") as fout:
    for ri in rows:
        o, s = inst[ri][:2]; mp = f"{ROOT}/{o}"; sp = f"{ROOT}/{s}"
        rec = {"engine": ATTN_VERSION, "row": ri, "onnx": o}
        if mp not in eng:
            e = AttnEngine(mp, "cuda", torch.float32)
            sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
            x = np.random.default_rng(0).uniform(0, 1, size=(1, 3, 32, 32)).astype(np.float32)
            ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
            t = torch.as_tensor(x, device="cuda")
            o0, *_ = e.propagate(t, t, 0)
            rec["ort_center_err"] = float(np.abs(o0.c.reshape(-1).cpu().numpy() - ref).max())
            eng[mp] = e
        e = eng[mp]
        spec = parse_vnnlib(sp, 3072, 10)
        lb, ub = spec.boxes[0]
        t0 = time.time()
        out, rws, K, ph = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), iters, 0.05)
        bnd, _ = terminal_query(out, rws, K, spec.disjuncts, 1000, 0.05)
        sh = []
        c = out.c.reshape(-1); G = out.G.reshape(K, c.numel())
        for atoms in spec.disjuncts:
            a0, b0 = atoms[0]; at = torch.as_tensor(a0, device="cuda", dtype=torch.float32)
            sh.append(float(b0 - c @ at + (G @ at).abs().sum()))
        rec.update({"K": K, "unstable": sum(int(p.idx.numel()) for p in ph), "shadow_worst": max(sh),
                    "lp_worst": max(bnd), "eps": float((ub - lb).max() / 2), "wall_s": time.time() - t0})
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(rec, flush=True)
