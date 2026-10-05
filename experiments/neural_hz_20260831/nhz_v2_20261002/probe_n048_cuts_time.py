"""N048: does the ideal-cut root loop (N045) reduce time-to-exclusion when the solver is
given 90 s?  Same rows/plan as N046 (sign-aware, single thread seed 0), with cut rounds
R in {0, 3, 8}; separation keeps only cuts violated by more than 1e-3."""
import sys, os, csv, json, time, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v5 import build_plan_v5, needs_binary
from nhz_ideal_cuts import separate
import highspy, threading
torch.cuda.set_per_process_memory_fraction(0.05)
csv.field_size_limit(10**9)
ov = {(r['benchmark'], r['iid']): r for r in csv.DictReader(open('/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv'))}
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam = sys.argv[1]; iids = sys.argv[2].split(","); outp = sys.argv[3]; tl = float(sys.argv[4])
out = open(outp, "a"); eng = {}


def mk(M, rhs, n, K, nb, g_full, integer):
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = M.shape[0]; M = M.tocsc()
    lp.col_cost_ = np.concatenate([-g_full, np.zeros(nb)]); lp.col_lower_ = np.concatenate([-np.ones(K + 1), np.zeros(nb)]); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data
    if integer:
        lp.integrality_ = [highspy.HighsVarType.kContinuous] * (K + 1) + [highspy.HighsVarType.kInteger] * nb
    return lp


for iid in iids:
    r = ov[(fam, iid)]
    mp = os.path.normpath(os.path.join(ROOT, fam, r['onnx'])); spp = os.path.normpath(os.path.join(ROOT, fam, r['vnnlib']))
    if mp not in eng:
        eng.clear(); eng[mp] = SoundEngineV9(mp, 'cuda')
    e = eng[mp]
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(spp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M0, rhs0, nb, _ = build_plan_v5(A, b, ph, K, 1, 99, g_full, extra, True)
    last = [p for p in ph if int(p["idx"].numel())][-1]; keep = np.flatnonzero(needs_binary(last, g_full, extra))
    units = []
    for j, s in enumerate(keep):
        a = np.zeros(K + 1); gx = last["gx"][s].double().cpu().numpy(); a[:gx.size] = gx
        gy = np.zeros(K + 1); g1 = last["gy"][s].double().cpu().numpy(); gy[:g1.size] = g1
        units.append(dict(a=a, cx=float(last["cx"][s]), gy=gy, cy=float(last["cy"][s]), ex=float(last["ex"][s]), ey=float(last["ey"][s]), col=K + 1 + j))
    n = K + 1 + nb; variants = {}
    for R in (0, 3, 8):
        M, rhs = M0, rhs0; ncut = 0
        for rd in range(R):
            h = highspy.Highs(); h.setOptionValue("output_flag", False); h.passModel(mk(M, rhs, n, K, nb, g_full, False)); h.run()
            x = np.array(h.getSolution().col_value)
            cuts = [c for c in separate(units, x[:K + 1], x[[u["col"] for u in units]], np.array([u["gy"] @ x[:K + 1] + u["cy"] for u in units])) if c[4] > 1e-3]
            if not cuts:
                break
            Rm = np.zeros((len(cuts), n)); rr = []
            for i, (cw, col, cd, rh, viol) in enumerate(cuts):
                Rm[i, :K + 1] = cw; Rm[i, col] = cd; rr.append(rh)
            M = sp.vstack([M, sp.csr_matrix(Rm)]).tocsc(); rhs = np.concatenate([rhs, rr]); ncut += len(cuts)
        variants[R] = (M, rhs, ncut)
    res = {}
    def run(R):
        M, rhs, ncut = variants[R]
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.setOptionValue("time_limit", tl)
        h.passModel(mk(M, rhs, n, K, nb, g_full, True)); h.setOptionValue("objective_bound", float(cc + pad + 1e-4))
        t0 = time.time(); h.run(); info = h.getInfo()
        res[R] = {"cuts": ncut, "status": h.modelStatusToString(h.getModelStatus()), "wall_s": time.time() - t0, "nodes": int(info.mip_node_count),
                  "upper": (-info.mip_dual_bound + cc + pad) if np.isfinite(info.mip_dual_bound) else None}
    th = [threading.Thread(target=run, args=(R,)) for R in variants]; [x.start() for x in th]; [x.join() for x in th]
    rec = {"family": fam, "iid": int(iid), "binaries": nb, "time_limit": tl, "variants": res}
    out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
