"""N045: root loop of ideal latent-box cuts (nhz_ideal_cuts.py) on top of the sign-aware
plan (terminal v5), then the same 4-seed portfolio.  SafeNLP CERT rows still open in
N044.  Diagnostic probe; reports root LP bound per round and the MILP outcome."""
import sys, os, csv, json, time, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v5 import build_plan_v5, needs_binary, run_portfolio
from nhz_ideal_cuts import separate
import highspy
torch.cuda.set_per_process_memory_fraction(0.05)
csv.field_size_limit(10**9)
ov = {(r['benchmark'], r['iid']): r for r in csv.DictReader(open('/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv'))}
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam = sys.argv[1]; iids = sys.argv[2].split(","); outp = sys.argv[3]; rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 20
budget = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
out = open(outp, "a"); eng = {}


def lp_solve(M, rhs, n, K, nb, g_full):
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = M.shape[0]
    lp.col_cost_ = np.concatenate([-g_full, np.zeros(nb)]); lp.col_lower_ = np.concatenate([-np.ones(K + 1), np.zeros(nb)])
    lp.col_upper_ = np.ones(n); lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; M = M.tocsc()
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.passModel(lp); h.run()
    return h.getInfo().objective_function_value, np.array(h.getSolution().col_value)


for iid in iids:
    r = ov[(fam, iid)]
    mp = os.path.normpath(os.path.join(ROOT, fam, r['onnx'])); spp = os.path.normpath(os.path.join(ROOT, fam, r['vnnlib']))
    if mp not in eng:
        eng.clear(); eng[mp] = SoundEngineV9(mp, 'cuda')
    e = eng[mp]
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(spp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    t0 = time.time()
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M, rhs, nb, _ = build_plan_v5(A, b, ph, K, 1, 99, g_full, extra, True)
    last = [p for p in ph if int(p["idx"].numel())][-1]
    keep = np.flatnonzero(needs_binary(last, g_full, extra))
    units = []
    for j, s in enumerate(keep):
        a = np.zeros(K + 1); gx = last["gx"][s].double().cpu().numpy(); a[:gx.size] = gx
        gy = np.zeros(K + 1); g1 = last["gy"][s].double().cpu().numpy(); gy[:g1.size] = g1
        units.append(dict(a=a, cx=float(last["cx"][s]), gy=gy, cy=float(last["cy"][s]), ex=float(last["ex"][s]),
                          ey=float(last["ey"][s]), col=K + 1 + j))
    n = K + 1 + nb; hist = []; ncuts = 0
    for rd in range(rounds + 1):
        f, x = lp_solve(M, rhs, n, K, nb, g_full)
        hist.append(round(cc - f + pad, 4))
        if rd == rounds:
            break
        cuts = separate(units, x[:K + 1], x[[u["col"] for u in units]], np.array([u["gy"] @ x[:K + 1] + u["cy"] for u in units]))
        if not cuts:
            break
        R = []; rr = []
        for cw, col, cd, rh, viol in cuts:
            row = np.zeros(n); row[:K + 1] = cw; row[col] = cd; R.append(row); rr.append(rh)
        M = sp.vstack([M, sp.csr_matrix(np.array(R))]).tocsc(); rhs = np.concatenate([rhs, rr]); ncuts += len(cuts)
    t_cut = time.time() - t0
    m = run_portfolio(M, rhs, nb, K, g_full, cc, pad, max(budget - t_cut, 0.5))
    rec = {"family": fam, "iid": int(iid), "baseline": r["raw_verdict"], "binaries": nb, "cuts": ncuts,
           "root_bounds": hist, "cut_s": t_cut, "milp_status": m["status"], "excluded": m["excluded"],
           "milp_upper": m["upper"], "total_s": time.time() - t0}
    out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
