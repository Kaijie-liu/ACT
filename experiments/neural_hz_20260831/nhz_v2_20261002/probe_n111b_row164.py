"""N111b: reproduce the stage-E infeasibility on row 164 and bisect: LP relaxation (no integrality),
with/without the objective cutoff, with/without _fold_tiny, and with the witness point fixed."""
import sys, os, csv, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v20 import SoundEngineV20
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v8 import build_plan_v8
from nhz_terminal_v7 import _fold_tiny, run_portfolio_v7
import highspy
torch.cuda.set_per_process_memory_fraction(0.3)
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"; fam = "cifar100_2024"; ri = 164
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv"))); o, s, _ = inst[ri]
e = SoundEngineV20(f"{ROOT}/{fam}/{o}", "cuda"); n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
spec = parse_vnnlib(f"{ROOT}/{fam}/{s}", n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
e.deadline = time.time() + 100; e.t_start = time.time()
out, rows, K, ph = e.propagate(lb, ub, 300, 0.05); nz = [p for p in ph if int(p["idx"].numel())]
A, b = rows.dense(K)
g, cc, pad, extra, T = epigraph_objective(out, K, spec.disjuncts[3]); g_full = np.zeros(K + 1); g_full[:len(g)] = g
M, lo, hi, clo, chi, cost, integ, nb, m_tot, _ = build_plan_v8(A, b, ph, [], K, len(nz), 99, g_full, extra, None, 2, sign_aware=True)
print("plan", M.shape, "nnz", M.nnz, "binaries", nb, "tiny coefs", int((np.abs(M.data) <= 1e-9).sum()), "cutoff", cc + pad + 1e-4)


def solve(M, lo, hi, clo, chi, cost, integ, cutoff, use_int, tl=60):
    lp = highspy.HighsLp(); lp.num_col_ = M.shape[1]; lp.num_row_ = M.shape[0]; Mc = M.tocsc()
    lp.col_cost_ = cost; lp.col_lower_ = clo; lp.col_upper_ = chi
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -highspy.kHighsInf); lp.row_upper_ = np.where(np.isfinite(hi), hi, highspy.kHighsInf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Mc.indptr.astype(np.int32); lp.a_matrix_.index_ = Mc.indices.astype(np.int32); lp.a_matrix_.value_ = Mc.data.astype(np.float64)
    if use_int:
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.setOptionValue("time_limit", tl)
    h.setOptionValue("presolve", "off"); h.passModel(lp)
    if cutoff is not None:
        h.setOptionValue("objective_bound", float(cutoff))
    t0 = time.time(); h.run(); st = h.modelStatusToString(h.getModelStatus()); info = h.getInfo()
    return st, round(time.time() - t0, 1), info.objective_function_value


for name, (Mx, lox, hix) in (("raw", (M, lo, hi)), ("folded", _fold_tiny(M, lo, hi, clo, chi))):
    print(name, "LP no cutoff      :", solve(Mx, lox, hix, clo, chi, cost, integ, None, False))
    print(name, "LP with cutoff    :", solve(Mx, lox, hix, clo, chi, cost, integ, cc + pad + 1e-4, False))
    print(name, "MILP with cutoff  :", solve(Mx, lox, hix, clo, chi, cost, integ, cc + pad + 1e-4, True, 30))
# portfolio as used by the path
r = run_portfolio_v7(M, lo, hi, clo, chi, cost, integ, K, cc, pad, 30.0, target=1e-6)
print("portfolio:", r["status"], "excluded", r["excluded"], "upper", r["upper"])
# zero-objective feasibility LP, presolve on vs off
z = np.zeros_like(cost)
print("feasibility LP presolve off:", solve(M, lo, hi, clo, chi, z, integ, None, False))
