"""Engine n008.1: n007 (nhz_sound_mp2) + exact-LP polishing of intermediate bounds for small LPs.

Rule (structural, frozen): if the current LP has K * R <= POLISH_CELLS and the layer
has at most POLISH_MAX_NEURONS unstable neurons, every unstable neuron's two bounds are
re-solved exactly with HiGHS (one model per layer, objective swapped, warm start).  The
HiGHS row duals are used only as multipliers nu >= 0 of the rigorous float64
weak-duality evaluation (sound_eval), so the bound stays valid whatever HiGHS returns;
the tighter of the GPU and polished bounds is kept.  This reproduces the frozen
baseline's tight_bounds=True LP query on small networks.
"""
import numpy as np
import torch

from nhz_sound import SRows, gamma, TINY, optimise_nu, sound_eval
from nhz_sound_mp import radius64
from nhz_sound_mp2 import MP_VERSION as _BASE, SoundEngineMP2

MP3_VERSION = "n008.1"
POLISH_CELLS = 6_000_000
POLISH_MAX_NEURONS = 1500


def _highs_duals(A64: np.ndarray, b64: np.ndarray, G: np.ndarray, sign: float):
    """Row duals (>= 0) for max sign*g_i^T w s.t. A w <= b, |w| <= 1, for each row g_i of G."""
    import highspy
    R, K = A64.shape
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    lp = highspy.HighsLp(); lp.num_col_ = K; lp.num_row_ = R
    lp.col_cost_ = np.zeros(K); lp.col_lower_ = -np.ones(K); lp.col_upper_ = np.ones(K)
    lp.row_lower_ = np.full(R, -highspy.kHighsInf); lp.row_upper_ = b64
    import scipy.sparse as sp
    M = sp.csc_matrix(A64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data
    h.passModel(lp)
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    out = np.zeros((G.shape[0], R))
    idx = np.arange(K, dtype=np.int32)
    for i in range(G.shape[0]):
        h.changeColsCost(K, idx, sign * G[i])
        h.run()
        if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            y = np.asarray(h.getSolution().row_dual)
            out[i] = np.abs(np.minimum(y, 0.0)) if y.size else 0.0   # maximisation: duals of <= rows are <= 0 in HiGHS
            if not np.any(out[i]):
                out[i] = np.maximum(y, 0.0)
    return out


class SoundEngineMP3(SoundEngineMP2):
    def _bounds(self, fc, fG32, fe, idx, rows: SRows, K: int, iters: int, lr: float, rad_all=None):
        l, u = super()._bounds(fc, fG32, fe, idx, rows, K, iters, lr, rad_all)
        R = rows.n_rows
        if iters <= 0 or R == 0 or idx.numel() == 0 or idx.numel() > POLISH_MAX_NEURONS or K * R > POLISH_CELLS:
            return l, u
        A, b = rows.dense(K); A = A.to(self.device); b = b.to(self.device)
        A64 = A.cpu().numpy(); b64 = b.cpu().numpy()
        g = fG32[:, idx].t().contiguous().double(); c = fc[idx]
        G = g.cpu().numpy()
        for sign in (1.0, -1.0):
            nu = torch.as_tensor(_highs_duals(A64, b64, G, sign), device=self.device, dtype=torch.float64)
            bd = sound_eval(sign * g, sign * c, A, b, nu) + fe[idx]
            if sign > 0:
                u = torch.minimum(u, bd)
            else:
                l = torch.maximum(l, -bd)
        return l, u
