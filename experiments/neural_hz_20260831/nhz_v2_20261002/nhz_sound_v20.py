"""Neural-HZ rigorous engine n020.1 = n019 + a time share for LP tightening (LOG N098, N099).

Once LP_TIGHTEN_SHARE of the row budget has passed (deadline set by the path):
  * pre-activation bounds of the remaining layers are computed without the GPU LP and without
    exact-LP polishing (shadow interval only), and
  * (.1) an exact-LP polishing loop already running stops at that time; units not yet polished
    keep the bound they already had (zero duals give the trivial box bound and the engine takes
    the minimum with the existing bound).
Looser bounds stay sound; the guard only keeps propagation from consuming the whole budget on
large models (relusplitter 110, 116, 141, 153, 165 in N039 v12; 116 spent 128 s in polishing).
Without a deadline the behaviour is identical to n019."""
import time

import numpy as np

import nhz_sound_v9 as _V9
from nhz_sound_mp3 import _highs_duals as _orig_highs_duals
from nhz_sound_v19 import SoundEngineV19

V20_VERSION = "n020.1"
LP_TIGHTEN_SHARE = 0.4
_POLISH_DEADLINE = [None]


def _timed_highs_duals(A64, b64, G, sign):
    dl = _POLISH_DEADLINE[0]
    if dl is None:
        return _orig_highs_duals(A64, b64, G, sign)
    import highspy
    import scipy.sparse as sp
    R, K = A64.shape
    out = np.zeros((G.shape[0], R))
    if time.time() > dl:
        return out
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    lp = highspy.HighsLp(); lp.num_col_ = K; lp.num_row_ = R
    lp.col_cost_ = np.zeros(K); lp.col_lower_ = -np.ones(K); lp.col_upper_ = np.ones(K)
    lp.row_lower_ = np.full(R, -highspy.kHighsInf); lp.row_upper_ = b64
    M = sp.csc_matrix(A64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data
    h.passModel(lp)
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    idx = np.arange(K, dtype=np.int32)
    for i in range(G.shape[0]):
        left = dl - time.time()
        if left <= 0:
            break
        h.setOptionValue("time_limit", float(max(left, 0.05)))
        h.changeColsCost(K, idx, sign * G[i])
        h.run()
        if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            y = np.asarray(h.getSolution().row_dual)
            out[i] = np.abs(np.minimum(y, 0.0)) if y.size else 0.0
            if not np.any(out[i]):
                out[i] = np.maximum(y, 0.0)
    return out


_V9._highs_duals = _timed_highs_duals          # v9's _bounds looks the name up in its module


class SoundEngineV20(SoundEngineV19):
    def _bounds(self, fc, fG32, fe, idx, rows, K, iters, lr, rad_all=None):
        dl = None
        if getattr(self, "deadline", None) is not None and getattr(self, "t_start", None) is not None:
            dl = self.t_start + LP_TIGHTEN_SHARE * (self.deadline - self.t_start)
            if iters > 0 and time.time() > dl:
                iters = 0
                self.lp_tighten_skipped = getattr(self, "lp_tighten_skipped", 0) + 1
        _POLISH_DEADLINE[0] = dl
        try:
            return super()._bounds(fc, fG32, fe, idx, rows, K, iters, lr, rad_all)
        finally:
            _POLISH_DEADLINE[0] = None
