"""Fresh actual box-input HZ with complete independent binary64 column proof."""
from fractions import Fraction as F
import numpy as np
from act.back_end.solver.solver_hz import sparse_hz_from_bounds
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def build_input(bounds,frame_id,*,pool,enabled=False):
    if not enabled:return None
    lo=bounds.lb.detach().cpu().double().numpy().reshape(-1)
    hi=bounds.ub.detach().cpu().double().numpy().reshape(-1)
    pool.charge('c100_complete_original_input_equations',128*len(lo)+1024)
    if lo.shape!=hi.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo>hi):
        raise ValueError('complete finite original box required')
    hz=sparse_hz_from_bounds(bounds,frame_id=frame_id)
    rows=[];radii=[];centers=[]
    for a,b in zip(lo,hi):
        c=float(F(float(F(float(a))+F(float(b))))*F(1,2))
        r=float(F(float(F(float(b))-F(float(a))))*F(1,2))
        if 0<abs(r)<=1e-12:raise ValueError('nonzero input radius would be dropped by original threshold')
        centers.append(c)
        if r:rows.append(len(centers)-1);radii.append(r)
    expected=np.asarray(centers,np.float64);radius=np.asarray(radii,np.float64)
    actual_rows=np.repeat(np.arange(hz.n_out),np.diff(hz.Gc.indptr))
    if (hz.c.tobytes()!=expected.tobytes() or hz.Gc.data.tobytes()!=radius.tobytes()
        or not np.array_equal(actual_rows,rows) or not np.array_equal(hz.Gc.indices,np.arange(len(rows)))
        or hz.n_cont!=len(rows) or hz.n_bin or hz.n_eq or hz.n_ineq or not hz.exact):
        raise ValueError('complete original input center/radius/active-column equations differ')
    return hz,dict(input_HZ_sha256=source_digest(hz),original_binary64_input_bits_proved=True,
        complete_input_coordinates=len(lo),active_input_columns=len(rows),
        actual_production_input_constructor_used=True,no_nonzero_input_radius_dropped=True,formal_gain=0)
