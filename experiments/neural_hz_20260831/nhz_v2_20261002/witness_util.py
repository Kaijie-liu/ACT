"""Witness construction on the float32 input grid (fix found by audit N022).

`to_box_f32(x, lb, ub)` returns a float32 vector whose every entry lies strictly
inside [lb, ub] when compared in float64 (one float32 ulp inside when the
rounded value would touch or cross a bound), or None if some coordinate's box
contains no such float32 value.  Exact decimal comparison is left to the
independent audit (audit_n022_witness_replay.py).
"""
import numpy as np


def to_box_f32(x, lb, ub):
    x32 = np.asarray(x, dtype=np.float64).astype(np.float32)
    lb = np.asarray(lb, dtype=np.float64); ub = np.asarray(ub, dtype=np.float64)
    point = lb == ub
    for _ in range(4):
        lo = (x32.astype(np.float64) <= lb) & ~point
        hi = (x32.astype(np.float64) >= ub) & ~point
        if not (lo.any() or hi.any()):
            break
        x32 = np.where(lo, np.nextafter(x32, np.float32(np.inf)), x32)
        x32 = np.where(hi, np.nextafter(x32, np.float32(-np.inf)), x32)
    v = x32.astype(np.float64)
    ok = np.where(point, v == lb, (v > lb) & (v < ub))
    if not ok.all():
        return None
    return x32
