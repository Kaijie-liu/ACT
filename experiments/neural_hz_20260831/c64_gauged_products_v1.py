"""Exact native products followed by ONE proved whole-row positive gauge."""
from fractions import Fraction
import numpy as np


def products(values,ratios,*,pool):
    if (type(values) is not np.ndarray or type(ratios) is not np.ndarray
            or values.dtype!=np.dtype(np.float64) or ratios.dtype!=values.dtype
            or values.ndim!=1 or ratios.shape!=values.shape):raise ValueError('native matching vectors required')
    n=len(values)
    if not n:return np.empty(0),dict(hits=0,general=0)
    # Unchanged C21 common/left/general prices. Unlike C21, a too-small exact
    # result is returned only to the mandatory whole-row gauge, never admitted.
    pool.charge('product_common',12*n+4)
    av,ar=np.abs(values),np.abs(ratios)
    if not (2.**-20<=av.min() and av.max()<=2.**40 and 2.**-60<=ar.min() and ar.max()<=1.):
        raise ValueError('unchanged normal operand domain')
    right=np.frexp(ar)[0]==.5;result=values*ratios;general=0
    if not right.all():
        pool.charge('product_left_classification',4*n)
        dyadic=right|(np.frexp(av)[0]==.5)
        for i in np.flatnonzero(~dyadic):
            pool.charge('product_general_exact',64);general+=1
            if Fraction(float(result[i]))!=Fraction(float(values[i]))*Fraction(float(ratios[i])):
                raise ValueError('inexact source product')
    # The old output abs/lower-flag pass is replaced by abs/min, at the same
    # prepaid common price. This exact minimum is consumed by the row gauge.
    return result,dict(hits=n,general=general,minimum_absolute_product=float(np.abs(result).min()))


def gauge_row(cv,bv,rhs,*,pool):
    # All incoming operands are exact normal results. Range includes unchanged
    # continuous terms, every binary coefficient and RHS, not only changed terms.
    n=len(cv)+len(bv)+int(bool(rhs));pool.charge('c64_complete_written_row_gauge',8*n+32)
    operands=np.r_[cv,bv,[rhs] if rhs else []]
    if not n:return cv,bv,rhs,0
    absolute=np.abs(operands)
    if not np.isfinite(absolute).all() or np.any(absolute==0):raise ValueError('finite nonzero exact row required')
    mantissa,exponent=np.frexp(absolute)
    low=-19-int(exponent.min());largest=int(exponent.max())
    high=(41 if float(mantissa[exponent==largest].max())==.5 else 40)-largest
    if low>high:raise ValueError('complete row has no unchanged-window gauge')
    q=min(max(0,low),high)
    scaled=[np.ldexp(a,q) for a in (cv,bv)];value=float(np.ldexp(rhs,q))
    if any(not np.array_equal(np.ldexp(b,-q),a) for a,b in zip((cv,bv),scaled)) or np.ldexp(value,-q)!=rhs:
        raise ValueError('written row gauge not reversible')
    return *scaled,value,q


def gauge_definition(cv,bv,rhs,minimum_product,*,pool):
    """Owned source/radix definition whose unchanged positive pivot dominates.

    The fresh generator's strict L1 normalization proves dominance of ALL old
    continuous/binary/RHS operands. Contractive substitutions cannot increase
    their magnitudes. This is not a checker for arbitrary externally given rows.
    Only changed products can violate the lower window; q>=0 keeps every old
    unchanged operand above that window. The unchanged pivot proves the upper.
    """
    pool.charge('c64_owned_definition_gauge_decision',24)
    if not len(cv) or cv[-1]<=0 or not 0<minimum_product<=2.**40:raise ValueError('owned normalized definition required')
    pm,pe=np.frexp(cv[-1])
    if pm!=.5:raise ValueError('positive dyadic dominant pivot required')
    q=max(0,-19-int(np.frexp(minimum_product)[1]))
    if q>41-int(pe):raise ValueError('complete dominant-pivot row has no native gauge')
    if q==0:return cv,bv,rhs,0
    # No magnitude scan when q==0. A real nonzero shift pays all writes and
    # reverse-bit checks, preserving the exact original whole-row relation.
    pool.charge('c64_nonzero_definition_scale',4*(len(cv)+len(bv))+8)
    scaled=[np.ldexp(a,q) for a in (cv,bv)];value=float(np.ldexp(rhs,q))
    if any(not np.array_equal(np.ldexp(b,-q),a) for a,b in zip((cv,bv),scaled)) or np.ldexp(value,-q)!=rhs:
        raise ValueError('whole definition scaling is not reversible')
    return *scaled,value,q
