"""Exact sparse half-alias row patch; no solver, source mutation or fallback.

The patch is mathematical evidence, not a physically admitted HZ. A complete
consumer is multiplied by TWO ONCE, including binary terms and RHS. Every
selected z=s*u/2 is replaced simultaneously. All untouched rows stay literal.
"""
from fractions import Fraction as F
import math
import numpy as np

LOW,HIGH=2.**-20,2.**40


class RowRejected(ValueError):
    def __init__(self,reason,*,column=-1,value=None):
        super().__init__(reason)
        self.record=dict(reason=reason,column=int(column))
        if value is not None and math.isfinite(value):self.record['diagnostic_value']=float(value)


def _canonical(cols,values):
    return (isinstance(cols,np.ndarray) and isinstance(values,np.ndarray)
        and cols.ndim==values.ndim==1 and len(cols)==len(values)
        and cols.dtype.kind in 'iu' and values.dtype==np.float64
        and (not len(cols) or (cols[0]>=0 and np.all(cols[1:]>cols[:-1])))
        and np.isfinite(values).all() and not np.any(values==0.))


def row_patch(cc,cv,bc,bv,rhs,aliases,*,pool,enabled=False):
    """aliases contains EVERY selected (z,parent,sign) occurring in this row.

Complete cohort/row coverage must be established by the enclosing checker.
This function proves a row only, not the completeness of a supplied cohort.
Returned parent values REPLACE scaled original parent values; they are not
extra summands. The retained, non-parent/non-z body and all binaries scale2.
    """
    if not enabled:return None
    n,k=len(cc)+len(bc),len(aliases)
    if n>64_000_000 or k>64_000_000:raise MemoryError('unchanged row entry cap')
    pool.charge('gauge_row_validation_and_window',64+16*n+32*k)
    if not _canonical(cc,cv) or not _canonical(bc,bv) or not math.isfinite(rhs):
        raise RowRejected('noncanonical_or_nonfinite_source_row')
    if not k:raise RowRejected('untouched_row_must_remain_literal')
    removed=set();parents=set();normalized=[]
    for z,parent,sign in aliases:
        if (not isinstance(z,(int,np.integer)) or not isinstance(parent,(int,np.integer))
                or isinstance(sign,(bool,np.bool_)) or sign not in (-1,1)
                or not 0<=parent<z or z in removed or parent in parents):
            raise RowRejected('nonindependent_half_alias_row')
        removed.add(int(z));parents.add(int(parent));normalized.append((int(z),int(parent),int(sign)))
    if removed & parents:raise RowRejected('internal_half_alias_dependency')
    skip=np.zeros(len(cc),bool);updates=[];overlaps=0;cancelled=0
    for z,parent,sign in normalized:
        pool.charge('gauge_alias_source_and_parent_search',24*max(1,len(cc).bit_length())+16)
        zi=int(np.searchsorted(cc,z));pi=int(np.searchsorted(cc,parent))
        if zi>=len(cc) or int(cc[zi])!=z:raise RowRejected('missing_actual_alias_occurrence',column=z)
        overlap=pi<len(cc) and int(cc[pi])==parent
        q=float(cv[zi]);w=float(cv[pi]) if overlap else 0.
        skip[zi]=True
        if overlap:skip[pi]=True
        pool.charge('gauge_exact_joint_parent',128)
        if not LOW<=abs(q)<=HIGH or (w!=0. and not LOW<=abs(w)<=HIGH):
            raise RowRejected('unchanged_projection_operand_window',column=z,value=q if not LOW<=abs(q)<=HIGH else w)
        exact=2*F(w)+sign*F(q);value=float(exact)
        if not math.isfinite(value) or F(value)!=exact:
            raise RowRejected('inexact_joint_parent',column=parent,value=value)
        if value and not LOW<=abs(value)<=HIGH:
            raise RowRejected('joint_parent_window',column=parent,value=value)
        updates.append((parent,value));overlaps+=int(overlap);cancelled+=int(value==0.)
    # For a finite binary64 scalar, multiplication by2 is exact unless it
    # overflows. Window checks below imply finite normal scaled coefficients;
    # do not create or round a tiny half-product as an intermediate.
    body_values=cv[~skip];body=np.abs(body_values);binary=np.abs(bv)
    bad=(body<LOW/2)|(body>HIGH/2)
    if np.any(bad):
        at=int(np.flatnonzero(bad)[0])
        raise RowRejected('scaled_other_continuous_window',column=int(cc[~skip][at]),value=float(body_values[at]))
    bad=(binary<LOW/2)|(binary>HIGH/2)
    if np.any(bad):
        at=int(np.flatnonzero(bad)[0])
        raise RowRejected('scaled_binary_window',column=int(bc[at]),value=float(bv[at]))
    pool.charge('gauge_exact_scaled_RHS',64)
    try:new_rhs=math.ldexp(float(rhs),1)
    except OverflowError:raise RowRejected('scaled_RHS_overflow') from None
    if not math.isfinite(new_rhs) or F(new_rhs)!=2*F(float(rhs)):
        raise RowRejected('inexact_scaled_RHS')
    pool.charge('gauge_sparse_patch_order_and_metadata',12*k*max(1,k.bit_length())+16*k)
    updates.sort()
    return dict(row_gauge_exponent=1,removed_columns=np.array(sorted(removed),np.int64),
        parent_columns=np.array([p for p,v in updates],np.int64),
        parent_values=np.array([v for p,v in updates],np.float64),rhs=new_rhs,
        selected_occurrences=k,parent_overlaps=overlaps,exact_cancellations=cancelled,
        source_continuous_nnz=len(cc),new_continuous_nnz=len(cc)-overlaps-cancelled,
        binary_nnz=len(bc),consumer_nnz_delta=-overlaps-cancelled,
        positive_EQ_INEQ_scale_proved=True,all_binary_terms_preserved=True,
        complete_HZ_or_lineage_proved=False,formal_gain=0)


def reconstruct_half(values,aliases,*,pool,enabled=False):
    """Fraction-only inverse for this independent cohort; NOT old aliases.

values is a global-slot dict of exact retained latent assignments in [-1,1].
Parent identities are preserved. Old witness lineage remains a separate duty.
    """
    if not enabled:return None
    pool.charge('gauge_exact_half_inverse',64*(len(values)+len(aliases)))
    if (any(not isinstance(k,(int,np.integer)) or isinstance(k,(bool,np.bool_)) or k<0 for k in values)
            or any(type(v) is not F or abs(v)>1 for v in values.values())):
        raise ValueError('exact boxed retained latent values required')
    result=dict(values);columns=set();parents=set()
    for z,parent,sign in aliases:
        if (not isinstance(z,(int,np.integer)) or not isinstance(parent,(int,np.integer))
                or not 0<=parent<z or sign not in (-1,1) or isinstance(sign,(bool,np.bool_))
                or z in values or z in columns or parent in parents):
            raise ValueError('invalid independent half-alias witness map')
        columns.add(int(z));parents.add(int(parent))
    if columns & parents or not parents<=values.keys():raise ValueError('missing retained half-alias parent')
    for z,parent,sign in aliases:
        value=sign*values[parent]/2
        if abs(value)>1:raise ValueError('reconstructed half alias exceeds original box')
        result[int(z)]=value
    return result
