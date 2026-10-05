"""Exact scalar joint coefficients; no rounded intermediate multiplication.

The enclosing immutable transaction proves canonical source support and actual
unique incidence. This primitive is not a runtime HZ or lineage certificate.
"""
from fractions import Fraction as F
import math
import numpy as np
from experiments.neural_hz_20260831.c10_predicate_census_v1 import exact_products
from experiments.neural_hz_20260831.c13_separated_affine_census_v1 import coefficient_l1_numerator,exact_box

LOW,HIGH=2.**-20,2.**40
GUARDS=('pivot_ok','ratio_exact','rhs_exact','operand_window','joint_exact','result_window','redundant_box')
REASONS=('individual_exact','already_alias','already_unit_splice','output_live',
    'not_degree_two','missing_definition','not_direct_definition','binary_definition',
    'pivot_window','ratio_inexact','operand_window','rhs_inexact','joint_coefficient_inexact',
    'joint_coefficient_window','nonredundant_box')
CODE={v:i for i,v in enumerate(REASONS)}


def exact_float(value):
    """Return representability plus a candidate; no approximate value is admitted."""
    try:result=float(value)
    except OverflowError:return False,0.
    return math.isfinite(result) and F(result)==value,result


def prove(pc,pv,tc,tv,pivot_position,pivot,q,d,c,*,pool):
    """pc/pv are the actual complete definition prefix; tc includes pivot z.

Complete canonical order/normal source bytes are checked by the source binding.
Neither a fake prefix nor an unproved pivot_position is an admission API.
"""
    pool.charge('joint_pair_scalar_checks',160)
    out={g:-1 for g in GUARDS}
    out.update(reason=CODE['pivot_window'],ratio=0.,checked=0,failed_term=-1,
        overlaps=0,cancellations=0,overlap_present=-1,nnz_delta=0,old_products_exact=-1)
    if not math.isfinite(pivot) or not LOW<=pivot<=HIGH or math.frexp(pivot)[0]!=.5:
        out['pivot_ok']=0;return out
    out['pivot_ok']=1
    if not math.isfinite(q) or not LOW<=abs(q)<=HIGH:
        out.update(reason=CODE['operand_window'],operand_window=0);return out
    ratio=math.ldexp(-q,-(math.frexp(pivot)[1]-1))
    if not math.isfinite(ratio) or F(ratio)*F(pivot)!=-F(q):
        out.update(reason=CODE['ratio_inexact'],ratio_exact=0);return out
    out.update(ratio=ratio,ratio_exact=1)
    if not all(map(math.isfinite,(d,c))):raise ValueError('bound RHS is not finite')
    if d:
        pool.charge('joint_exact_RHS',128)
        ok,_=exact_float(F(c)+F(ratio)*F(d))
        if not ok:out.update(reason=CODE['rhs_inexact'],rhs_exact=0);return out
    out['rhs_exact']=1
    if pivot_position==0:out['overlap_present']=0
    prefix=tc[:pivot_position]
    for j in range(len(pc)):
        pool.charge('joint_parent_prefix_search',12*max(1,len(prefix).bit_length())+8 if len(prefix) else 8)
        pos=int(np.searchsorted(prefix,int(pc[j]))) if len(prefix) else 0
        matched=pos<len(prefix) and int(prefix[pos])==int(pc[j])
        t=float(tv[pos]) if matched else 0.
        pool.charge('joint_exact_coefficient',128)
        a=float(pv[j])
        if not (math.isfinite(a) and LOW<=abs(a)<=HIGH
                and (not matched or math.isfinite(t) and LOW<=abs(t)<=HIGH)):
            out.update(reason=CODE['operand_window'],operand_window=0,failed_term=j);return out
        if matched:out['overlaps']+=1;out['overlap_present']=1
        # This Fraction is the mathematical joint value; no float product is used.
        value=F(ratio)*F(a)+F(t)
        ok,combined=exact_float(value)
        out['checked']+=1
        if not ok:
            out.update(reason=CODE['joint_coefficient_inexact'],joint_exact=0,failed_term=j);return out
        if value and not LOW<=abs(combined)<=HIGH:
            out.update(reason=CODE['joint_coefficient_window'],result_window=0,failed_term=j);return out
        out['cancellations']+=int(value==0)
    out.update(operand_window=1,joint_exact=1,result_window=1)
    out['overlap_present']=int(out['overlaps']>0)
    pool.charge('joint_exact_box_norm',16*len(pv))
    if not exact_box(coefficient_l1_numerator(pv),d,pivot):
        out.update(reason=CODE['nonredundant_box'],redundant_box=0);return out
    out['redundant_box']=1
    # Separate diagnostic contrast, performed ONLY after all new guards pass.
    pool.charge('joint_admitted_old_product_contrast',64*len(pv))
    if len(pv):precise,_=exact_products(pv,ratio);out['old_products_exact']=int(precise.all())
    else:out['old_products_exact']=1
    out.update(reason=CODE['individual_exact'],nnz_delta=-2-out['overlaps']-out['cancellations'])
    return out
