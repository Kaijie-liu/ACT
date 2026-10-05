# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent original-coordinate oracle, complete inverse and ordinary guards."""
from copy import deepcopy
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.test_c114_joint_sink_census_v1 import fixture,project
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import native_row,extend_actual
from experiments.neural_hz_20260831.c88_inline_tile_v1 import actual_rows
from experiments.neural_hz_20260831.c115_atomic_word_plan_v1 import _plan_row,reduce_rows,PreparedAtomic,emit_tile
from experiments.neural_hz_20260831.c116_row_composition_v1 import prove,reconstruct


def case(kind='none',offset=False):
    aux,out,base=deepcopy(fixture());aliases=[]
    if kind in ('negative','half'):
        scale=-1. if kind=='negative' else .5
        aux[1]['coefficients']=((11,1.),(0,-scale))
        aliases=[dict(old=11,representative=10,scale=scale)]
    elif kind=='late':
        aux[1]['coefficients']=((11,1.),(0,1.))
        aliases=[dict(old=10,representative=11,scale=-1.)]
    if offset:
        aux[-2]['coefficients']=tuple((c,v if c==18 else v/2) for c,v in aux[-2]['coefficients'])
    for row in [*aux,*out]:row['coefficients']=tuple(sorted(row['coefficients']))
    mapping={a['old']:(a['representative'],F(a['scale'])) for a in aliases}
    pool=WorkPool(4_000_000);rows=[]
    for row in [*aux,*out]:
        if row['slot'] in mapping:continue
        values={}
        for col,value in row['coefficients']:
            col,scale=mapping.get(col,(col,F(1)));values[col]=values.get(col,F(0))+F(value)*scale
        canonical=native_row(sorted(values.items()),0,slot=row['slot'])
        terms=[]
        for col,val in canonical['coefficients']:
            if col==row['slot']:continue
            value=F(val)/F(2)**canonical['gauge']
            terms.append((col,value.numerator,-(value.denominator.bit_length()-1)))
        pool.charge('test_complete_word_source',64+8*(len(terms)+1))
        rows.append(_plan_row(terms,row['slot'],0,pool=pool))
    state=reduce_rows(rows,len(aux)-len(aliases),{18,19},base,len(aliases),pool=pool)
    labels=[r['pivot'] for r in state['rows'][:state['new_factors']]]
    state['source_report']={}
    ready=PreparedAtomic(state,pool,base,dict(new_factors=state['new_factors']))
    report,packet=emit_tile(ready,base,pool=pool,enabled=True)
    new_aux,new_out=actual_rows(report,packet)
    if offset:
        aux[-2]['rhs']=.125
        for row in new_out:row['rhs']=float(F(1,16)*F(2)**row['gauge'])
    return aux,out,new_aux,new_out,base,labels,aliases


@pytest.mark.parametrize('kind,offset',[('none',False),('negative',False),('half',False),('late',False),('none',True)])
@pytest.mark.parametrize('phase',[-1,1])
def test_complete_composition_matches_independent_polynomials_and_binary_inverse(kind,offset,phase):
    args=case(kind,offset);before=deepcopy(args);pool=WorkPool(4_000_000)
    proof,evidence=prove(*args,pool=pool,enabled=True)
    aux,out,new_aux,new_out,base,labels,aliases=args
    expected=project(aux,out,base)
    assert expected==project(new_aux,new_out,base)
    point=[F((i%3)-1,4) for i in range(base)];point[0]=F(phase,2)
    for row,poly in zip(out,expected,strict=True):
        slot=row['slot'];point[slot]=-sum(v*point[c] if c>=0 else v for c,v in poly.items() if c!=slot)/poly[slot]
    original=extend_actual(aux,point);candidate=extend_actual(new_aux,point)
    assert reconstruct(candidate,evidence,pool=pool)==original
    assert point[0]-F(phase,2)==0 and point[1]+F(phase,4)<=F(3,2)
    assert proof['all_original_output_equations_equal']==len(out)
    assert proof['all_retained_defining_equations_equal']==len(new_aux)
    assert proof['all_removed_factors_reconstructible']==len(aux)-len(new_aux)
    assert proof['strict_total_nnz_decrease'] and args==before


@pytest.mark.parametrize('corruption',['coefficient','rhs','gauge','missing_output','duplicate_map','missing_alias','wrong_alias_scale','nonredundant_box'])
def test_complete_native_and_map_corruption_rejected(corruption):
    args=list(case('negative'));aux,out,new_aux,new_out,base,labels,aliases=args
    if corruption=='coefficient':
        row=new_out[0];row['coefficients']=tuple((c,v*1.125 if c!=row['slot'] else v) for c,v in row['coefficients'])
    elif corruption=='rhs':new_out[0]['rhs']+=.0625
    elif corruption=='gauge':new_out[0]['gauge']+=1
    elif corruption=='missing_output':new_out.pop()
    elif corruption=='duplicate_map':labels[0]=labels[1]
    elif corruption=='missing_alias':aliases.clear()
    elif corruption=='wrong_alias_scale':aliases[0]['scale']=.5
    elif corruption=='nonredundant_box':
        row=new_aux[0];row['coefficients']=tuple((c,v*2 if c!=row['slot'] else v) for c,v in row['coefficients'])
    with pytest.raises(ValueError):prove(*args,pool=WorkPool(4_000_000),enabled=True)


def test_hidden_auxiliary_consumer_of_deleted_sink_rejected():
    args=list(case());aux,out,new_aux,new_out,base,labels,aliases=args
    aux.append(dict(slot=20,rhs=0.,gauge=0,coefficients=((18,-.25),(20,1.))))
    slot=base+len(new_aux)
    new_aux.append(dict(slot=slot,rhs=0.,gauge=0,coefficients=((0,-.25),(slot,1.))))
    labels.append(20)
    with pytest.raises(ValueError,match='output-only'):prove(*args,pool=WorkPool(4_000_000),enabled=True)


def test_default_off_has_no_work_or_input_access():
    pool=WorkPool(0)
    assert prove(None,None,None,None,0,None,None,pool=pool) is None and pool.used==0


def test_zero_resource_budget_rejects_before_native_traversal():
    with pytest.raises(MemoryError):prove(*case(),pool=WorkPool(0),enabled=True)
