"""Independent finite-forest optimality and exact eliminated-row identities."""
from fractions import Fraction as F
from itertools import product as masks
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c61_retained_boundary_v1 import optimize,assess,precise,complete_interval
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import word,fraction
from experiments.neural_hz_20260831.c59_full_consumer_census_v2 import assess as complete_profile,gauge
from experiments.neural_hz_20260831.c58_source_inverse_census_v2 import assess as inverse
from experiments.neural_hz_20260831.test_c53_logical_singleton_census_v2 import complete


def binary64_mantissa_fits(value):
    n=abs(value.numerator)
    while n and not n%2:n//=2
    return n.bit_length()<=53


def data(kind):
    r=F(2**23+3,2**24)
    parents={1:0,2:1,3:2} if kind in ('chain','mixed') else {1:0,2:1,3:1,4:2,5:3}
    ratios={v:(-r if v%2 else r) for v in parents}
    if kind=='mixed':ratios[2]=F(1,2)
    if kind=='halves':ratios={v:F(-1,2) for v in parents}
    leaves=set(parents)-set(parents.values());external={v:[r,-F(3,4)] if v in leaves else [] for v in parents}
    states={};keep={};remove={}
    for v,p in parents.items():
        weights={p:ratios[v]}
        if p in parents:weights.update({a:ratios[v]*fraction(w) for a,w in states[p].items()})
        states[v]={a:word(x.numerator,-(x.denominator.bit_length()-1)) for a,x in weights.items()}
        keep[v]={a:precise(w) for a,w in states[v].items()}
        remove[v]={a:all(binary64_mantissa_fits(c*fraction(w)) for c in external[v]) for a,w in states[v].items()}
    return parents,ratios,external,states,keep,remove


@pytest.mark.parametrize('kind',['chain','branch','mixed','halves'])
def test_optimum_against_all_representation_masks_and_fraction_inverse(kind):
    parents,ratios,external,states,keep,remove=data(kind);columns=list(parents)
    optimum,selected,anchors,_=optimize(parents,states,keep,remove,WorkPool(256_000_000))
    feasible=[]
    for mask in masks((False,True),repeat=len(columns)):
        removed=dict(zip(columns,mask));valid=True
        for v,p in parents.items():
            multiplier=ratios[v];a=p
            while a in removed and removed[a]:multiplier*=ratios[a];a=parents[a]
            valid&=(all(binary64_mantissa_fits(c*multiplier) for c in external[v]) if removed[v]
                    else binary64_mantissa_fits(multiplier))
        if valid:feasible.append(sum(mask))
    assert optimum==max(feasible)==sum(selected.values())
    # Arbitrary retained values need not satisfy the predicates: residual identity
    # proves both directions, rather than checking only a single feasible point.
    values={0:F(-2,7)}
    for v in parents:values[v]=ratios[v]*values[parents[v]] if selected[v] else F((v%5)-2,5)
    for v,p in parents.items():
        original=values[v]-ratios[v]*values[p]
        if selected[v]:
            assert original==0 and abs(values[v])<=1
            a=anchors[v];assert values[v]==fraction(states[v][a])*values[a]
            for c in external[v]:assert c*values[v]==c*fraction(states[v][a])*values[a]
        else:
            a=anchors[p] if p in parents else p
            assert original==values[v]-fraction(states[v][a])*values[a]


@pytest.mark.parametrize('kind',['chain','shared','conv'])
def test_complete_original_source_cohort_and_legacy_comparator(kind):
    _,saved=complete(kind);pool=WorkPool(256_000_000)
    packet=inverse(saved,pool=pool,enabled=True)['general_scalar_diagnostic']['packet']
    old=complete_profile(saved,packet,pool=pool,enabled=True)
    if old['counts'].get('coalesced_terms',0):
        with pytest.raises(ValueError,match='full-root support coalescence'):assess(saved,packet,pool=pool,enabled=True)
    else:
        result=assess(saved,packet,pool=pool,enabled=True)
        assert result['complete'] and result['native_symbolic_optimality_proved']
        assert result['raw_nodes']==old['symbolic_removed_continuous']
        assert result['precision_optimal_removed']>=result['legacy_selected']
        assert result['symbolic_predicate_nnz']==result['original_predicate_nnz']-2*result['precision_optimal_removed']
        assert not result['physical_HZ_native_or_source_first_proved']


@pytest.mark.parametrize('which',['binary','rhs'])
def test_universal_row_envelope_includes_unchanged_binary_and_rhs(which):
    lo,hi,q=gauge([(3,-39)]);assert q==18
    binary=[2.**23] if which=='binary' else []
    rhs=2.**23 if which=='rhs' else 0.
    assert complete_interval(lo,hi,binary,rhs)[2] is None


def test_deterministic_tie_retains_current_node():
    parents={1:0,2:1};states={1:{0:(1,-1)},2:{1:(1,-1),0:(1,-2)}}
    keep={1:{0:True},2:{1:True,0:True}};remove={1:{0:True},2:{1:True,0:False}}
    total,selected,_,_=optimize(parents,states,keep,remove,WorkPool(256_000_000))
    assert total==1 and selected=={1:False,2:True}


def test_default_off_and_prepaid_resource_guard():
    assert assess(None,None,pool=None) is None
    with pytest.raises(MemoryError):optimize({1:0},{1:{0:(1,-1)}},{1:{0:True}},{1:{0:True}},WorkPool(0))
