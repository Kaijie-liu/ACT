"""Exact inverse primitive checks; no synthetic cohort can promote an HZ."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import Probe, word, fraction
from experiments.neural_hz_20260831.c58_reconstruction_equations_v1 import build, audit, encode, decode, operands, seal
from experiments.neural_hz_20260831.c58_source_inverse_census_v2 import assess
from experiments.neural_hz_20260831.test_c53_logical_singleton_census_v2 import complete


def packet_for(values):
    pool=WorkPool(256_000_000)
    probe=Probe(len(values)+1,len(values),pool)
    for slot,v in enumerate(values,1):probe.append(slot,slot-1,F(v),slot-1)
    packet=build(probe.roots,probe.weights,n_bin=2,source_binding='test-frame',pool=pool,enabled=True)
    return probe,packet,pool


@pytest.mark.parametrize('values',[(.9,.7),(-.9,.7),(.9,-.7),(.3,.6,.7)])
def test_signed_general_composition_and_restore(values):
    p,packet,pool=packet_for(values)
    a=audit(packet,p.roots,p.weights,n_bin=2,source_binding='test-frame',pool=pool)
    b=audit(decode(encode(packet)),p.roots,p.weights,n_bin=2,source_binding='test-frame',pool=pool)
    assert a==b and a['coordinates_checked']==len(values)+1
    expected=F(1)
    for v in values:expected*=F(v)
    assert fraction(p.weights[-1])==expected


@pytest.mark.parametrize('floor',[-1,-20,-21,-38])
def test_observed_scale_range_uses_real_bounded_operands(floor):
    ratio=word(3,floor-1);n,d=operands(ratio)
    assert fraction(n)/fraction(d)==fraction(ratio)
    assert F(1,2**20)<=abs(fraction(n))<=2**40
    assert F(1,2**20)<=fraction(d)<=2**40
    assert d[1]==max(0,-20-floor)


@pytest.mark.parametrize('kind',['chain','shared','conv'])
def test_complete_source_cohort_and_unchanged_binaries(kind):
    _,saved=complete(kind)
    result=assess(saved,pool=WorkPool(256_000_000),enabled=True)
    inverse=result['general_scalar_diagnostic']
    assert inverse['proof']['coordinates_checked']==saved['hz'].n_cont
    assert inverse['proof']==inverse['restored_proof']
    assert inverse['packet']['n_bin']==saved['hz'].n_bin
    assert inverse['counts'].get('singletons',0)==result['totals']['all_output_dead_singletons']
    assert not inverse['complete_consumer_analysis']
    assert not inverse['complete_physical_HZ_reduction_proved']


def test_repeated_scale_retains_distinct_roots():
    p=Probe(4,2,WorkPool(256_000_000));p.append(2,0,F(.9),0);p.append(3,1,F(.9),1)
    packet=build(p.roots,p.weights,n_bin=1,source_binding='shared-frame',pool=p.pool,enabled=True)
    report=audit(packet,p.roots,p.weights,n_bin=1,source_binding='shared-frame',pool=p.pool)
    assert report['independent_equations_checked']==2
    assert int(packet['frame'][2]) & (2**32-1)==0
    assert int(packet['frame'][3]) & (2**32-1)==1


def test_wrong_equation_cannot_pass_original_coordinate_proof():
    p,packet,pool=packet_for([.9,.7])
    packet['frame'][-1]=packet['frame'][1]
    packet['seal']=seal(packet)
    with pytest.raises(ValueError):audit(packet,p.roots,p.weights,n_bin=2,source_binding='test-frame',pool=pool)


def test_default_off_and_work_cap():
    assert build(None,None,n_bin=0,source_binding='',pool=None) is None
    assert assess(None,pool=None) is None
    with pytest.raises(MemoryError):build(np.array([0]),[(1,0)],n_bin=1,source_binding='x',pool=WorkPool(0),enabled=True)


def test_unchanged_operand_window_rejects_unrepresentable_equation():
    with pytest.raises(ValueError):operands((1,-61))


def test_unchanged_precision_rejects_unrepresentable_mantissa():
    with pytest.raises(MemoryError):operands(((1<<512)+1,-513))
