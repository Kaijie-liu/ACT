"""Ordinary source-chain and sparse-consumer mathematical checks."""
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c53_logical_singleton_census_v2 import assess as old_assess
from experiments.neural_hz_20260831.test_c53_logical_singleton_census_v2 import complete
from experiments.neural_hz_20260831.c57_real_scalar_census_v1 import assess
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v1 import Probe,word,native_word,fraction,in_window,product
from experiments.neural_hz_20260831.c56_gauged_carrier_v1 import digit_rows


@pytest.mark.parametrize('values',[(.9,.7),(-.9,.7),(.9,-.7),(.9,.7,.3,.6)])
def test_ordinary_general_chain_exact_inverse(values):
    p=Probe(len(values)+1,len(values),WorkPool(256_000_000));expected=F(1)
    for slot,v in enumerate(values,1):
        p.append(slot,slot-1,F(v),slot-1);expected*=F(v)
        assert fraction(p.weights[slot])==expected and int(p.roots[slot])==0
        assert int(p.depth[slot])==slot and abs(expected)<=1
    assert p.counts['singletons']==len(values)
    assert p.counts['composed_longer_than_binary64']>0


def test_shared_root_exact_consumer_cancellation():
    p=Probe(4,2,WorkPool(256_000_000))
    p.append(1,0,F(.9),0);p.append(2,0,F(.9),1)
    updates,delta=p.translated(np.array([1,2,3]),np.array([.3,-.3,.75]))
    assert updates=={} and delta==-2
    assert p.consumer['changed_terms']==2 and p.consumer['cancelled_roots']==1


def test_changed_parent_collides_with_unchanged_root_exactly():
    p=Probe(3,1,WorkPool(256_000_000));p.append(1,0,F(.9),0)
    updates,delta=p.translated(np.array([0,1,2]),np.array([.3,.7,1.]))
    assert fraction(updates[0])==F(.3)+F(.7)*F(.9)
    assert delta==-1 and p.consumer['coalesced_terms']==1


def test_real_consumer_carrier_reuse_and_same_C56_digit_cost():
    a=sp.csr_matrix([[-.9,1,0,0],[0,-.7,1,0],[0,0,.5,1.]])
    u=sp.csr_matrix([[0,0,.5,1.]])
    g=sp.csr_matrix([[0,0,0,1.]])
    hz=SimpleNamespace(n_cont=4,n_eq=3,n_ineq=1,n_bin=1,Ac=a,Ab=sp.csr_matrix([[0.],[0.],[1.]]),
        Auc=u,Aub=sp.csr_matrix([[1.]]),Gc=g,Gb=sp.csr_matrix((1,1)))
    p=Probe(4,3,WorkPool(256_000_000));p.append(1,0,F(.9),0);p.append(2,1,F(.7),1)
    r=p.finish(hz)
    exact=F(.5)*F(.7)*F(.9);w=word(exact.numerator,-(exact.denominator.bit_length()-1))
    digits=digit_rows(w)
    aux_nnz=sum(1+(i>0)+bool(d) for i,(d,width) in enumerate(digits))
    assert r['carriers']==1 and r['consumer_counts']['long_group_occurrences']==2
    assert r['auxiliary_continuous']==len(digits)
    assert r['added_radix_numeric_entries']==2*aux_nnz+3*len(digits)
    assert r['unchanged_n_binary']==1 and r['formal_gain']==0
    assert not r['complete_physical_ledger_or_new_HZ_proved']


@pytest.mark.parametrize('kind',['chain','shared','conv'])
def test_complete_original_source_guards_and_all_singleton_population(kind):
    _,saved=complete(kind)
    before=old_assess(saved,pool=WorkPool(256_000_000),enabled=True)
    after=assess(saved,pool=WorkPool(256_000_000),enabled=True)
    assert before['totals']==after['totals']
    general=after['general_scalar_diagnostic']
    assert general['counts'].get('singletons',0)==before['totals']['all_output_dead_singletons']
    assert after['complete_original_source_paths_frame_and_output_maps_checked']
    assert not general['actual_source_writer_native_witness_or_solver_executed']


def test_window_failure_is_counted_without_chain_restart_or_admission():
    p=Probe(4,3,WorkPool(256_000_000))
    for i in range(1,4):p.append(i,i-1,F(1,256),i-1)
    assert fraction(p.weights[3])==F(1,2**24) and int(p.roots[3])==0
    assert p.counts['composed_window_violations']==1
    assert not in_window(p.weights[3])


def test_unchanged_precision_cap_fails_closed():
    with pytest.raises(MemoryError):word((1<<512)+1,-513)


def test_default_off_and_budget_fail_closed():
    assert assess(object(),pool=object()) is None
    with pytest.raises(MemoryError):Probe(4,3,WorkPool(0))


def test_lost_topological_definition_fails_closed():
    p=Probe(3,1,WorkPool(256_000_000))
    with pytest.raises(ValueError):p.append(1,2,F(.5),0)


def test_original_frame_mismatch_fails_closed():
    _,saved=complete();saved['eq_scales'][-1]+=1
    with pytest.raises(ValueError):assess(saved,pool=WorkPool(256_000_000),enabled=True)
