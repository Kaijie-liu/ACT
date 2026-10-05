from dataclasses import replace
from fractions import Fraction as F
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono

from experiments.neural_hz_20260831 import c26_tagged_transplant_v1 as t
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c15_unit_row_splice_v1 import splice
from experiments.neural_hz_20260831.test_c15_unit_row_splice_v1 import fixture


def setup(*,mixed=False,subtract=False,redirect=False):
    # The unchanged C15 oracle includes fixed Python/certificate overhead;
    # its original 512-pair fixture passes that separate component gate.
    hz,kw,info=fixture(count=512,mixed=mixed,subtract=subtract)
    if redirect:
        assert not mixed
        kw.update(logical_n_cont=hz.n_cont,eq_roots=np.arange(hz.n_eq,dtype=np.int64),eq_scales=np.zeros(hz.n_eq,np.int64))
    eq=np.arange(hz.n_eq,dtype=np.int64)
    le=hz.n_eq+np.arange(hz.n_ineq,dtype=np.int64)
    plans=[t.Plan(col,d,c,kind,int(eq[d]),int((le if kind else eq)[c]),p,s,h,
        out if redirect else None,(out,)) for col,out,d,kind,c,p,s,h,beta in info]
    source=actual_words(hz,kw['old_n_cont'],kw['logical_n_cont'],eq,le)
    overlay=Overlay(source,np.zeros(0,np.uint64),hz.n_eq+hz.n_ineq)
    return hz,kw,info,eq,le,plans,overlay


def compiled(kw,plans,pool=None):
    return t.compile_lineage(kw['eq_roots'],kw['eq_scales'],plans,old_n_cont=kw['old_n_cont'],
        old_n_eq=kw['old_n_eq'],pool=pool or WorkPool(256_000_000),enabled=True)


def test_default_off_does_not_read_any_input():
    assert t.compile_lineage(object(),object(),object(),old_n_cont=object(),old_n_eq=object(),pool=object()) is None


@pytest.mark.parametrize('exponent',[-20,-1,0,1,40])
@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('inequality',[False,True])
def test_bit_exact_disjoint_tag_roundtrip(exponent,sign,inequality):
    code=t.encode_splice(91,142,inequality,2.**exponent,sign)
    assert t.decode(code)==('splice',91,142,inequality,2.**exponent,sign)
    assert 0<code<2**63 and t.decode(-19)==('alias',18)
    assert t.decode(t.REDIRECT|513)==('redirect',513)


@pytest.mark.parametrize('bad',[1<<60,t.SPLICE|(1<<60),t.REDIRECT|(1<<24),t.SPLICE|(61<<21),-t.UID_LIMIT-1])
def test_reserved_and_out_of_domain_tags_reject(bad):
    with pytest.raises(ValueError): t.decode(bad)


@pytest.mark.parametrize('mixed,subtract,redirect',[(False,False,False),(True,True,False),(False,True,True)])
def test_full_uid_transfer_owner_equality_and_exact_extension(mixed,subtract,redirect):
    hz,kw,info,eq,le,plans,overlay=setup(mixed=mixed,subtract=subtract,redirect=redirect)
    old_roots=kw['eq_roots'].copy(); old_scales=kw['eq_scales'].copy()
    d=compiled(kw,plans)
    oracle,metrics=splice(hz,enabled=True,**kw)
    keep=np.ones(hz.n_eq,bool); keep[[p.definition for p in plans]]=False
    neweq,newle=eq[keep].copy(),le.copy()
    pool=WorkPool(256_000_000)
    for p in plans:
        loc=p.consumer if p.inequality else d.eq_row(p.consumer,pool=pool)
        (newle if p.inequality else neweq)[loc]=p.producer_uid
        assert d.retired_to(p.consumer_uid,pool=pool)==p.producer_uid
        assert d.eq_row(p.definition,pool=pool) is None
    assert len(set(map(int,np.r_[neweq,newle])))==len(neweq)+len(newle)
    expected=actual_words(oracle.hz,kw['old_n_cont'],kw['logical_n_cont'],neweq,newle)
    assert list(d.iter_words(overlay,pool=pool))==list(expected)
    assert [d.owner_query(i,overlay,pool=pool) for i in range(len(expected))]==list(expected)
    for i in range(hz.n_eq):
        assert d.eq_row(i,pool=pool)==(int(np.count_nonzero(keep[:i])) if keep[i] else None)
    for first in (F(-1),F(0),F(1)):
        point=[F(0)]*hz.n_cont; point[0]=first; point[1]=F(1,3)
        assert d.reconstruct_fraction(oracle.hz,point,pool=pool)==oracle.reconstruct_fraction(point)
    assert np.array_equal(kw['eq_roots'],old_roots) and np.array_equal(kw['eq_scales'],old_scales)
    restored=pickle.loads(pickle.dumps(d,protocol=5)); restored.validate()
    assert restored.fingerprint()==d.fingerprint()


@pytest.mark.parametrize('bad',['order','definition','consumer','uid','pivot','offset','tail','consumer_main','copycap'])
def test_rejected_compile_never_changes_source_maps(bad):
    hz,kw,info,eq,le,plans,overlay=setup()
    before=(kw['eq_roots'].copy(),kw['eq_scales'].copy())
    pool=WorkPool(256_000_000)
    if bad=='order': plans=plans[::-1]
    elif bad=='definition': plans[0]=replace(plans[0],definition=plans[1].definition)
    elif bad=='consumer': plans[0]=replace(plans[0],consumer=plans[1].consumer)
    elif bad=='uid': plans[0]=replace(plans[0],producer_uid=plans[0].consumer_uid)
    elif bad=='pivot': plans[0]=replace(plans[0],pivot=1.5)
    elif bad=='offset': plans[0]=replace(plans[0],offset=float('nan'))
    elif bad=='tail': plans[0]=replace(plans[0],tail=(0,))
    elif bad=='consumer_main': plans[0]=replace(plans[0],consumer_main=plans[0].column)
    else: pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)): compiled(kw,plans,pool)
    assert np.array_equal(kw['eq_roots'],before[0]) and np.array_equal(kw['eq_scales'],before[1])


@pytest.mark.parametrize('field',['eq_roots','eq_scales','columns','retired','tails','hidden'])
def test_provisional_payload_mutation_rejects(field):
    hz,kw,info,eq,le,plans,overlay=setup(redirect=True)
    d=compiled(kw,plans)
    if field=='hidden': d.hidden=np.zeros(5)
    else: getattr(d,field)[0]+=1
    with pytest.raises(ValueError): d.numeric_roots()


def test_explicit_cost_contains_real_map_copies_and_all_sparse_metadata():
    hz,kw,info,eq,le,plans,overlay=setup(redirect=True)
    pool=WorkPool(256_000_000); d=compiled(kw,plans,pool)
    assert pool.parts['functional_lineage_map_copies']==2*len(kw['eq_roots'])
    assert len(d.columns)==len(plans)==len(d.retired)==len(d.tails)
    assert set(d.numeric_roots())=={'eq_roots','eq_scales','columns','retired','tails'}
    # New offsets live in EXISTING eq_scales slots, no C15 offset array.
    for p in plans:
        at=kw['old_n_eq']+p.column-kw['old_n_cont']
        assert d.eq_scales.view(np.float64)[at]==p.offset


def test_legacy_alias_extension_is_composed_after_unit_extension():
    hz,kw,info,eq,le,plans,overlay=setup(redirect=True)
    d=compiled(kw,plans); oracle,_=splice(hz,enabled=True,**kw)
    # Append an UNUSED global coordinate representing an already-eliminated
    # independent alias; no live consumer definition/output is overwritten.
    col=oracle.hz.n_cont
    d.eq_roots=np.r_[d.eq_roots,np.int64(-(plans[0].column+1))]
    d.eq_scales=np.r_[d.eq_scales,np.array([.5]).view(np.int64)]
    d.seal=d.fingerprint()
    base=oracle.hz
    pad=lambda m: sp.hstack((m,sp.csr_matrix((m.shape[0],1))),format='csr')
    expanded=SparseHZono(base.c,pad(base.Gc),base.Gb,pad(base.Ac),base.Ab,base.b,
        pad(base.Auc),base.Aub,base.ub,frame_id=base.frame_id,exact=True)
    point=[F(0)]*expanded.n_cont; point[0]=F(1,3)
    out=d.reconstruct_fraction(expanded,point,pool=WorkPool(256_000_000))
    assert out[col]==F(1,2)*out[plans[0].column]


@pytest.mark.parametrize('bad',[None,'root_reseal','scale_reseal','retired_reseal','tails_reseal','redirect_reseal','source','oracle'])
def test_complete_independent_audit_rejects_resealed_wrong_content(bad):
    from experiments.neural_hz_20260831.c22_uid_runs_v1 import pack
    from experiments.neural_hz_20260831.c26_transplant_audit_v1 import verify
    hz,kw,info,eq,le,plans,overlay=setup(redirect=True)
    d=compiled(kw,plans); oracle,_=splice(hz,enabled=True,**kw)
    oracle_fingerprint=oracle.fingerprint()
    closed=SimpleNamespace(**kw,hz=hz,validate=lambda:None,
        uid_slabs=np.asarray([pack(1,0,len(kw['eq_roots'])-1)],np.uint64))
    if bad=='root_reseal': d.eq_roots[1]^=np.int64(1<<27)
    elif bad=='scale_reseal': d.eq_scales.view(np.float64)[1]+=.125
    elif bad=='retired_reseal': d.retired[0]^=np.uint64(1)
    elif bad=='tails_reseal': d.tails[0]^=np.uint64(1)
    elif bad=='redirect_reseal': d.eq_roots[1+len(plans)]^=np.int64(1)
    elif bad=='source': hz.b[0]+=.125
    elif bad=='oracle': oracle.hz.Ac.data[0]+=.125; oracle.seal=oracle.fingerprint()
    d.seal=d.fingerprint()
    action=lambda:verify(closed,hz,overlay,np.asarray([p.column for p in plans]),eq,le,d,oracle,
        pool=WorkPool(256_000_000),expected_oracle_fingerprint=oracle_fingerprint)
    if bad is None:
        report=action()
        assert report['complete_transferred_incidence_equal'] and report['all_tagged_unit_pairs_checked']==len(plans)
    else:
        with pytest.raises(ValueError): action()
