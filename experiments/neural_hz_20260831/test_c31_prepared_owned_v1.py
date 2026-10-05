import copy
import hashlib
import pickle
import numpy as np
import pytest

from experiments.neural_hz_20260831.c31_prepared_owned_rows_v1 import make_encoder,PreparedOwnedEncoder
from experiments.neural_hz_20260831.c31_prepared_emission_v1 import lift
from experiments.neural_hz_20260831.c31_prepared_report_audit_v1 import audit
from experiments.neural_hz_20260831.c17_owned_rows_v1 import OwnedRowEncoder,WorkPool
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import Ledger,word
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool as DiagnosticPool
from experiments.neural_hz_20260831.c24_closed_state_v1 import close,export,restore
from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
from experiments.neural_hz_20260831.test_c24_dense_closed_v1 import fixture
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def encoder(cls,*,work=256_000_000,**kw):
    pool=WorkPool(0,0,max_work=work)
    # MAIN slots2,3 belong to current original row7. Protected slots0,1 and
    # radix slots>=4 never enter the MAIN ledger.
    ledger=Ledger(np.array([word(7),word(7)],np.int64),2,pool=pool)
    return cls(4,1,100,pool=pool,ledger=ledger,radix_uid_base=40,**kw),pool,ledger


def test_default_off_no_source_access():
    assert make_encoder(object(),enabled=False) is None
    assert lift(object(),object()) is None


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('inequality',[False,True])
@pytest.mark.parametrize('binary',[False,True])
def test_real_store_labels_and_radix_events_equal_original(packed,inequality,binary):
    actual,p,l=encoder(PreparedOwnedEncoder);old,q,m=encoder(OwnedRowEncoder)
    values=np.array([2.**-90 if packed else .25,-.5,1.])
    cc=np.array([0,2,3],np.int64)
    bc=np.array([0],np.int64) if binary else np.empty(0,np.int64)
    bv=np.array([.125]) if binary else np.empty(0)
    a=actual.encode_uid(7,cc,values,bc,bv,.125,inequality=inequality)
    b=old.encode_uid(7,cc,values,bc,bv,.125,inequality=inequality)
    assert a==b and actual.eq_uids==old.eq_uids and actual.ineq_uids==old.ineq_uids
    assert actual.def_rows==old.def_rows and actual.extra_work==old.extra_work
    assert np.array_equal(l.words,m.words) and l.updates==m.updates
    for rs,ss in ((actual.eq,old.eq),(actual.ineq,old.ineq)):
        assert len(rs)==len(ss)
        for r,s in zip(rs,ss):
            for x,y in zip(r[:4],s[:4]):assert np.array_equal(x.view(np.uint8),y.view(np.uint8))
            assert np.float64(r[4]).tobytes()==np.float64(s[4]).tobytes()
    for k,v in q.parts.items():assert p.parts[k]==v
    report=actual.discard_unpublished_heads()
    assert report['logical_input_rows']==1 and report['logical_input_coefficients']==3+len(bv)
    assert report['actual_preparation_attempts']==1+len(actual.def_rows)+actual.packed_rows
    assert actual.eq_heads is None and actual.ineq_heads is None
    with pytest.raises(ValueError):actual.encode_uid(7,cc,values,bc,bv,0.)
    with pytest.raises(ValueError):actual.discard_unpublished_heads()


def test_direct_fit_does_not_require_old_emit_hook(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('old emit hook incorrectly assumed')
    monkeypatch.setattr(OwnedRowEncoder,'emit',forbidden)
    enc,p,l=encoder(PreparedOwnedEncoder)
    enc.encode_uid(7,np.array([2,3]),np.array([-.5,1.]),np.empty(0,np.int64),np.empty(0),0.)
    assert enc.eq_uids==[7] and p.parts['ownership_physical_row_label']==1


@pytest.mark.parametrize('bad',['unowned','nested','uid','known_shift','work','aux','entry','head_count'])
def test_unowned_invalid_or_budget_exhausted_emission_fails_closed(bad):
    enc,p,l=encoder(PreparedOwnedEncoder,work=0 if bad=='work' else 256_000_000,
        **({'max_aux':0} if bad=='aux' else {'max_extra_entries':0} if bad=='entry' else {}))
    cc,cv=np.array([2,3]),np.array([-.5,1.]);empty=np.empty(0,np.int64);values=np.empty(0)
    if bad=='nested':enc.pending_uid=7
    if bad=='entry':enc.base_entries=0
    with pytest.raises((ValueError,MemoryError)):
        if bad=='unowned':enc.encode(cc,cv,empty,values,0.)
        elif bad=='known_shift':enc.emit(cc,cv,np.zeros(2,np.int64),empty,values,empty,0.,known_shift=0)
        elif bad=='head_count':enc.eq_heads.append(1);enc.discard_unpublished_heads()
        else:enc.encode_uid(-1 if bad=='uid' else 7,cc,np.array([2.**-90,1.]) if bad=='aux' else cv,empty,values,0.)


@pytest.mark.parametrize('kind',['ordinary','old_radix','main_radix','main_binary','shared_conv'])
def test_fresh_prepared_generator_full_new_math_proof_and_report(kind):
    old,original,maps=fixture(kind)
    old_closed,_=close(old,original.hz,maps,enabled=True)
    fields,old_raw=export(old_closed)
    expr_before=source_digest(old.hz)
    candidate=lift(old.expression,old.keep,enabled=True)
    proof=audit(candidate,old_closed,pool=DiagnosticPool(256_000_000))
    assert proof['all_report_fields_checked'] and proof['all_HZ_map_owner_UID_bits_equal']
    assert candidate.report['prepared_encoding']['direct_store_canonical_UID_hooks']
    assert not candidate.report['prepared_encoding']['head_metadata_retained_after_alias']
    assert source_digest(old.hz)==expr_before
    with pytest.raises(ValueError):bind(candidate,old_raw,expected_proof_sha256=hashlib.sha256(old_raw).hexdigest(),enabled=True)
    new,new_proof=close(candidate,original.hz,maps,enabled=True)
    assert new.fingerprint()!=old_closed.fingerprint()
    assert new_proof['main_boxes_and_alias_extension_proved']
    fields,raw=export(new)
    assert hashlib.sha256(raw).hexdigest()!=hashlib.sha256(old_raw).hexdigest()
    restored=restore(pickle.loads(pickle.dumps(fields)),raw,expected_proof_sha256=hashlib.sha256(raw).hexdigest())
    restored.validate();assert restored.fingerprint()==new.fingerprint()
    assert not new_proof['whole_live_path_proved']


@pytest.mark.parametrize('bad',['credit','attempts','total','branch','tariff','radix','head_retained','hidden','owner'])
def test_report_or_owner_mutation_cannot_use_reseal_as_proof(bad):
    old,original,maps=fixture();old_closed,_=close(old,original.hz,maps,enabled=True)
    candidate=lift(old.expression,old.keep,enabled=True)
    if bad=='credit':candidate.report['logical_coefficient_credit_preflight']+=1
    elif bad=='attempts':candidate.report['prepared_encoding']['actual_preparation_attempts']+=1
    elif bad=='total':candidate.report['total_work_upper']-=1
    elif bad=='branch':candidate.report['largest_branch_work_upper']-=1
    elif bad=='tariff':candidate.report['alias_quotient']['work_parts']['continuous_incidence_scan']-=1
    elif bad=='radix':candidate.report['radix_auxiliary_reserve']+=1
    elif bad=='head_retained':candidate.report['prepared_encoding']['head_metadata_retained_after_alias']=True
    elif bad=='hidden':candidate.report['unpriced_extra']=1
    else:candidate.owners[0]+=1
    candidate.seal=candidate.fingerprint()
    with pytest.raises(ValueError):audit(candidate,old_closed,pool=DiagnosticPool(256_000_000))


@pytest.mark.parametrize('kw',[{'max_work':0},{'max_branch_work':0},{'max_entries':0},
    {'max_work':256_000_001},{'max_branch_work':200_000_001},{'max_entries':64_000_001}])
def test_generator_caps_not_relaxed(kw):
    old,_,_=fixture()
    with pytest.raises((ValueError,MemoryError)):lift(old.expression,old.keep,enabled=True,**kw)
