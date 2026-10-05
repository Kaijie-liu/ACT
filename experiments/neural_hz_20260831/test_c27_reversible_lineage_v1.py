from dataclasses import replace
from fractions import Fraction as F
import hashlib
import gc
import pickle
import sys
import tracemalloc
import weakref
import numpy as np
import pytest

from experiments.neural_hz_20260831 import c27_reversible_lineage_v1 as t
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import compile_lineage
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays
from experiments.neural_hz_20260831.test_c26_tagged_transplant_v1 import setup
from experiments.neural_hz_20260831.test_c25_live_closed_v1 import proof_fixture
from experiments.neural_hz_20260831.c15_unit_row_splice_v1 import splice


def permission(kw):
    # Toy-owned arrays are born in setup(), never restored or published.
    # Real callers can obtain a permit only from t.generate().
    return t._FreshPermit(t._FRESH_KEY,kw['eq_roots'],kw['eq_scales'])


def compile_test(kw,plans,permit,pool):
    return t.compile_owned(kw['eq_roots'],kw['eq_scales'],plans,old_n_cont=kw['old_n_cont'],
        old_n_eq=kw['old_n_eq'],pool=pool,ownership=permit,enabled=True)


def test_default_off_no_input_or_permission_read():
    assert t.generate(object(),object()) is None
    assert t.compile_owned(object(),object(),object(),old_n_cont=object(),old_n_eq=object(),pool=object()) is None


@pytest.mark.parametrize('old_scale',[-8192,-1,0,1,8191])
@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('kind',[False,True])
def test_all_tag_fields_invert_exactly(old_scale,sign,kind):
    raw=t.encode_splice(91,101,kind,2.**-20,sign,old_scale)
    assert t.decode(raw)==('splice',91,101,kind,2.**-20,sign)
    assert t.original_entry(raw,123)==(91,old_scale)
    redirect=t.REDIRECT|(371<<20)|629
    assert t.decode(redirect)==('redirect',629)
    assert t.original_entry(redirect,-17)==(371,-17)


@pytest.mark.parametrize('bad',[-8193,8192,1.5])
def test_unrepresentable_old_scale_rejected_before_encoding(bad):
    with pytest.raises(ValueError):t.encode_splice(1,2,False,1.,1,bad)


@pytest.mark.parametrize('mixed,subtract,redirect',[(False,False,False),(True,True,False),(False,True,True)])
def test_owned_edit_has_same_math_and_no_full_map_clone(mixed,subtract,redirect):
    hz,kw,info,eq,le,plans,overlay=setup(mixed=mixed,subtract=subtract,redirect=redirect)
    old_digest=digest_arrays(kw['eq_roots'],kw['eq_scales'])
    functional=compile_lineage(kw['eq_roots'],kw['eq_scales'],plans,old_n_cont=kw['old_n_cont'],
        old_n_eq=kw['old_n_eq'],pool=WorkPool(256_000_000),enabled=True)
    oracle,_=splice(hz,enabled=True,**kw)
    ids=(id(kw['eq_roots']),id(kw['eq_scales']))
    permit=permission(kw); pool=WorkPool(256_000_000)
    result=compile_test(kw,plans,permit,pool)
    assert (id(result.eq_roots),id(result.eq_scales))==ids and permit.used
    assert 'functional_lineage_map_copies' not in pool.parts
    assert t.inverse_digest(result.eq_roots,result.eq_scales)==old_digest
    for a,b in zip(result.eq_roots,functional.eq_roots):
        masked=int(a)&(t.SPLICE|((1<<48)-1)) if a>=t.SPLICE else (t.REDIRECT|(int(a)&t.MASK) if a>=t.REDIRECT else int(a))
        assert masked==int(b)
    assert np.array_equal(result.eq_scales,functional.eq_scales)
    assert list(result.iter_words(overlay,pool=pool))==list(functional.iter_words(overlay,pool=pool))
    point=[F(0)]*hz.n_cont; point[0]=F(1,3)
    assert result.reconstruct_fraction(oracle.hz,point,pool=pool)==functional.reconstruct_fraction(oracle.hz,point,pool=pool)
    copy=pickle.loads(pickle.dumps(result,protocol=5));copy.validate()
    assert t.inverse_digest(copy.eq_roots,copy.eq_scales)==old_digest
    with pytest.raises((TypeError,ValueError)):pickle.dumps(permit)
    with pytest.raises(ValueError):compile_test(kw,plans,permit,pool)


@pytest.mark.parametrize('kind',['missing','wrong','strong_alias','view','unowned'])
def test_nonexclusive_or_unissued_arrays_cannot_be_edited(kind):
    hz,kw,info,eq,le,plans,overlay=setup()
    original=digest_arrays(kw['eq_roots'],kw['eq_scales'])
    permit=permission(kw)
    if kind=='missing':permit=None
    elif kind=='wrong':permit=t._FreshPermit(t._FRESH_KEY,kw['eq_roots'].copy(),kw['eq_scales'].copy())
    elif kind=='strong_alias':extra=kw['eq_roots']
    elif kind=='view':extra=kw['eq_roots'].view()
    else:kw['eq_roots']=kw['eq_roots'].view();permit=permission(kw)
    with pytest.raises(ValueError):compile_test(kw,plans,permit,WorkPool(256_000_000))
    assert digest_arrays(kw['eq_roots'],kw['eq_scales'])==original


def test_failure_poisoned_permit_cannot_retry_transaction():
    hz,kw,info,eq,le,plans,overlay=setup()
    permit=permission(kw); plans[0]=replace(plans[0],pivot=1.5)
    with pytest.raises(ValueError):compile_test(kw,plans,permit,WorkPool(256_000_000))
    assert permit.used
    plans[0]=replace(plans[0],pivot=1.)
    with pytest.raises(ValueError):compile_test(kw,plans,permit,WorkPool(256_000_000))


def test_full_inverse_stream_hash_has_no_full_array_scratch_copy():
    roots=np.arange(400000,dtype=np.int64); scales=np.zeros(400000,np.int64)
    old=digest_arrays(roots,scales)
    roots[9]=t.encode_splice(9,201,False,2.,1,0); scales.view(np.float64)[9]=.25
    roots[201]=t.REDIRECT|(201<<20)|101
    tracemalloc.start()
    actual=t.inverse_digest(roots,scales)
    current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert actual==old and peak<100000


def test_actual_fresh_generation_issues_nonserializable_permit():
    expr,raw,sha=proof_fixture()
    draft,permit=t.generate(expr,np.ones(expr.n_out,bool),enabled=True,frame_widths=(12,6))
    assert permit.roots() is draft.eq_roots and permit.scales() is draft.eq_scales and not permit.used
    with pytest.raises(ValueError):t._FreshPermit(object(),draft.eq_roots,draft.eq_scales)
    with pytest.raises(TypeError):pickle.dumps(permit)


def test_real_fresh_binding_export_retirement_then_exclusive_edit():
    from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
    from experiments.neural_hz_20260831.c24_closed_state_v1 import export
    from experiments.neural_hz_20260831.c27_source_image_v1 import verify
    expr,raw,sha=proof_fixture()
    draft,permit=t.generate(expr,np.ones(expr.n_out,bool),enabled=True,frame_widths=(12,6))
    closed,_=bind(draft,raw,expected_proof_sha256=sha,enabled=True)
    values,raw=export(closed)
    slot=next(i for i in range(closed.old_n_eq,len(closed.eq_roots)) if closed.eq_roots[i]>0)
    # Ownership/inversion fixture only; no new-predicate proof is asserted.
    plan=t.Plan(column=closed.old_n_cont+slot-closed.old_n_eq,definition=int(closed.eq_roots[slot]),
        consumer=0,inequality=False,producer_uid=1,consumer_uid=2,pivot=1.,sign=1,
        offset=.125,consumer_main=None,tail=())
    old_digest=digest_arrays(values['eq_roots'],values['eq_scales'])
    refs=[weakref.ref(v) for v in (closed,draft)]
    del closed,draft
    gc.collect()
    assert all(ref() is None for ref in refs)
    journal=t.compile_owned(values['eq_roots'],values['eq_scales'],[plan],old_n_cont=values['old_n_cont'],
        old_n_eq=values['old_n_eq'],pool=WorkPool(256_000_000),ownership=permit,enabled=True)
    assert journal.eq_roots is values['eq_roots']
    assert t.inverse_digest(journal.eq_roots,journal.eq_scales)==old_digest
    assert not verify(values,raw,expected_proof_sha256=sha)['new_splice_math_proved_by_this_check']


@pytest.mark.parametrize('bad',[None,'new_offset','target','columns','retired','tail','reference_reseal'])
def test_independent_new_semantics_cannot_be_replaced_by_original_image(bad):
    from experiments.neural_hz_20260831.c27_reference_transfer_v1 import verify
    hz,kw,info,eq,le,plans,overlay=setup(redirect=True)
    reference=compile_lineage(kw['eq_roots'],kw['eq_scales'],plans,old_n_cont=kw['old_n_cont'],
        old_n_eq=kw['old_n_eq'],pool=WorkPool(256_000_000),enabled=True)
    expected=reference.fingerprint()
    candidate=compile_test(kw,plans,permission(kw),WorkPool(256_000_000))
    at=candidate.old_n_eq+int(candidate.columns[0])-candidate.old_n_cont
    if bad=='new_offset':candidate.eq_scales[at]^=1
    elif bad=='target':candidate.eq_roots[at]^=1
    elif bad=='columns':candidate.columns[0]+=1
    elif bad=='retired':candidate.retired[0]^=np.uint64(1)
    elif bad=='tail':candidate.tails[0]^=np.uint64(1)
    elif bad=='reference_reseal':
        reference.eq_scales[at]^=1;reference.seal=reference.fingerprint()
    candidate.seal=candidate.fingerprint()
    action=lambda:verify(candidate,reference,expected_reference_fingerprint=expected,pool=WorkPool(256_000_000))
    if bad is None:assert action()['all_selected_columns_compared']==len(plans)
    else:
        with pytest.raises(ValueError):action()


@pytest.mark.parametrize('bad',[None,'old_scale','old_row','source','hz','owners','slabs','map','report','proof'])
def test_complete_original_image_transfer_not_just_map_hash(bad):
    from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
    from experiments.neural_hz_20260831.c24_closed_state_v1 import export
    from experiments.neural_hz_20260831.c27_source_image_v1 import verify
    expr,raw,sha=proof_fixture()
    draft,permit=t.generate(expr,np.ones(expr.n_out,bool),enabled=True,frame_widths=(12,6))
    closed,_=bind(draft,raw,expected_proof_sha256=sha,enabled=True)
    values,raw=export(closed)
    slot=next(i for i in range(closed.old_n_eq,len(closed.eq_roots)) if closed.eq_roots[i]>=0)
    original_row=int(values['eq_roots'][slot]); original_scale=int(values['eq_scales'][slot])
    # Test only inversion/authentication here; arbitrary new-unit semantics
    # are intentionally NOT certified by this original-image checker.
    values['eq_roots'][slot]=t.encode_splice(original_row,original_row,False,1.,1,original_scale)
    values['eq_scales'].view(np.float64)[slot]=.125
    if bad=='old_scale':values['eq_roots'][slot]^=np.int64(1<<48)
    elif bad=='old_row':values['eq_roots'][slot]^=np.int64(1<<28)
    elif bad=='source':expr.bias[0]+=.125
    elif bad=='hz':values['hz'].b[0]+=.125
    elif bad=='owners':values['owners'][0]+=1
    elif bad=='slabs':values['uid_slabs'][0]+=np.uint64(1)
    elif bad=='map':values['ineq_scales'][0]+=1
    elif bad=='report':values['report']['total_work_upper']+=1
    elif bad=='proof':raw+=b' '
    action=lambda:verify(values,raw,expected_proof_sha256=sha)
    if bad is None:
        result=action()
        assert not result['full_old_map_arrays_allocated']
        assert not result['new_splice_math_proved_by_this_check']
        # Correct inversion alone says nothing about the NEW offset.
        values['eq_scales'].view(np.float64)[slot]=.375
        assert action()['complete_original_source_image_sha256']==result['complete_original_source_image_sha256']
    else:
        with pytest.raises(ValueError):action()
