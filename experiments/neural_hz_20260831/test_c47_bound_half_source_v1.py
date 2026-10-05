import copy
from fractions import Fraction as F
import gc
import hashlib
import pickle
import weakref
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.neural_hz_20260831 import c47_bound_half_source_v1 as binding
from experiments.neural_hz_20260831.c47_source_fixture_v1 import execute,feasible_point,_dot
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import unpack_descriptors
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect


def setup(monkeypatch,sign=1,layer=78):
    tf,runtime,old_hz,want=execute(monkeypatch,sign=sign,layer=layer)
    pool=WorkPool(256_000_000);old=runtime['lifted']
    state,report=binding.build(old,pool=pool,enabled=True)
    return tf,runtime,old,state,report,pool


@pytest.fixture(scope='module')
def frozen_source():
    # A test-only public C32 export, not a copied receipt. Adversarial tests
    # re-admit an independent deep copy against the unchanged proof bytes.
    from experiments.neural_hz_20260831.c32_splice_binding_v1 import export
    with pytest.MonkeyPatch.context() as local:
        _,runtime,_,_=execute(local)
        return copy.deepcopy(export(runtime['lifted']))


def cached_setup(frozen_source):
    from experiments.neural_hz_20260831.c32_splice_binding_v1 import admit
    old,_=admit(enabled=True,**copy.deepcopy(frozen_source))
    pool=WorkPool(256_000_000);state,report=binding.build(old,pool=pool,enabled=True)
    return old,state,report,pool


def predicates(hz,point,binary):
    return (all(_dot(hz.Ac,r,point)+_dot(hz.Ab,r,binary)==F(float(hz.b[r])) for r in range(hz.n_eq))
        and all(_dot(hz.Auc,r,point)+_dot(hz.Aub,r,binary)<=F(float(hz.ub[r])) for r in range(hz.n_ineq)))


def check_points(old,state,tf,layer,pool):
    desc=unpack_descriptors(state.half_words,pool=pool);results=[]
    for input_sign in (-1,1):
        for free in (-.5,0.,.5):
            expected,binary,values=feasible_point(old,tf,layer=layer,input_sign=input_sign,free=free,pool=pool)
            point=np.asarray([float(v) for v in expected],np.float64)
            if [F(float(v)) for v in point]!=expected:raise ValueError('synthetic exact point lost in f64')
            erased=list(map(int,old.lineage.columns))+list(map(int,desc['column']))
            erased += [old.lineage.old_n_cont+int(at)-old.lineage.old_n_eq for at in np.flatnonzero(old.lineage.eq_roots<0)]
            point[erased]=0.
            if not predicates(state.hz,[F(float(v)) for v in point],binary):raise ValueError('new point is infeasible')
            got,proof=state.reconstruct_fraction(point,pool=pool)
            if got!=expected:raise ValueError('bound source extension differs from independent original point')
            half_restored=[F(float(v)) for v in point]
            for t in desc:half_restored[int(t['column'])]=int(t['sign'])*half_restored[int(t['parent'])]/2
            if old.lineage.reconstruct_fraction(old.hz,half_restored,pool=pool)!=got:
                raise ValueError('original lineage reader and composed bound reader differ')
            if not predicates(old.hz,got,binary) or not predicates(old.original_fields['hz'],got,binary):
                raise ValueError('complete original affine/post predicate failed')
            out=[F(float(state.hz.c[r]))+_dot(state.hz.Gc,r,got)+_dot(state.hz.Gb,r,binary) for r in range(state.hz.n_out)]
            if out!=[max(F(0),v) for v in values]:raise ValueError('exact concrete toy ReLU output differs')
            results.append(dict(input_sign=input_sign,free=free,
                complete_fraction_point_sha256=hashlib.sha256(str(got).encode()).hexdigest(),
                pre_old_new_EQ_INEQ_and_binary_boxes_passed=True,complete_original_reader_equal=True,
                exact_toy_ReLU_output_equal=True,proof=proof))
    return results


def test_default_off_does_not_inspect_inputs():
    assert binding.build(object(),pool=object()) is None


@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('layer',[2,78,1001])
def test_issued_source_complete_equivalence_and_six_exact_feasible_points(monkeypatch,sign,layer):
    tf,runtime,old,state,report,pool=setup(monkeypatch,sign,layer)
    assert report['source_proof_is_test_only'] and report['source_association_proved_relative_to_issued_C32_chain']
    assert report['half_count']==1 and report['derived_original_incidence_degrees']==[7]
    assert state.hz.n_eq==old.hz.n_eq-1 and state.hz.n_bin==old.hz.n_bin==9
    assert state.lineage is old.lineage and state.events is old.events
    assert state.receipt is not old.receipt and state.hz is not old.hz
    assert not isinstance(state,SimpleNamespace) and type(state) is binding.BoundHalfState
    assert state.validate(pool=pool)==report
    assert len(check_points(old,state,tf,layer,pool))==6
    assert not report['native_admission_proved'] and not report['original_network_input_or_property_binding_proved']
    assert not report['complete_live_or_runtime_payment_proved'] and report['formal_gain']==0


@pytest.mark.parametrize('bad',['source','source_proof','transfer','lineage','events','new_data','row_shape',
    'column_shape','pointer','index','index_duplicate','zero','frame','words','runs','report','old_report',
    'receipt','hidden','external_buffer','map_identity','geometry','kernel'])
def test_mutation_geometry_and_forged_success_fail_closed(frozen_source,bad):
    old,state,_,pool=cached_setup(frozen_source)
    if bad=='source':state.original_fields['hz'].b[0]+=.125
    elif bad=='source_proof':state.source_proof_bytes+=b' '
    elif bad=='transfer':state.transfer_proof_bytes+=b' '
    elif bad=='lineage':state.lineage.eq_roots[0]+=1
    elif bad=='events':
        state.events=state.events.copy();state.events[0]+=np.uint64(1)
    elif bad=='new_data':state.hz.Ac.data[0]+=.125
    elif bad=='row_shape':state.hz.Ac._shape=(state.hz.n_eq+1,state.hz.n_cont)
    elif bad=='column_shape':state.hz.Ab._shape=(state.hz.n_eq,state.hz.n_bin+1)
    elif bad=='pointer':state.hz.Ac.indptr[-1]-=1
    elif bad=='index':state.hz.Ac.indices[0]=state.hz.n_cont
    elif bad=='index_duplicate':
        m=state.hz.Ac;row=int(np.flatnonzero(np.diff(m.indptr)>1)[0]);a=int(m.indptr[row])
        m.indices[a+1]=m.indices[a];m.has_canonical_format=True;m.has_sorted_indices=True
    elif bad=='zero':state.hz.Ac.data[0]=0.
    elif bad=='frame':state.hz.frame_id+=1
    elif bad=='words':state.half_words[1]^=np.uint64(1<<38)
    elif bad=='runs':state.gauge_runs[0]+=np.uint64(1)
    elif bad=='report':state.transaction_report['complete_functional_HZ_and_inverse_passed']=False
    elif bad=='old_report':state.construction_report['native_payload_work']=0
    elif bad=='receipt':state.receipt=old.receipt
    elif bad=='hidden':state.hidden=np.zeros(3)
    elif bad=='external_buffer':state.half_words=np.frombuffer(state.half_words.tobytes(),dtype=np.uint64)
    elif bad=='map_identity':state.original_fields['eq_roots']=state.original_fields['eq_roots'].copy()
    elif bad=='kernel':state.original_fields['expression'].terms[0].operators[0]._kernel.flat[0]+=.125
    else:state.original_post_geometry=(*state.original_post_geometry[:4],999,7)
    with pytest.raises((ValueError,MemoryError)):state.validate(pool=pool)


def test_receipt_is_not_replayable_copy_or_portable_certificate(frozen_source):
    _,state,_,pool=cached_setup(frozen_source)
    with pytest.raises(ValueError,match='different'):copy.copy(state).validate(pool=pool)
    with pytest.raises(TypeError):pickle.dumps(state)
    with pytest.raises(ValueError):binding._Receipt(object())


def test_old_state_and_receipt_can_die_without_breaking_source_association(monkeypatch):
    tf,runtime,old,state,report,pool=setup(monkeypatch)
    old_ref=weakref.ref(old);hz_ref=weakref.ref(old.hz);receipt_ref=weakref.ref(old.receipt)
    del tf,runtime,old
    gc.collect()
    assert old_ref() is None and hz_ref() is None and receipt_ref() is None
    assert state.validate(pool=pool)==report


@pytest.mark.parametrize('bad',['unissued','wrong_type','whole_budget','separate_branch'])
def test_bad_construction_returns_no_new_authority_and_does_not_mutate_source(monkeypatch,bad):
    tf,runtime,hz,want=execute(monkeypatch);old=runtime['lifted'];pool=WorkPool(256_000_000)
    before=source_digest(hz);branch=None
    if bad=='unissued':old.receipt=None
    elif bad=='wrong_type':old=SimpleNamespace(**vars(old))
    elif bad=='whole_budget':pool=WorkPool(1)
    else:
        from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
        branch=BranchPool(WorkPool(256_000_000))
    with pytest.raises((ValueError,MemoryError)):binding.build(old,pool=pool,branch=branch,enabled=True)
    assert source_digest(hz)==before


def test_complete_registered_state_roots_are_visible_to_existing_strict_owner_ledger(frozen_source):
    old,state,_,pool=cached_setup(frozen_source)
    roots=state.numeric_roots(pool=pool)
    assert set(roots)==(set(vars(state))-{'receipt'})|{'issued_source_record'}
    measured=collect(SimpleNamespace(),roots).measure()
    assert measured.resident_bytes>0 and measured.resident_entries>0
    assert state.hz is roots['hz'] and old.hz is not roots['hz']


@pytest.mark.parametrize('bad',['short','float32','nan','box','budget'])
def test_bound_witness_requires_complete_finite_box_vector(frozen_source,bad):
    _,state,_,pool=cached_setup(frozen_source);point=np.zeros(state.hz.n_cont,np.float64)
    if bad=='short':point=point[:-1]
    elif bad=='float32':point=point.astype(np.float32)
    elif bad=='nan':point[0]=np.nan
    elif bad=='box':point[0]=2.
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):state.reconstruct_fraction(point,pool=pool)


def test_original_hash_does_not_replace_explicit_CSR_shape_binding(frozen_source):
    _,state,report,pool=cached_setup(frozen_source)
    state.hz.Ab._shape=(state.hz.n_eq,state.hz.n_bin+1)
    assert source_digest(state.hz)==report['new_post_sha256']
    with pytest.raises(ValueError,match='geometry'):state.validate(pool=pool)


@pytest.mark.parametrize('used',[255_056_237,255_618_253])
def test_observed_whole_paths_cannot_pay_even_one_new_map_check(used):
    from experiments.neural_hz_20260831.c47_source_payment_floor_v1 import append_screen
    result=append_screen(used,244312,26)
    assert result['one_mandatory_metadata_check_work']==3_910_912
    assert result['append_impossible_from_lower_bound'] and not result['actual_attempt_authorized']


def test_lower_bound_that_fits_never_authorizes_a_full_path():
    from experiments.neural_hz_20260831.c47_source_payment_floor_v1 import append_screen
    result=append_screen(0,244312,26)
    assert not result['append_impossible_from_lower_bound']
    assert not result['actual_attempt_authorized'] and not result['complete_runtime_payment_proved']
    with pytest.raises(ValueError):append_screen(0,244312,26,cap=256_000_001)
