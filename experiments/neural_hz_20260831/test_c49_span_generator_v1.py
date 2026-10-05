import copy
import hashlib
import pickle
from fractions import Fraction as F
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c49_span_emission_v1 import lift
from experiments.neural_hz_20260831.c49_span_report_audit_v1 import audit
from experiments.neural_hz_20260831.c31_prepared_emission_v1 import lift as old_lift
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c24_closed_state_v1 import close,export,restore
from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c10_portable_binding_v1 import reconstruct_fraction
from experiments.neural_hz_20260831.test_c24_dense_closed_v1 import fixture


def wide_fixture():
    width=128
    gc=(sp.eye(width,format='csr')+sp.csr_matrix((np.ones(width),
        (np.arange(width),(np.arange(width)+1)%width)),shape=(width,width)))*.25
    hz=SparseHZono(np.zeros(width),gc,sp.csr_matrix((width,1)),
        sp.csr_matrix(([.5],([0],[0])),shape=(1,width)),sp.csr_matrix([[-.5]]),np.zeros(1),
        sp.csr_matrix(([.5],([0],[1])),shape=(1,width)),sp.csr_matrix([[.5]]),np.ones(1),frame_id=149)
    dense=sp.csr_matrix(np.ones((width,width))/256.)
    diagonal=sp.diags(np.where(np.arange(width)%2,.5,-.5),format='csr')
    tail=sp.csr_matrix(np.ones((3,width))/256.)
    expr=cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(hz,(dense,diagonal,tail)),),np.zeros(3),3,hz.frame_id)
    original=original_lift(expr,np.ones(3,bool),enabled=True)
    maps={k:getattr(original,k) for k in ('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows')}
    return original,maps


def pair(kind):
    if kind=='wide_miss_and_hit':original,maps=wide_fixture()
    else:_,original,maps=fixture(kind)
    old=old_lift(original.expression,original.keep,enabled=True)
    old_closed,_=close(old,original.hz,maps,enabled=True)
    return original,maps,old_closed,lift(original.expression,original.keep,enabled=True)


def test_default_off_no_input_access():
    assert lift(object(),object()) is None
    assert audit(object(),object(),object(),object(),pool=object()) is None


@pytest.mark.parametrize('kind',['ordinary','old_radix','main_radix','main_binary','shared_conv','wide_miss_and_hit'])
def test_fresh_generator_all_original_bits_new_close_and_full_vector_reconstruction(kind):
    original,maps,old,new=pair(kind)
    whole=WorkPool(256_000_000);branch=BranchPool(whole)
    report=audit(new,old,original.hz,maps,pool=branch,enabled=True)
    assert report['all_report_fields_checked'] and report['all_HZ_map_owner_UID_bits_equal']
    assert whole.used==branch.used
    if kind=='wide_miss_and_hit':
        route=new.report['alias_quotient']['source_span_routing']
        assert route['proved_empty_rows']>=128 and report['strict_component_payment']
        assert new.report['alias_quotient']['alias_products_checked']>0
    _,old_raw=export(old)
    with pytest.raises(ValueError):bind(new,old_raw,expected_proof_sha256=hashlib.sha256(old_raw).hexdigest(),enabled=True)
    closed,proof=close(new,original.hz,maps,enabled=True)
    assert proof['main_boxes_and_alias_extension_proved']
    assert proof['all_MAIN_ownership_checked']==len(new.owners)
    fields,raw=export(closed);assert raw!=old_raw
    restored=restore(pickle.loads(pickle.dumps(fields)),raw,expected_proof_sha256=hashlib.sha256(raw).hexdigest())
    restored.validate()
    for v in (F(-1),F(0),F(1,2),F(1)):
        x=[v]*new.hz.n_cont
        assert reconstruct_fraction(restored,x)==reconstruct_fraction(old,x)
    assert not report['native_or_full_LIVE_payment_proved'] and not proof['whole_live_path_proved']


@pytest.mark.parametrize('bad',['total','branch','scan','rows','misses','seal_cost','private_cost',
    'reserve','products','hidden','owner','old_reference','oracle_map'])
def test_report_reseal_or_old_receipt_never_replaces_complete_independent_proof(bad):
    original,maps,old,new=pair('ordinary');q=new.report['alias_quotient']
    if bad=='total':new.report['total_work_upper']-=1
    elif bad=='branch':new.report['largest_branch_work_upper']-=1
    elif bad=='scan':q['work_parts']['continuous_incidence_scan']-=1
    elif bad=='rows':q['source_span_routing']['rows']-=1
    elif bad=='misses':q['source_span_routing']['proved_empty_rows']+=1
    elif bad=='seal_cost':q['work_parts']['c48_successor_full_construction_seals_and_retirement']-=1
    elif bad=='private_cost':q['work_parts']['c49_private_source_boundary_and_complete_routing_report']-=1
    elif bad=='reserve':new.report['source_span_index_transient_reserve_bytes']-=1
    elif bad=='products':q['product_certification']['hits']-=1
    elif bad=='hidden':q['unpriced_extra']=1
    elif bad=='owner':new.owners[0]+=1
    elif bad=='oracle_map':maps=dict(maps,eq_roots=maps['eq_roots'].copy());maps['eq_roots'][0]=maps['eq_roots'][-1]
    else:old=close(new,original.hz,maps,enabled=True)[0]
    new.seal=new.fingerprint()
    with pytest.raises((ValueError,MemoryError)):
        audit(new,old,original.hz,maps,pool=BranchPool(WorkPool(256_000_000)),enabled=True)


@pytest.mark.parametrize('kw',[{'max_work':0},{'max_branch_work':0},{'max_entries':0},
    {'max_work':256_000_001},{'max_branch_work':200_000_001},{'max_entries':64_000_001}])
def test_all_original_generation_caps_unchanged(kw):
    _,original,_=fixture()
    with pytest.raises((ValueError,MemoryError)):lift(original.expression,original.keep,enabled=True,**kw)
