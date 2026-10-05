"""Complete fresh source equivalence on ordinary nonconvex dense Conv graphs."""
from fractions import Fraction as F
import copy
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_cnn import SparseHZAffineExpr, SparseHZAffineTerm
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift as old_lift
from experiments.neural_hz_20260831.c98_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c98_source_audit_v1 import audit,expression_key
from experiments.neural_hz_20260831.c98_circuit_stream_v1 import install,matrices
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c90_actual_circuit_proof_v1 import prove
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import row,extend,recover
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c65_physical_archive_v1 import check_restored
from experiments.neural_hz_20260831.c62_local_equations_v1 import reconstruct,SCHEMA


def expression(c=16,k=32,h=4):
    n=c*h*h
    gc=sp.diags(np.resize([.25,.5],n),format='csr')
    gb=sp.csr_matrix((np.full(n,.125),(np.arange(n),np.zeros(n,int))),shape=(n,1))
    ac=sp.csr_matrix(([1.],([0],[0])),shape=(1,n));ab=sp.csr_matrix([[-.25]])
    auc=sp.csr_matrix(([1.],([0],[1])),shape=(1,n))
    src=SparseHZono(np.full(n,.03125),gc,gb,ac,ab,np.zeros(1),
        auc,sp.csr_matrix([[.125]]),np.ones(1),frame_id=17,exact=True)
    weights=np.broadcast_to(np.array([[1,2,4],[2,4,8],[4,8,16]],np.float64)/32,(k,c,3,3)).copy()
    op=ImplicitConv2DOp(weights,(1,c,h,h))
    # The source remains truly binary/nonconvex and every parent is multi-term.
    d1=sp.diags(np.full(op.shape[0],.75),format='csr')
    d2=sp.diags(np.full(op.shape[0],.5),format='csr')
    return SparseHZAffineExpr((SparseHZAffineTerm(src,(op,d1,d2)),),np.zeros(op.shape[0]),op.shape[0],src.frame_id)


@pytest.mark.parametrize('c,k,h',[(16,32,4),(32,32,4),(16,32,5)])
def test_complete_original_expression_single_CSR_and_independent_equations(c,k,h):
    expr=expression(c,k,h);keep=np.ones(expr.n_out,bool);pool=WorkPool(256_000_000)
    old=old_lift(expr,keep,enabled=True)['fields']
    draft=lift(expr,keep,enabled=True);state=draft['state'];fields=state['fields']
    proof=audit(old,state,draft['construction']['circuits'],pool=pool,enabled=True)
    assert proof['complete_original_rows_checked']==old['hz'].n_eq
    assert proof['original_binary_factors']==1 and proof['all_original_maps_and_other_predicates_preserved']
    report=fields['report'];assert report['circuit_generation']['one_final_CSR']
    assert not report['circuit_generation']['old_source_CSR_constructed']
    assert fields['hz'].Ac.nnz<old['hz'].Ac.nnz and fields['hz'].n_bin==old['hz'].n_bin
    assert report['total_work_upper']-old['report']['total_work_upper']==report['circuit_generation']['new_work']
    for p in draft['construction']['circuits']:
        originals=[];gauges=[]
        for pivot in p['pivots'][p['new_factors']:]:
            rank=old['old_n_eq']+int(pivot)-old['old_n_cont'];r=int(old['eq_roots'][rank])
            columns,values=row(old['hz'].Ac,r)
            originals.append(dict(coefficients=list(zip(columns,values)),rhs=old['hz'].b[r],pivot=int(pivot)))
            gauges.append(int(old['eq_scales'][rank]))
        result=prove(p,originals,gauges,old_n_cont=old['hz'].n_cont,
            first_aux=int(p['pivots'][0]),new_factors=p['new_factors'],pool=pool,enabled=True)
        assert result['original_source_equivalence'] and result['universal_unique_box_extension']
    expected=actual_words(fields['hz'],fields['old_n_cont'],fields['logical_n_cont'],
        draft['construction']['eq_uids'],draft['construction']['ineq_uids'])
    assert np.array_equal(fields['owners'],expected)
    point=[F((i%5)-2,8) for i in range(old['hz'].n_cont)]
    expected_point=reconstruct(point,old['eq_roots'],old['eq_scales'],old_n_cont=old['old_n_cont'],
        old_n_eq=old['old_n_eq'],n_cont=old['hz'].n_cont,schema=SCHEMA)
    expanded=extend(state,point,pool=pool);assert recover(state,expanded,pool=pool)==expected_point
    with pytest.raises(ValueError):
        check_restored(dict(schema=state['schema'],native_or_LIVE_admission=False))


def test_portable_semantic_binding_includes_all_coefficients_and_sharing():
    expr=expression();clone=copy.deepcopy(expr);pool=WorkPool(256_000_000)
    assert expression_key(expr,pool)==expression_key(clone,pool)
    clone.terms[0].source.Gc.data[0]*=2
    assert expression_key(expr,pool)!=expression_key(clone,pool)


def test_default_off_and_source_work_caps():
    assert lift(None,None) is None and install(None,None,None,None,old_nc=0,old_neq=0) is None
    expr=expression()
    with pytest.raises(MemoryError):lift(expr,np.ones(expr.n_out,bool),enabled=True,max_work=0)
    with pytest.raises(ValueError):matrices(object(),[])


@pytest.mark.parametrize('field',['unchanged_row','new_row','owner','binary','inverse'])
def test_independent_full_source_audit_rejects_actual_changed_state(field):
    expr=expression();keep=np.ones(expr.n_out,bool)
    old=old_lift(expr,keep,enabled=True)['fields'];draft=lift(expr,keep,enabled=True)
    state=draft['state'];fields=state['fields']
    if field=='unchanged_row':fields['hz'].Ac.data[0]*=2
    if field=='new_row':fields['hz'].Ac.data[-1]*=2
    if field=='owner':state['auxiliary_records'][0,5]+=1
    if field=='binary':
        fields['hz'].Ab=fields['hz'].Ab.copy();fields['hz'].Ab.data[0]*=2
    if field=='inverse':fields['eq_roots']=fields['eq_roots'].copy();fields['eq_roots'][-1]+=1
    with pytest.raises(ValueError):audit(old,state,draft['construction']['circuits'],pool=WorkPool(256_000_000),enabled=True)
