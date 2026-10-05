"""Ordinary complete native equations, new UID tails, and nonzero inverses."""
from dataclasses import asdict
from fractions import Fraction as F
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c22_uid_runs_v1 import pack
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA as LOCAL
from experiments.neural_hz_20260831.c70_native_proof_v1 import extract,digest,same_matrix
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import SCHEMA,extend
from experiments.neural_hz_20260831.c99_circuit_consumer_v1 import CircuitSource,resolve,closed_uid_tables,inject_phase,bind_phase
from experiments.neural_hz_20260831.c99_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c99_circuit_journal_v1 import compile_journal,reconstruct
from experiments.neural_hz_20260831.c99_native_proof_v1 import verify,factor_plans
from experiments.neural_hz_20260831.c99_writer_bound_v1 import writer_bound
from experiments.neural_hz_20260831.c32_boundary_budget_v1 import WriterPool
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool as CoupledPool
from experiments.neural_hz_20260831.c98_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift as old_lift
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression


def pool():return WorkPool(256_000_000)


def packet(old):
    h=old.hz
    slots=[(h.n_cont+2*i,h.n_cont+2*i+1,h.n_bin+i) for i in range(h.n_out)]
    post=sparse_hz_apply_relu_exact(h,np.full(h.n_out,-8.),np.full(h.n_out,8.),
        slots,h.n_cont+2*h.n_out,h.n_bin+h.n_out)
    return extract(h,post,old_n_cont=old.old_n_cont,old_n_eq=old.old_n_eq,
        logical_n_cont=old.logical_n_cont,first_uid=old.report['radix_uid_base']+16384,
        provenance={'fixture':True},pool=pool(),enabled=True)


def fixture():
    old=SparseHZono(np.zeros(1),sp.csr_matrix([[1.,0.]]),sp.csr_matrix((1,1)),
        sp.csr_matrix([[1.,0.],[-.25,1.]]),sp.csr_matrix([[-.25],[0.]]),np.array([0.,.125]),
        sp.csr_matrix([[0.,.25]]),sp.csr_matrix([[.125]]),np.ones(1),frame_id=23,exact=True)
    # The inequality uses input 0, leaving MAIN 1 exactly two consumers after
    # adding its redundant circuit copy. Both binary EQ and INEQ stay present.
    old.Auc=sp.csr_matrix([[.25,0.]])
    fields=dict(hz=old,old_n_cont=1,old_n_eq=1,logical_n_cont=2,
        eq_roots=np.array([0,1],np.int64),eq_scales=np.zeros(2,np.int64),
        ineq_roots=np.array([0],np.int64),ineq_scales=np.zeros(1,np.int64),
        def_rows=np.empty(0,np.int64),uid_slabs=np.array([pack(2,0,1)],np.uint64),
        report=dict(radix_uid_base=3,new_lineage_schema=LOCAL))
    p=packet(SimpleNamespace(**fields))
    new=SparseHZono(old.c,sp.csr_matrix([[1.,0.,0.]]),old.Gb,
        sp.csr_matrix([[1.,0.,0.],[-.25,1.,0.],[0.,-1.,1.]]),
        sp.csr_matrix([[-.25],[0.],[0.]]),np.r_[old.b,0.],
        sp.csr_matrix([[.25,0.,0.]]),old.Aub,old.ub,frame_id=23,exact=True)
    fields=dict(fields,hz=new)
    aux=np.array([[2,3,2,np.array(1.).view(np.int64).item(),0,0,
        np.array(-1.).view(np.int64).item(),np.array(1.).view(np.int64).item()]],np.int64)
    state=dict(schema=SCHEMA,fields=fields,old_source_n_cont=2,old_source_n_eq=2,
        auxiliary_records=aux,output_routes=np.empty((0,3),np.int64),block_records=np.empty((0,8),np.int64),
        original_source_proof=b'fixture',original_circuit_proof=b'fixture',native_or_LIVE_admission=False,formal_gain=0)
    source=CircuitSource(state);eq,le=closed_uid_tables(source)
    fields['owners']=actual_words(new,1,2,eq,le)
    aux[:,5]=actual_words(new,2,3,eq,le)
    return source,p


def consumer(source,p):
    injected=inject_phase(source,p,pool=pool(),enabled=True)
    view=bind_phase(source,p,injected,pool=pool(),enabled=True);first=p['first_uid']
    overlay,_=build(source.owners,[(view.eq_c,first),(view.le_c,first+len(view.eq_rhs))],
        old_n_cont=source.old_n_cont,old_uid_ceiling=first,pool=pool(),enabled=True)
    plans,stats=discover_append(source,view,overlay,pool=pool(),enabled=True)
    expected,*_=factor_plans(source,view,first_uid=first,pool=pool())
    assert [asdict(v) for v in plans]==[asdict(v) for v in expected]
    return view,overlay,plans,stats


@pytest.mark.parametrize('value',[F(-1,4),F(1,8),F(3,8)])
def test_new_circuit_consumer_UID_and_nonzero_exact_inverse(value):
    source,p=fixture();before=digest(p)
    view,overlay,plans,stats=consumer(source,p)
    assert len(plans)==1 and stats['selected_old_consumers']==1
    assert resolve(source,3,pool=pool())==(False,2)
    new,_=splice_append(view,plans,pool=pool(),enabled=True)
    j=compile_journal(source,plans,pool=pool(),enabled=True)
    proof=verify(source,view,overlay,plans,new,j,pool=pool())
    assert proof['all_circuit_factors']==1 and proof['all_circuit_incidence_equal']
    assert len(j.circuit_tails)==1 and j.local.eq_row(2,pool=pool())==1
    original=[value,F(1,8)+value/4]
    point=extend(source.state,original,pool=pool())+[F(0),F(0)]
    point[1]=F(7,8) # eliminated slot is unconstrained in native state
    recovered=reconstruct(source,new,j,plans,point,pool=pool(),enabled=True)
    assert recovered['original_point']==original
    assert recovered['proof']['circuit_equations']==1
    assert not recovered['proof']['feasibility_or_concrete_witness_claim']
    assert digest(p)==before


@pytest.mark.parametrize('c,k,h',[(16,32,4),(16,32,5)])
def test_complete_fresh_dense_Conv_phase_image_and_full_consumer_population(c,k,h):
    expr=expression(c,k,h);keep=np.ones(expr.n_out,bool)
    old=old_lift(expr,keep,enabled=True)['fields']
    draft=lift(expr,keep,enabled=True);source=CircuitSource(draft['state']);p=packet(SimpleNamespace(**old))
    view,overlay,plans,_=consumer(source,p)
    eq,le=closed_uid_tables(source)
    assert np.array_equal(eq,draft['construction']['eq_uids'])
    assert np.array_equal(le,draft['construction']['ineq_uids'])
    for r in source.state['auxiliary_records']:
        assert resolve(source,int(r[1]),pool=pool())==(False,int(r[0]))
    assert source.hz.n_bin==old['hz'].n_bin==1
    if plans:
        new,_=splice_append(view,plans,pool=pool(),enabled=True)
        j=compile_journal(source,plans,pool=pool(),enabled=True)
        assert verify(source,view,overlay,plans,new,j,pool=pool())['all_circuit_incidence_equal']


@pytest.mark.parametrize('field',['frame_id','first_uid','source_n_cont','pre_c'])
def test_original_phase_source_binding_mismatch(field):
    source,p=fixture()
    if field=='pre_c':p[field]=p[field]+.125
    else:p[field]+=1
    with pytest.raises(ValueError,match='binding'):inject_phase(source,p,pool=pool(),enabled=True)


@pytest.mark.parametrize('field',['eq_c','le_c','Gc','eq_b','eq_rhs'])
def test_complete_phase_literal_or_coordinate_change_rejected(field):
    source,p=fixture();injected=inject_phase(source,p,pool=pool(),enabled=True)
    injected=copy.deepcopy(injected)
    if field in ('eq_c','le_c','Gc'):injected[field].indices[0]=source.state['old_source_n_cont']
    elif field=='eq_b':injected[field].data[0]*=2
    else:injected[field][0]+=.125
    with pytest.raises(ValueError):bind_phase(source,p,injected,pool=pool(),enabled=True)


def test_missing_circuit_tail_and_inverse_equation_are_rejected():
    source,p=fixture();view,overlay,plans,_=consumer(source,p)
    new,_=splice_append(view,plans,pool=pool(),enabled=True)
    j=compile_journal(source,plans,pool=pool(),enabled=True)
    bad=type(j)(j.local,j.state,np.empty(0,np.uint64))
    with pytest.raises(ValueError,match='circuit owners'):verify(source,view,overlay,plans,new,bad,pool=pool())
    point=[F(1,8),F(0),F(0),F(0),F(0)]
    with pytest.raises(ValueError,match='circuit equation'):reconstruct(source,new,j,plans,point,pool=pool(),enabled=True)


def test_default_off_and_budget_fail_closed():
    assert inject_phase(None,None,pool=None) is None
    assert bind_phase(None,None,None,pool=None) is None
    assert compile_journal(None,None,pool=None) is None
    assert reconstruct(None,None,None,None,None,pool=None) is None
    source,p=fixture()
    with pytest.raises(MemoryError):inject_phase(source,p,pool=WorkPool(0),enabled=True)


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('subtract',[False,True])
def test_full_writer_bill_matches_every_paid_operation(mixed,subtract):
    from experiments.neural_hz_20260831.test_c28_consumer_discovery_v1 import source
    from experiments.neural_hz_20260831.test_c30_first_write_v1 import view_from
    from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append as old_discover
    c,h,overlay,*_=source(mixed=mixed,subtract=subtract)
    view=view_from(c,h);plans,_=old_discover(c,view,overlay,pool=pool(),enabled=True)
    b=writer_bound(view,plans,pool=pool(),enabled=True)
    coupled=CoupledPool(0,0);writer=WriterPool(coupled,payload_cap=b['native_payload_upper'])
    splice_append(view,plans,pool=writer,enabled=True)
    assert coupled.used==b['incremental_upper']
    assert writer.native.used==b['native_payload_upper']
