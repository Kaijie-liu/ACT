"""Ordinary circuit custody, actual phase image, final property and input recovery."""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import pytest
import torch
from act.back_end.solver import solver_hz as backend
from act.back_end.hybridz_tf.tf_cnn import SparseHZAffineExpr,SparseHZAffineTerm
from act.front_end.specs import OutputSpec,OutKind
from experiments.neural_hz_20260831.test_c99_circuit_native_v1 import fixture as toy
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c91_physical_archive_v1 import bind
from experiments.neural_hz_20260831.c99_circuit_consumer_v2 import CircuitSource,inject_phase,bind_phase
from experiments.neural_hz_20260831.c99_append_discovery_v2 import discover_append
from experiments.neural_hz_20260831.c99_circuit_journal_v2 import compile_journal,reconstruct
from experiments.neural_hz_20260831.c99_native_proof_v2 import verify as verify_native
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c70_native_proof_v1 import digest
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c100_native_binding_v1 import admit_source,admit_native,phase_image,journal_image
from experiments.neural_hz_20260831.c100_native_terminal_v1 import affine,recover
from experiments.neural_hz_20260831.c100_runtime_final_binding_v1 import verify,final_extra
from experiments.neural_hz_20260831.c100_terminal_preflight_v1 import dimensions
from experiments.neural_hz_20260831.c100_live_runtime_v1 import installed as runtime,extra_bound
from experiments.neural_hz_20260831.c100_terminal_observer_v1 import installed


def pool():return WorkPool(256_000_000)


def native():
    old,p=toy();state=old.state;f=state['fields'];h=f['hz']
    inp=backend.SparseHZono(np.zeros(1),sp.csr_matrix([[1.]]),sp.csr_matrix((1,1)),
        sp.csr_matrix([[1.]]),sp.csr_matrix([[-.25]]),np.zeros(1),
        sp.csr_matrix((0,1)),sp.csr_matrix((0,1)),np.zeros(0),frame_id=h.frame_id,exact=True)
    f.update(old_n_bin=1,keep=np.ones(1,bool),radix_gauges=np.empty(0,np.int64),
        expression=SparseHZAffineExpr((SparseHZAffineTerm(inp,(sp.eye(1,format='csr'),)),),
            np.zeros(1),1,h.frame_id))
    # Authentication-interface fixture for x1=.125+.25*x0 and x2=x1.
    # The real source theorem is independently established by inherited C98/C99
    # tests and full target proof, never by this fixture's proof flags.
    _,raw=bind(state,dict(all_original_maps_and_other_predicates_preserved=True,
        independent_complete_owner_delta_proved=True,every_fresh_native_literal_matches_original_theorem=True))
    s,_=admit_source(state,raw,expected_sha256=hashlib.sha256(raw).hexdigest(),enabled=True)
    c=s.consumer;injected=inject_phase(c,p,pool=pool(),enabled=True)
    view=bind_phase(c,p,injected,pool=pool(),enabled=True);first=p['first_uid']
    overlay,_=build(c.owners,[(view.eq_c,first),(view.le_c,first+len(view.eq_rhs))],
        old_n_cont=c.old_n_cont,old_uid_ceiling=first,pool=pool(),enabled=True)
    plans,_=discover_append(c,view,overlay,pool=pool(),enabled=True)
    hz,_=splice_append(view,plans,pool=pool(),enabled=True)
    journal=compile_journal(c,plans,pool=pool(),enabled=True)
    assert verify_native(c,view,overlay,plans,hz,journal,pool=pool())['all_circuit_incidence_equal']
    q=[F(0),F(0),F(1,8),F(0),F(0)]
    inv=reconstruct(c,hz,journal,plans,q,pool=pool(),enabled=True)
    assert inv['proof']['all_equations_exact']
    transfer=dict(schema='c100_complete_circuit_native_transfer_v1',complete_C99_archive_authenticated=True,
        complete_independent_inverse_restored=True,source_identity=s.validate()['identity'],
        source_proof_sha256=s.expected_proof_sha256,new_HZ_sha256=source_digest(hz),
        journal_identity=digest(journal_image(journal)),packet_identity=digest(injected),
        event_sha256=hashlib.sha256(overlay.events.tobytes()).hexdigest(),
        packet_header={k:injected[k] for k in ('schema','offline_only','fresh_native_execution','provenance',
            'coordinate_injection','original_packet_schema')},
        original_source_proof_utf8='fixture',original_circuit_proof_utf8='fixture',
        complete_component_proof=dict(plans=[asdict(p) for p in plans]),full_LIVE_admission=False,formal_gain=0)
    raw=json.dumps(transfer,sort_keys=True).encode()
    got,_=admit_native(enabled=True,source=s,hz=hz,lineage=journal,events=overlay.events,
        actual_phase_image=phase_image(s,view,json.loads(raw)),transfer_proof_bytes=raw,
        expected_transfer_sha256=hashlib.sha256(raw).hexdigest(),construction_report=dict(new_phase_binaries=1))
    return got,inp


def fixture(ids=(78,79,80)):
    n,inp=native();w=np.array([[.75],[-.125],[.375]]);bias=np.array([.125,-.25,.5])
    spec=OutputSpec(OutKind.UNSAFE_LINEAR,c=torch.tensor([[1.,-1.,0.]],dtype=torch.float64),
        d=torch.tensor([-.125],dtype=torch.float64))
    post=SimpleNamespace(id=ids[0],kind='RELU')
    dense=SimpleNamespace(id=ids[1],kind='DENSE',params=dict(weight=w,bias=bias))
    assertion=SimpleNamespace(id=ids[2],kind='ASSERT',params=spec.encode_linear(B=1,n_out=3,
        device=torch.device('cpu'),dtype=torch.float64))
    net=SimpleNamespace(layers=[post,dense,assertion],preds={ids[1]:[ids[0]],ids[2]:[ids[1]]},
        succs={ids[0]:[ids[1]],ids[1]:[ids[2]],ids[2]:[]})
    out=backend.sparse_hz_linear(n.hz,sp.csr_matrix(w),bias);shape=(1,1)
    proof=affine(n,net,ids[0],out,inp,shape,spec,pool=pool(),enabled=True)
    raw=json.dumps(proof,sort_keys=True).encode()
    tf=SimpleNamespace(_net=net,_sparse_hz_cache={ids[0]:n.hz,ids[1]:out},_sparse_affine_expr_cache={})
    state=dict(lifted=n,tf=tf,layer=post,native_block_calls=1)
    kw=dict(input_shape=shape,batch_size=1,n_out=out.n_out,timelimit=45.)
    return state,out,inp,spec,kw,raw,hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('ids',[(78,79,80),(4,13,19),(182,512,1055)])
def test_full_circuit_native_input_property_and_cost_without_id_trigger(ids):
    args=fixture(ids);p=pool();report=verify(*args,pool=p,enabled=True)
    assert report['actual_predicates_shared_by_identity']
    assert p.used+512==final_extra(3)
    n=args[0]['lifted'];assert type(n.source.consumer) is CircuitSource
    assert n.numeric_roots()['complete_circuit_source'] is n.source.circuit_state
    assert n.lineage.local.eq_roots is n.source.eq_roots


@pytest.mark.parametrize('bad',['final','input','property','circuit','phase','proof','budget'])
def test_original_problem_or_complete_circuit_change_rejected(bad):
    state,out,inp,spec,kw,raw,sha=fixture();p=pool()
    if bad=='final':out.c[0]+=.125
    elif bad=='input':inp.c[0]+=.125
    elif bad=='property':spec.kind=OutKind.LINEAR_LE
    elif bad=='circuit':state['lifted'].source.circuit_state['auxiliary_records'][0,4]+=1
    elif bad=='phase':state['lifted'].actual_phase_image['eq_rhs'][0]+=.125
    elif bad=='proof':raw+=b' '
    else:p=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):verify(state,out,inp,spec,kw,raw,sha,pool=p,enabled=True)


@pytest.mark.parametrize('coordinate',[-.25,.125,.375])
def test_nonzero_unit_circuit_inverse_and_original_input(coordinate):
    state,out,inp,spec,kw,*_=fixture();n=state['lifted'];m=backend._lower_hz_milp(out)
    x=np.zeros(m.n_var)
    x[np.flatnonzero(m.cont_source==0)[0]]=coordinate
    x[np.flatnonzero(m.cont_source==2)[0]]=.125+.25*coordinate
    point,report=recover(n,m,x,inp,kw['input_shape'],0,backend.HZSolver._recover_input,pool=pool(),enabled=True)
    assert point.item()==coordinate and report['inverse']['circuit_equations']==1
    assert report['inverse']['unit_equations']==1 and report['inverse']['all_equations_exact']


@pytest.mark.parametrize('status',[1,2])
def test_base_failure_does_not_bypass_or_launch_rescue(monkeypatch,status):
    state,out,inp,spec,kw,*_=fixture();calls=[];events=[]
    def solve(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status=status,message='fixture',success=False,x=None,mip_node_count=0)
    monkeypatch.setattr(backend,'milp',solve)
    with installed(state['lifted'],out,inp,kw['input_shape'],dimensions(out,pool=pool()),
        enabled=True,emit=events.append,on_point=lambda *a:pytest.fail('base failure created point')):
        result=backend.HZSolver().evaluate_spec(out,spec,input_hz=inp,**kw)
    assert len(calls)==1 and np.count_nonzero(calls[0]['c'])==0
    assert result[0].status.name=='UNKNOWN'


def test_stored_column_dimensions_match_ordinary_native_including_explicit_zero():
    _,out,*_=fixture()
    for _ in range(2):
        d=dimensions(out,pool=pool());m=backend._lower_hz_milp(out)
        assert (d['native_lowered_n_cont'],d['native_lowered_n_bin'])==(m.n_cont,m.n_bin)
        out.Gc.data[0]=0.


def test_default_off_and_upfront_runtime_bound():
    assert admit_source(None,None,expected_sha256=None) is None and admit_native() is None
    assert affine(*([None]*7),pool=None) is None and recover(*([None]*7),pool=None) is None
    assert verify(*([None]*7),pool=None) is None
    with runtime():pass
    n,_=native()
    with runtime(enabled=True,source_bytes=n.source.proof_bytes,source_sha=n.source.expected_proof_sha256,
        transfer_bytes=n.transfer_proof_bytes,transfer_sha=n.expected_transfer_sha256):pass
    assert extra_bound(1,1,1,4,3)['source_binding_setup']==512
    assert final_extra(81)==6976
