import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from act.back_end.solver.solver_hz import sparse_hz_linear,_lower_hz_milp,HZSolver
from act.front_end.specs import OutputSpec,OutKind
from experiments.neural_hz_20260831.c34_terminal_binding_v1 import suffix,verify
from experiments.neural_hz_20260831.c34_witness_reconstruction_v1 import recover
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute


def fixture(monkeypatch,ids=(78,79,80)):
    tf,state,post,_=execute(monkeypatch,layer_id=ids[0])
    weight=torch.tensor([[.25,-.5,.125,1.,-.25,.5],[1.,.25,.5,-.125,0.,.25]],dtype=torch.float64)
    bias=torch.tensor([.125,-.25],dtype=torch.float64)
    spec=OutputSpec(OutKind.UNSAFE_LINEAR,c=torch.tensor([[1.,-1.]],dtype=torch.float64),d=torch.tensor([-.125],dtype=torch.float64))
    params=spec.encode_linear(B=1,n_out=2,device=torch.device('cpu'),dtype=torch.float64)
    dense=SimpleNamespace(id=ids[1],kind='DENSE',params=dict(weight=weight,bias=bias))
    assertion=SimpleNamespace(id=ids[2],kind='ASSERT',params=params)
    tf._net=SimpleNamespace(layers=[state['layer'],dense,assertion],preds={ids[1]:[ids[0]],ids[2]:[ids[1]]},
        succs={ids[0]:[ids[1]],ids[1]:[ids[2]],ids[2]:[]})
    final=sparse_hz_linear(post,sp.csr_matrix(weight.numpy()),bias.numpy());tf._sparse_hz_cache[ids[1]]=final
    input_hz=state['expression'].terms[0].source
    kwargs=dict(batch_size=1,n_out=2,input_shape=(1,input_hz.n_out),timelimit=45.)
    _,_,signature=suffix(tf._net,ids[0],pool=WorkPool(256_000_000))
    proof=dict(schema='c34_independent_final_affine_transfer_v1',completed=True,whole_actual_spliced_source_bound=True,
        all_final_output_and_predicate_bits_equal=True,unchanged_original_input_and_property_bound=True,
        suffix_signature=signature,post_HZ_sha256=source_digest(post),final_HZ_sha256=source_digest(final),
        underlying_splice_transfer_sha256=state['lifted'].transfer_proof_sha256,
        input_HZ_sha256=source_digest(input_hz),input_shape=list(kwargs['input_shape']))
    raw=json.dumps(proof,sort_keys=True).encode()
    return state,final,input_hz,spec,kwargs,raw,hashlib.sha256(raw).hexdigest()


def test_default_off_does_not_inspect_any_input():
    assert verify(*([object()]*7),pool=object()) is None
    assert recover(*([object()]*7),pool=object()) is None


@pytest.mark.parametrize('ids',[(78,79,80),(3,19,80),(812,932,1210)])
def test_final_native_predicate_input_property_binding_without_id_menu(monkeypatch,ids):
    args=fixture(monkeypatch,ids)
    report=verify(*args,pool=WorkPool(256_000_000),enabled=True)
    assert report['actual_predicates_shared_by_identity'] and not report['base_feasibility_shortcut']


@pytest.mark.parametrize('bad',['final','input','property','polarity','weight','bias','cache','predicate_copy','RHS_copy','topology','shape','budget','proof','workcap'])
def test_final_binding_rejects_wrong_problem_or_source(monkeypatch,bad):
    state,final,inp,spec,kw,raw,sha=fixture(monkeypatch);pool=WorkPool(256_000_000)
    if bad=='final':final.c[0]+=.125
    elif bad=='input':inp.c[0]+=.125
    elif bad=='property':spec.d+=.125
    elif bad=='polarity':spec.kind=OutKind.LINEAR_LE
    elif bad=='weight':state['tf']._net.layers[1].params['weight'][0,0]+=.125
    elif bad=='bias':state['tf']._net.layers[1].params['bias'][0]+=.125
    elif bad=='cache':state['tf']._sparse_hz_cache.pop(79)
    elif bad=='predicate_copy':final.Ac=final.Ac.copy()
    elif bad=='RHS_copy':final.b=final.b.copy()
    elif bad=='topology':state['tf']._net.preds[79]=[77]
    elif bad=='shape':kw['input_shape']=(inp.n_out,1)
    elif bad=='budget':kw['timelimit']=46.
    elif bad=='proof':raw+=b' '
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):
        verify(state,final,inp,spec,kw,raw,sha,pool=pool,enabled=True)


@pytest.mark.parametrize('coordinate',[-.125,0.,.125])
def test_exact_full_extension_and_unchanged_original_input_recovery(monkeypatch,coordinate):
    state,final,inp,spec,kw,raw,sha=fixture(monkeypatch)
    model=_lower_hz_milp(final,project_inactive_cont=False,fix_implied_binary=False)
    x=np.zeros(model.n_var,np.float64)
    x[np.flatnonzero(model.cont_source==1)[0]]=coordinate
    got,proof=recover(state['lifted'],model,x,inp,kw['input_shape'],0,HZSolver._recover_input,
        pool=WorkPool(256_000_000),enabled=True)
    assert torch.equal(got,HZSolver._recover_input(model,x,inp,kw['input_shape'],0))
    assert proof['all_unit_pairs_reconstructed']>0 and proof['legacy_aliases_reconstructed']>0
    assert proof['original_input_latents_unchanged'] and proof['concrete_network_validation_still_required']


@pytest.mark.parametrize('bad',['box','nonfinite','coords','frame','shape','native','cap'])
def test_witness_guard_failure_returns_no_promotable_witness(monkeypatch,bad):
    state,final,inp,spec,kw,raw,sha=fixture(monkeypatch)
    model=_lower_hz_milp(final,project_inactive_cont=False,fix_implied_binary=False)
    x=np.zeros(model.n_var,np.float64);pool=WorkPool(256_000_000);native=HZSolver._recover_input
    if bad=='box':x[0]=2.
    elif bad=='nonfinite':x[0]=np.nan
    elif bad=='coords':model.cont_source[0]=-1
    elif bad=='frame':inp.frame_id+=1
    elif bad=='shape':kw['input_shape']=(2,inp.n_out//2)
    elif bad=='native':native=lambda *a:torch.zeros(inp.n_out,dtype=torch.float64)
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):
        recover(state['lifted'],model,x,inp,kw['input_shape'],0,native,pool=pool,enabled=True)
