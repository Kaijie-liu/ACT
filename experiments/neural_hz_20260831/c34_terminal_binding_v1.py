"""Default-off complete final affine/input/property binding of actual C32 HZ."""
import hashlib
import json
import time
import numpy as np
import torch

from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState


def array(value):
    return value.detach().cpu().double().numpy() if isinstance(value,torch.Tensor) else np.asarray(value,np.float64)


def property_signature(encoded):
    kind=str(encoded['kind']);m=int(encoded['M'])
    c,t=array(encoded['C']),array(encoded['thresholds'])
    if c.ndim!=2 or t.shape!=(1,m) or c.shape[0]!=m or not np.isfinite(c).all() or not np.isfinite(t).all():
        raise ValueError('registered single-lane finite property encoding required')
    return dict(kind=kind,M=m,n_out=c.shape[1],coefficient_threshold_sha256=digest_arrays(c,t))


def suffix(net,producer,*,pool):
    pool.charge('terminal_suffix_topology_metadata',256+64*len(net.layers))
    layers={layer.id:layer for layer in net.layers}
    following=net.succs.get(producer,[])
    if len(following)!=1:raise ValueError('selected source lacks one affine suffix')
    dense=layers[following[0]];following=net.succs.get(dense.id,[])
    if dense.kind!='DENSE' or net.preds.get(dense.id)!=[producer] or len(following)!=1:
        raise ValueError('selected suffix is not a single original dense consumer')
    assertion=layers[following[0]]
    if assertion.kind!='ASSERT' or net.preds.get(assertion.id)!=[dense.id] or net.succs.get(assertion.id):
        raise ValueError('dense output is not the original terminal assertion input')
    weight=array(dense.params['weight']);bias=dense.params.get('bias')
    bias=np.zeros(weight.shape[0],np.float64) if bias is None else array(bias).reshape(-1)
    if weight.ndim!=2 or bias.shape!=(weight.shape[0],) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('invalid original dense operator/bias')
    signature=dict(weight_shape=list(weight.shape),weight_bias_sha256=digest_arrays(weight,bias),
        assertion=property_signature(assertion.params))
    return dense,assertion,signature


def load(raw,expected_sha):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('independent final affine proof missing/changed')
    proof=json.loads(raw)
    if (proof.get('schema')!='c34_independent_final_affine_transfer_v1' or proof.get('completed') is not True
            or proof.get('whole_actual_spliced_source_bound') is not True
            or proof.get('all_final_output_and_predicate_bits_equal') is not True
            or proof.get('unchanged_original_input_and_property_bound') is not True):
        raise ValueError('incomplete final source/output/box/reconstruction proof')
    return proof


def verify(state,output,input_hz,out_spec,kwargs,raw,expected_sha,*,pool,enabled=False):
    if not enabled:return None
    started=time.monotonic();proof=load(raw,expected_sha);new=state['lifted'];tf=state['tf']
    pool.charge('terminal_actual_source_frame_and_publication',1024)
    if type(new) is not SplicedState:raise ValueError('terminal lacks the NEW reconstructable splice state')
    new.validate()
    dense,assertion,signature=suffix(tf._net,state['layer'].id,pool=pool)
    if (signature!=proof['suffix_signature'] or kwargs.get('batch_size')!=1
            or kwargs.get('n_out')!=output.n_out or list(kwargs.get('input_shape',()))!=proof['input_shape']
            or kwargs.get('timelimit')!=45. or tf._sparse_hz_cache.get(dense.id) is not output
            or dense.id in tf._sparse_affine_expr_cache or state['native_block_calls']!=1):
        raise ValueError('terminal source/cache/shape/property/budget changed')
    if (new.transfer_proof_sha256!=proof['underlying_splice_transfer_sha256']
            or source_digest(new.hz)!=proof['post_HZ_sha256'] or source_digest(output)!=proof['final_HZ_sha256']
            or source_digest(input_hz)!=proof['input_HZ_sha256']
            or output.frame_id!=new.hz.frame_id or output.frame_id!=input_hz.frame_id
            or (output.n_cont,output.n_bin)!=(new.hz.n_cont,new.hz.n_bin)
            or input_hz.n_cont>new.lineage.old_n_cont
            or input_hz.n_bin>new.original_fields['old_n_bin']):
        raise ValueError('complete final HZ/input or shared latent identity differs')
    for name in ('Ac','Ab','Auc','Aub'):
        if getattr(output,name) is not getattr(new.hz,name):raise ValueError('final affine copied/replaced actual predicate graph')
    for name in ('b','ub'):
        a,b=getattr(output,name),getattr(new.hz,name)
        if a.shape!=b.shape or a.strides!=b.strides or a.__array_interface__['data'][0]!=b.__array_interface__['data'][0]:
            raise ValueError('final affine replaced actual source RHS storage')
    encoded=out_spec.encode_linear(B=1,n_out=output.n_out,device=torch.device('cpu'),dtype=torch.float64)
    if property_signature(encoded)!=signature['assertion']:
        raise ValueError('solver output property polarity/rows differ from original assertion')
    return dict(complete_final_HZ_sha256=proof['final_HZ_sha256'],complete_input_HZ_sha256=proof['input_HZ_sha256'],
        original_suffix_signature=signature,actual_predicates_shared_by_identity=True,
        reconstructable_state_preserved=True,source_hash_authentication_elapsed_s=time.monotonic()-started,
        hash_byte_scans_separate_from_generator_work=True,base_feasibility_shortcut=False,formal_gain=0)
