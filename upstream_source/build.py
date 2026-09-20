"""Capture only input HZ and first expert Conv2d lowering. No full propagation."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

from router_source.capture import ROOT,sha
from router_source.build import JOB,JOB_HASH,save

OLD=ROOT/'data/moe/results/router_source_conv98_20260920_v1/relocated'
SOURCE=ROOT/'data/moe/results/conv_pre_f0_rational_20260915_r2/rank24_monolithic'


def capture(job,source,pair):
    import torch
    from act.back_end.core import Bounds
    from act.back_end.moe.factory import load_output_moe_checkpoint
    from act.back_end.solver.solver_hz import sparse_hz_from_bounds,sparse_hz_linear
    from act.back_end.solver.hz_lp_export import snapshot
    from act.back_end.hybridz_tf.tf_cnn import sparse_conv2d_matrix_from_layer
    from act.pipeline.moe.staged_verifier import _model_state_identity,_tensor_identity
    from router_source.checker import inputs
    inputs(source,source['request'])
    torch.set_num_threads(1);parent=job['parent_request'];subject=parent['subject']
    if sha(subject['checkpoint'])!=subject['checkpoint_sha256'] or sha(parent['tensors']['path'])!=parent['tensors']['sha256']:
        raise ValueError('registered model/input files changed')
    model,payload=load_output_moe_checkpoint(subject['checkpoint'],map_location='cpu')
    model=model.cpu().double().eval()
    if payload['format']!='act-output-conv-moe-v1' or _model_state_identity(model)!=source['request']['model_state']:
        raise ValueError('materialized expert state differs')
    tensors=torch.load(parent['tensors']['path'],map_location='cpu',weights_only=True)
    for k in ('center','lower','upper'):
        if _tensor_identity(tensors[k])!=source['request'][k]:raise ValueError('materialized box differs')
    def encode(t):
        t=t.detach().cpu().contiguous()
        return {'dtype':str(t.dtype),'shape':list(t.shape),'byte_order':'little',
                'bytes':base64.b64encode(t.view(torch.uint8).numpy().tobytes()).decode()}
    # Fail if any module forward is accidentally introduced into this capture.
    with patch.object(torch.nn.Module,'_call_impl',side_effect=AssertionError('forward forbidden')):
        entry=sparse_hz_from_bounds(Bounds(tensors['lower'],tensors['upper']),frame_id=1)
        experts=[]
        expected=[torch.nn.Conv2d,torch.nn.ReLU,torch.nn.Conv2d,torch.nn.ReLU,
                  torch.nn.AvgPool2d,torch.nn.Flatten,torch.nn.Linear,torch.nn.ReLU,torch.nn.Linear]
        for i in pair:
            net=model.experts[i];layers=list(net.children())
            if type(net)is not torch.nn.Sequential or [type(m) for m in layers]!=expected:
                raise ValueError('unregistered expert topology')
            layer=layers[0]
            graph={'stride':list(layer.stride),'padding':list(layer.padding),
                'dilation':list(layer.dilation),'groups':layer.groups,
                'padding_mode':layer.padding_mode,'training':layer.training}
            params={'input_shape':list(tensors['lower'].shape),'weight':layer.weight,'bias':layer.bias,
                    **{k:graph[k] for k in ('stride','padding','dilation','groups')}}
            operator,bias=sparse_conv2d_matrix_from_layer(SimpleNamespace(params=params))
            out=sparse_hz_linear(entry,operator,bias)
            experts.append({'expert':i,'layer_index':0,'graph':graph,
                'topology_inspected':[type(m).__name__ for m in layers],
                'weight_name':f'experts.{i}.0.weight','bias_name':f'experts.{i}.0.bias',
                'weight':encode(layer.weight),'bias':encode(layer.bias),'output':snapshot(out)})
    return snapshot(entry),experts


def build(root):
    start=time.monotonic();freeze=json.loads((ROOT/'docs/upstream_source_v1_freeze.json').read_bytes())
    for path,h in freeze['artifacts'].items():
        if sha(ROOT/path)!=h:raise ValueError('frozen artifact changed: '+path)
    if sha(JOB)!=JOB_HASH:raise ValueError('old job changed')
    source=json.loads((OLD/'router_source.json').read_bytes())
    pair=freeze['pair'];t=time.monotonic()
    entry,experts=capture(json.loads(JOB.read_bytes()),source,pair)
    capture_seconds=time.monotonic()-t;destination=root/'relocated';destination.mkdir()
    t=time.monotonic()
    for name in ('router_source.json','router_proof.json'):
        shutil.copyfile(OLD/name,destination/name)
    for name in ('joint_hz.json','router_hz.json'):
        shutil.copyfile(SOURCE/name,destination/name)
    save(destination/'input_hz.json',entry)
    bindings=[]
    for expert in experts:
        output=expert.pop('output');name=f"expert{expert['expert']}_conv0.json"
        save(destination/name,output);expert['output_file']=name;bindings.append(expert)
    save(destination/'experts.json',bindings)
    for src,dst in [('router_source/checker.py','router_check.py'),
                    ('upstream_source/checker.py','local_check.py'),('upstream_source/verify.py','verify_upstream.py')]:
        shutil.copyfile(ROOT/src,destination/dst)
    manifest={'schema':'UPSTREAM_LOCAL_AUDIT_V1','request':source['request'],'pair':pair,
        'files':{p.name:sha(p) for p in sorted(destination.iterdir())},
        'origin':'New local input/first-Conv lowering control; old terminal expert HZ unchanged. Not a recreated historical layer trace.',
        'historical_source_sha256':{n:sha(SOURCE/n) for n in ('joint_hz.json','router_hz.json')}}
    save(destination/'manifest.json',manifest)
    save(root/'generation.json',{'manifest_sha256':sha(destination/'manifest.json'),
        'capture_seconds':capture_seconds,'serialization_seconds':time.monotonic()-t,
        'whole_seconds_before_publication':time.monotonic()-start,
        'bundle_bytes':sum(p.stat().st_size for p in destination.iterdir()),
        'native_solver_calls':0,'network_forward_calls':0,'new_full_network_propagations':0,
        'new_input_constructions':1,'new_local_conv_transfers':len(experts),
        'scope':'Source-audit cost, not end-to-end verification. Old output propagation/proposals excluded.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    build(p.parse_args().root.resolve())
