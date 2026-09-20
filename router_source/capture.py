"""Capture exact parameter/input bytes; no forward, HZ propagation or solver."""
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(job, statement):
    import torch
    from act.back_end.moe.factory import load_output_moe_checkpoint
    from act.pipeline.moe.staged_verifier import _tensor_identity,_model_state_identity
    if sys.byteorder!='little':raise ValueError('capture requires explicit little-endian platform')
    torch.set_num_threads(1)
    parent=job['parent_request'];subject=parent['subject']
    if sha(subject['checkpoint'])!=subject['checkpoint_sha256'] or sha(parent['tensors']['path'])!=parent['tensors']['sha256']:
        raise ValueError('frozen checkpoint/input file drift')
    model,payload=load_output_moe_checkpoint(subject['checkpoint'],map_location='cpu')
    model=model.cpu().double().eval()
    if payload['format']!='act-output-conv-moe-v1':raise ValueError('unsupported checkpoint graph family')
    if _model_state_identity(model)!=statement['request']['model_state']:
        raise ValueError('materialized state differs from old proof')
    modules=list(model.router.children());nn=torch.nn
    if (type(model.router)is not nn.Sequential or len(modules)!=3 or
            [type(m) for m in modules]!=[nn.AvgPool2d,nn.Flatten,nn.Linear]):
        raise ValueError('not the registered affine router')
    pool,flatten,linear=modules
    def scalar(v):
        if type(v)is int:return v
        if type(v)is tuple and len(v)==2 and v[0]==v[1]:return v[0]
        raise ValueError('non-square pooling')
    graph={'schema':'REAL_NONOVERLAP_AVGPOOL_FLATTEN_LINEAR_V1',
        'pool':scalar(pool.kernel_size),'pool_stride':scalar(pool.stride),
        'pool_padding':scalar(pool.padding),'ceil_mode':pool.ceil_mode,
        'count_include_pad':pool.count_include_pad,'divisor_override':pool.divisor_override,
        'flatten':[flatten.start_dim,flatten.end_dim],'training':model.router.training,
        'weight':'router.2.weight','bias':'router.2.bias'}
    def encode(t):
        t=t.detach().cpu().contiguous()
        return {'dtype':str(t.dtype),'shape':list(t.shape),'byte_order':'little',
                'bytes':base64.b64encode(t.view(torch.uint8).numpy().tobytes()).decode()}
    state=model.state_dict()
    images=torch.load(parent['tensors']['path'],map_location='cpu',weights_only=True)
    for name in ('center','lower','upper'):
        if _tensor_identity(images[name])!=statement['request'][name]:raise ValueError('input tensor identity')
    result={'schema':'AFFINE_ROUTER_SOURCE_V1','request':statement['request'],'graph':graph,
        'state_inventory':[{'name':name,**_tensor_identity(t)} for name,t in sorted(state.items())],
        'parameters':{name:encode(state[name]) for name in ('router.2.weight','router.2.bias')},
        'input':{name:encode(images[name]) for name in ('center','lower','upper')},
        'origin':{'checkpoint_sha256':subject['checkpoint_sha256'],
            'factory_config':payload['factory_config'],
            'factory_source_sha256':sha(ROOT/'act/back_end/moe/conv_factory.py'),
            'checkpoint_deserialization_and_graph_correspondence':'captured/structurally inspected, not independently verified native program semantics'}}
    return result
