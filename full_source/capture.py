"""Capture the registered experts' parameters and inspected topology, no forward."""
import base64
import sys
from unittest.mock import patch
from router_source.capture import sha


def capture(job, source, pair):
    import torch
    from act.back_end.moe.factory import load_output_moe_checkpoint
    from act.pipeline.moe.staged_verifier import _model_state_identity
    if sys.byteorder!='little':raise ValueError('explicit little endian required')
    subject=job['parent_request']['subject']; path=subject['checkpoint']
    if sha(path)!=subject['checkpoint_sha256']:raise ValueError('checkpoint identity')
    torch.set_num_threads(1)
    model,payload=load_output_moe_checkpoint(path,map_location='cpu');model=model.cpu().double().eval()
    if payload['format']!='act-output-conv-moe-v1' or _model_state_identity(model)!=source['request']['model_state']:
        raise ValueError('registered state changed')
    def encode(t):
        t=t.detach().cpu().contiguous()
        return {'dtype':str(t.dtype),'shape':list(t.shape),'byte_order':'little',
                'bytes':base64.b64encode(t.view(torch.uint8).numpy().tobytes()).decode()}
    def two(v):return [v,v] if type(v)is int else list(v)
    out=[];nn=torch.nn
    expected=[nn.Conv2d,nn.ReLU,nn.Conv2d,nn.ReLU,nn.AvgPool2d,nn.Flatten,nn.Linear,nn.ReLU,nn.Linear]
    with patch.object(nn.Module,'_call_impl',side_effect=AssertionError('forward prohibited')):
        for i in pair:
            net=model.experts[i]; layers=list(net.children()); rows=[]
            if type(net)is not nn.Sequential or [type(m) for m in layers]!=expected:raise ValueError('expert topology')
            for j,m in enumerate(layers):
                row={'kind':type(m).__name__,'index':j,'training':m.training}
                if type(m) in (nn.Conv2d,nn.Linear):
                    row.update(weight_name=f'experts.{i}.{j}.weight',bias_name=f'experts.{i}.{j}.bias',
                               weight=encode(m.weight),bias=encode(m.bias))
                if type(m)is nn.Conv2d:
                    row['graph']={'stride':two(m.stride),'padding':two(m.padding),'dilation':two(m.dilation),
                                  'groups':m.groups,'padding_mode':m.padding_mode,'training':m.training}
                if type(m)is nn.ReLU:row['inplace']=m.inplace
                if type(m)is nn.AvgPool2d:
                    row.update(kernel=two(m.kernel_size),stride=two(m.stride),padding=two(m.padding),
                               ceil_mode=m.ceil_mode,count_include_pad=m.count_include_pad,divisor_override=m.divisor_override)
                if type(m)is nn.Flatten:row['dimensions']=[m.start_dim,m.end_dim]
                rows.append(row)
            out.append({'expert':i,'layers':rows})
    return {'schema':'DECLARED_FULL_EXPERTS_V1','request':source['request'],'pair':pair,'experts':out,
        'graph_correspondence':'Captured topology inspected; declared real graph, not native floating execution proof.'}
