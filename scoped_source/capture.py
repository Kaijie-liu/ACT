"""Generic finite source extraction. No forward, propagation or solver call."""
import base64
import hashlib
from fractions import Fraction as F
import sys
from router_source.checker import tensor
from scoped_source.graph import clock


def capture(model, center, *, label, radius, margin, clip, deadline):
    import torch
    from act.back_end.moe.model import OutputLevelMoE
    from act.back_end.moe.schema import GateKind
    tick=clock(deadline);nn=torch.nn
    if (type(model)is not OutputLevelMoE or model.training or model.shared_expert is not None or
        model.spec.top_k!=2 or model.spec.gate!=GateKind.SELECTED_SOFTMAX or not model.spec.normalized or
        sys.byteorder!='little'):raise ValueError('registered output top2 eval model required')
    if any(m._forward_hooks or m._forward_pre_hooks or 'forward' in vars(m) for m in model.modules()):
        raise ValueError('hooked/instance-overridden execution is not the declared graph')
    def encode(t):
        tick()
        if t.dtype!=torch.float64 or t.device.type!='cpu' or not bool(torch.isfinite(t).all()):
            raise ValueError('finite CPU float64 stored coefficients required')
        t=t.detach().contiguous()
        return {'dtype':'torch.float64','shape':list(t.shape),'byte_order':'little',
            'bytes':base64.b64encode(t.view(torch.uint8).numpy().tobytes()).decode()}
    encoded={n:encode(t) for n,t in sorted(model.state_dict().items())}
    inv=[{'name':n,**tensor(t)[0]} for n,t in encoded.items()];h=hashlib.sha256()
    for v in inv:h.update(v['name'].encode());h.update(v['sha256'].encode())
    networks=[];classes=None
    for name,prefix,net in [('router','router',model.router)]+[
        (f'expert{i}',f'experts.{i}',m) for i,m in enumerate(model.experts)]:
        if type(net)is not nn.Sequential or net.training:raise ValueError('sequential eval source required')
        layers=[]
        for j,m in enumerate(net):
            tick()
            if type(m) not in (nn.Linear,nn.ReLU,nn.Flatten) or m.training:raise ValueError('unsupported captured operator/mode')
            row={'kind':type(m).__name__,'index':j,'training':False}
            if type(m)is nn.Linear:
                if m.bias is None:raise ValueError('explicit affine bias required')
                for role in ('weight','bias'):
                    n=f'{prefix}.{j}.{role}';row[role+'_name']=n;row[role]=encoded[n]
            elif type(m)is nn.ReLU:row['inplace']=m.inplace
            else:row['dimensions']=[m.start_dim,m.end_dim]
            layers.append(row)
        if not layers or type(net[-1])is not nn.Linear:raise ValueError('final affine endpoint required')
        if name!='router':
            if classes is None:classes=net[-1].out_features
            if classes!=net[-1].out_features:raise ValueError('shared output classes required')
        networks.append({'name':name,'layers':layers})
    center_doc=encode(center)
    result={'schema':'SCOPED_DECLARED_TOP2_V1','request':{'experts':len(model.experts),'classes':classes,
        'label':label,'top_k':2,'gate':'SELECTED_SOFTMAX','tie_policy':'ANY_LEGAL_TOPK','training':False,
        'center':tensor(center_doc)[0],'radius':str(F(radius)),'margin':str(F(margin)),
        'clip':[str(F(x)) for x in clip],
        'model_state':{'sha256':h.hexdigest(),'tensor_count':len(inv),
            'parameter_count':sum(t.numel() for t in model.state_dict().values())}},
        'center':center_doc,'state_inventory':inv,'networks':networks,
        'trust':'declared real graph; graph/program correspondence and native floating execution are separate'}
    tick();return result
