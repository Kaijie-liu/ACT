"""Tiny deterministic converter controls. No checkpoint/data/native solver.

Documents a current defect, does not change the production converter. A local
counterfactual rewires only paired BN BIAS to its SCALE in the toy interpreter.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from act.back_end.moe.factory import OutputMoEFactoryConfig,build_output_moe,build_act_moe_program
from act.front_end.specs import OutputSpec,OutKind
from act.util.device_manager import initialize_device
from metamoe_assignment_layers import concrete_layer
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def observe_bn(scale=2.,bias=1.,spatial=2,insert_conv=True):
    initialize_device('cpu','float64');torch.manual_seed(123)
    model=build_output_moe(OutputMoEFactoryConfig((1,spatial,spatial),2,num_experts=2)).double().eval()
    conv=nn.Conv2d(1,1,1,bias=False,dtype=torch.float64)
    bn=nn.BatchNorm2d(1,eps=0.,dtype=torch.float64)
    fc=nn.Linear(spatial*spatial,2,bias=False,dtype=torch.float64)
    with torch.no_grad():
        conv.weight.fill_(1.);bn.weight.fill_(scale);bn.bias.fill_(bias)
        bn.running_mean.zero_();bn.running_var.fill_(1.);fc.weight.fill_(1.)
    expert=nn.Sequential(*([conv] if insert_conv else []),bn,nn.Flatten(),fc).eval()
    model.experts[0]=expert
    x=torch.full((1,1,spatial,spatial),2.,dtype=torch.float64)
    spec=OutputSpec(kind=OutKind.LINEAR_LE,c=torch.zeros(1,2),d=torch.zeros(1))
    net=build_act_moe_program(model,center=x,lower=x,upper=x,output_spec=spec).experts[0]
    values,corrected={},{};edges=[]
    for layer in net.layers:
        if layer.kind=='ASSERT':continue
        pred=net.preds[layer.id];p=pred[0] if pred else None
        if len(pred)>1:raise ValueError('unexpected toy branch')
        actual=concrete_layer(layer,values.get(p),x)
        cp=p
        if layer.kind=='BIAS' and layer.params.get('paired_with_scale'):
            previous=net.layers[layer.id-1]
            if previous.kind!='SCALE' or list(previous.out_vars)!=list(layer.in_vars):
                raise ValueError('not an unambiguous BN pair')
            cp=previous.id
            edges.append({'bias':layer.id,'recorded_predecessors':pred,'declared_input_producer':cp,
                          'scale_bypassed':p!=cp})
        corrected[layer.id]=concrete_layer(layer,corrected.get(cp),x)
        values[layer.id]=actual
    last=net.layers[-2].id
    with torch.no_grad():source=expert(x).reshape(-1)
    return {'scale':scale,'bias':bias,'spatial':spatial,'insert_conv':insert_conv,'edges':edges,
            'source':source.tolist(),'converted':values[last].tolist(),'toy_rewired':corrected[last].tolist(),
            'source_vs_converted':float((source-values[last]).abs().max()),
            'source_vs_toy_rewired':float((source-corrected[last]).abs().max()),
            'production_code_changed':False,'real_model_repaired':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=[observe_bn(*v) for v in ((2.,1.,2,True),(-2.,1.,2,True),(1.,1.,2,True),(2.,1.,2,False),(2.,1.,1,True))]
    write(a.output,{'finding':'BN_BIAS_EDGE_BYPASSES_SCALE','cases':rows,'native_queries':0,
        'converter_sha256':sha256(Path('act/pipeline/verification/torch2act.py')),
        'diagnostic_sha256':sha256(Path(__file__)),
        'limit':'Toy counterfactual only; no real-model repair, no source-complete proof.'})
    print(json.dumps(rows,indent=2))
