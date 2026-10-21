"""Frozen sequential selected-softmax models; no verification-based selection."""
from types import SimpleNamespace
import torch
from torch import nn
from act.back_end.moe.factory import _mlp
from act.back_end.moe.multilayer import RoutedLayer
from act.back_end.moe.multilayer_external import SequentialTop2


def build(config, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers=[]
        width=config['input_dim']
        for depth in range(config['layers']):
            output=config['classes'] if depth+1==config['layers'] else config['hidden_output']
            layers.append(RoutedLayer(_mlp(width,config['router_hidden'],config['experts']),
                [_mlp(width,config['expert_hidden'],output) for _ in range(config['experts'])],
                k=2,mode='selected_softmax'))
            width=output
    return TrainableSequential(layers)


class TrainableSequential(SequentialTop2):
    def forward_with_routing(self,x):
        x=x.flatten(1)
        decisions=[]
        for layer in self.layers:
            scores=layer.router(x)
            selected,indices=scores.topk(2,dim=1)
            weights=selected.softmax(dim=1)
            values=torch.stack([expert(x) for expert in layer.experts],dim=1)
            chosen=values.gather(1,indices[:,:,None].expand(-1,-1,values.shape[2]))
            x=(chosen*weights[:,:,None]).sum(1)
            decisions.append(SimpleNamespace(scores=scores,indices=indices,weights=weights))
        return x,decisions


def load_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['format']!='act-sequential-top2-v1':raise ValueError('multilayer checkpoint format')
    model=build(payload['architecture'],payload['seed'])
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.eval(),payload
