"""Frozen sequential selected-softmax models; no verification-based selection."""
from types import SimpleNamespace
import torch
from torch import nn
from act.back_end.moe.factory import _mlp
from act.back_end.moe.multilayer import RoutedLayer
from act.back_end.moe.multilayer_external import SequentialTop2


def build(config, seed):
    if config.get('family')=='residual_stem_v1':
        return build_residual(config,seed)
    if config.get('family') is not None:raise ValueError('unknown multilayer family')
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


def build_residual(config,seed):
    """Stage1b: every routed block128->128, explicit linear stem/head."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        width=config['hidden_output']
        stem=nn.Linear(config['input_dim'],width)
        layers=[RoutedLayer(_mlp(width,config['router_hidden'],config['experts']),
            [_mlp(width,config['expert_hidden'],width) for _ in range(config['experts'])],
            k=2,mode='selected_softmax') for _ in range(config['layers'])]
        head=nn.Linear(width,config['classes'])
    return TrainableResidual(stem,layers,head)


def balance_term(decisions,experts,config):
    from act.pipeline.moe.train import _balance_loss
    terms=torch.stack([_balance_loss(d,experts,'switch') for d in decisions])
    return terms.sum() if config.get('family')=='residual_stem_v1' else terms.mean()


class TrainableResidual(nn.Module):
    def __init__(self,stem,layers,head):
        super().__init__();self.stem=stem;self.layers=nn.ModuleList(layers);self.head=head
        self.residual=True

    def forward(self,x):
        x=self.stem(x.flatten(1))
        for layer in self.layers:x=x+layer(x)
        return self.head(x)

    def forward_with_routing(self,x):
        x=self.stem(x.flatten(1));decisions=[]
        for layer in self.layers:
            scores=layer.router(x);selected,indices=scores.topk(2,dim=1)
            weights=selected.softmax(dim=1)
            values=torch.stack([expert(x) for expert in layer.experts],dim=1)
            chosen=values.gather(1,indices[:,:,None].expand(-1,-1,values.shape[2]))
            x=x+(chosen*weights[:,:,None]).sum(1)
            decisions.append(SimpleNamespace(scores=scores,indices=indices,weights=weights))
        return self.head(x),decisions


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
