"""Explicit static selected-softmax pair, not a replacement dynamic MoE.

For a fixed legal pair S={a,b}, this is F_S on the SAME input with variable
weights. Proving it on the entire input box is sufficient, generally stronger
than proving F_S only on its guard. Exhaustive legal-route coverage is separate.
"""
import torch
from torch import nn
from act.back_end.moe.schema import GateKind


class StaticSelectedSoftmaxPair(nn.Module):
    def __init__(self, model, pair):
        super().__init__()
        if any(module.training for module in model.modules()):
            raise ValueError('static obligation requires eval model')
        if (model.spec.top_k != 2 or model.spec.gate != GateKind.SELECTED_SOFTMAX
                or not model.spec.normalized or model.shared_expert is not None):
            raise ValueError('only output selected-softmax top2 without shared expert')
        if (len(pair) != 2 or any(type(i) is not int for i in pair)
                or not 0 <= pair[0] < pair[1] < len(model.experts)):
            raise ValueError('canonical unordered pair required')
        self.pair = tuple(pair)
        self.router = model.router
        self.expert_a = model.experts[pair[0]]
        self.expert_b = model.experts[pair[1]]
        self.eval()

    def forward(self, x):
        scores = self.router(x)
        a, b = self.pair
        selected = torch.cat((scores[:, a:a+1], scores[:, b:b+1]), dim=1)
        weights = torch.softmax(selected, dim=1)
        return weights[:, 0:1] * self.expert_a(x) + weights[:, 1:2] * self.expert_b(x)
