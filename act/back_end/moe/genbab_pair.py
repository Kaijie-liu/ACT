"""Opt-in real-semantics pair graph exposing nonlinear gates and router rows.

Native floating-point equivalence is not asserted. The old softmax/CROWN graph
is deliberately unchanged. No external library is imported by this adapter.
"""
import torch
from act.back_end.moe.static_pair import StaticSelectedSoftmaxPair


class GenBaBPairWithRouter(StaticSelectedSoftmaxPair):
    """Output [F_{a,b}, r] on one shared input; a<b is not score ordering."""

    def forward(self, x):
        scores = self.router(x)
        a, b = self.pair
        weight = torch.sigmoid(scores[:, a:a+1] - scores[:, b:b+1])
        ea, eb = self.expert_a(x), self.expert_b(x)
        return torch.cat((eb + weight * (ea - eb), scores), dim=1)
