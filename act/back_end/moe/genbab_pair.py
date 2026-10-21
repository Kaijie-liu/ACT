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


class GenBaBGuardedProperties(GenBaBPairWithRouter):
    """One output per property: max(qF+c, all outsider-minus-selected guards).

    A nonpositive output is an unsafe guarded conjunction. Multiple property
    violations are OR alternatives, never a single conjunction of these outputs.
    The graph retains the same variable sigmoid weight and shared input.
    """

    def __init__(self, model, pair, properties, *, guarded=True):
        super().__init__(model,pair)
        reference=next(model.parameters())
        q=torch.as_tensor([p['q'] for p in properties],dtype=reference.dtype,device=reference.device)
        c=torch.as_tensor([p['c'] for p in properties],dtype=reference.dtype,device=reference.device)
        if q.ndim!=2 or q.shape[0]==0 or c.shape!=(q.shape[0],) or not torch.isfinite(q).all() or not torch.isfinite(c).all():
            raise ValueError('complete finite property rows/constants')
        self.register_buffer('property_rows',q)
        self.register_buffer('property_constants',c)
        self.classes=q.shape[1]
        self.guard_pairs=tuple((selected,outside) for selected in self.pair
                              for outside in range(model.spec.num_experts) if outside not in self.pair) if guarded else ()

    def forward(self,x):
        joint=super().forward(x)
        values=torch.nn.functional.linear(joint[:,:self.classes],self.property_rows,self.property_constants)
        if not self.guard_pairs:return values
        scores=joint[:,self.classes:]
        first=self.guard_pairs[0]
        guard=scores[:,first[1]:first[1]+1]-scores[:,first[0]:first[0]+1]
        for selected,outside in self.guard_pairs[1:]:
            atom=scores[:,outside:outside+1]-scores[:,selected:selected+1]
            guard=guard+torch.relu(atom-guard)
        return values+torch.relu(guard-values)


def scalar_bad_set_clauses(property_count):
    """Raw VNNLIB-style OR list: each alternative has exactly ONE <=0 row."""
    if type(property_count) is not int or property_count<=0:
        raise ValueError('nonempty property roster')
    return [([[float(j==i) for j in range(property_count)]],[0.])
            for i in range(property_count)]
