import unittest
import torch
from torch import nn
from act.back_end.moe.model import OutputLevelMoE
from act.back_end.moe.schema import OutputLevelMoESpec, GateKind
from act.back_end.moe.static_pair import StaticSelectedSoftmaxPair


class StaticPairTests(unittest.TestCase):
    def model(self):
        router=nn.Linear(2,3,dtype=torch.float64)
        experts=[nn.Linear(2,2,dtype=torch.float64) for _ in range(3)]
        with torch.no_grad():
            router.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[-1.,0.]]));router.bias.zero_()
            for i,e in enumerate(experts):
                e.weight.copy_(torch.tensor([[i+1.,1.],[-1.,i+2.]]));e.bias.fill_(i)
        return OutputLevelMoE(router,experts,OutputLevelMoESpec(3,2,GateKind.SELECTED_SOFTMAX)).eval()

    def test_forced_branches_and_literal_legal_route(self):
        model=self.model();x=torch.tensor([[.2,.1],[-.2,.1],[0.,0.]],dtype=torch.float64)
        for pair in [(0,1),(0,2),(1,2)]:
            adapter=StaticSelectedSoftmaxPair(model,pair)
            values=torch.stack([model.experts[i](x) for i in pair],dim=1)
            reference=(torch.softmax(model.router(x)[:,list(pair)],dim=1).unsqueeze(-1)*values).sum(dim=1)
            torch.testing.assert_close(adapter(x),reference,rtol=0,atol=0)
        full,route=model.forward_with_routing(x)
        for i in range(len(x)):
            pair=sorted(route.indices[i].tolist())
            torch.testing.assert_close(StaticSelectedSoftmaxPair(model,pair)(x[i:i+1]),full[i:i+1],rtol=0,atol=1e-14)

    def test_gate_varies_and_input_gradient_includes_router(self):
        model=self.model();x=torch.tensor([[.2,.1]],dtype=torch.float64,requires_grad=True)
        adapter=StaticSelectedSoftmaxPair(model,[0,1]);result=adapter(x)
        dynamic=torch.autograd.grad(result.sum(),x,retain_graph=True)[0]
        weights=torch.softmax(model.router(x)[:,:2],dim=1).detach()
        frozen=weights[:,0:1]*model.experts[0](x)+weights[:,1:2]*model.experts[1](x)
        frozen_grad=torch.autograd.grad(frozen.sum(),x)[0]
        self.assertGreater(float((dynamic-frozen_grad).abs().max()),1e-3)

    def test_reject_training_wrong_pair_and_gate(self):
        model=self.model()
        for pair in ([1,0],[0,0],[0,3],[True,2],[0]):
            with self.assertRaises(ValueError):StaticSelectedSoftmaxPair(model,pair)
        model.train()
        with self.assertRaises(ValueError):StaticSelectedSoftmaxPair(model,[0,1])
        model.eval();model.shared_expert=nn.Identity()
        with self.assertRaises(ValueError):StaticSelectedSoftmaxPair(model,[0,1])


if __name__=='__main__':unittest.main()
