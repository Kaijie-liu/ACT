"""Shared-input scalar guard semantics; no verifier or trained checkpoint."""
import unittest
import torch
from act.back_end.moe.factory import OutputMoEFactoryConfig,build_output_moe
from act.back_end.moe.schema import GateKind
from act.back_end.moe.genbab_pair import GenBaBGuardedProperties,scalar_bad_set_clauses


class GuardedPropertiesControls(unittest.TestCase):
    def model(self):
        model=build_output_moe(OutputMoEFactoryConfig(input_shape=(1,),num_classes=3,
            num_experts=4,top_k=2,gate=GateKind.SELECTED_SOFTMAX,expert_hidden=(),seed=0)).cpu().double().eval()
        with torch.no_grad():
            model.router[1].weight.copy_(torch.tensor([[1.],[-1.],[0.],[0.]]))
            model.router[1].bias.copy_(torch.tensor([0.,0.,-.5,-.25]))
            for expert,i in zip(model.experts,range(4)):
                expert[1].weight.copy_(torch.tensor([[(-1.)**i],[.125],[0.]]))
                expert[1].bias.copy_(torch.tensor([.5,0.,.25]))
        return model

    @torch.no_grad()
    def test_all_guards_constants_and_ties(self):
        model=self.model();props=[{'q':[1.,-1.,0.],'c':.125},{'q':[0.,1.,-1.],'c':-.25}]
        pair=(0,1);graph=GenBaBGuardedProperties(model,pair,props)
        self.assertEqual(len(graph.guard_pairs),4)
        for x in (-1.,-.25,0.,.25,1.):
            point=torch.tensor([[x]],dtype=torch.float64);r=model.router(point)[0]
            ea,eb=model.experts[0](point)[0],model.experts[1](point)[0]
            f=eb+torch.sigmoid(r[0]-r[1])*(ea-eb)
            guards=[float(r[j]-r[i]) for i in pair for j in (2,3)]
            wanted=[max(float(torch.tensor(p['q'],dtype=torch.float64)@f)+p['c'],*guards) for p in props]
            torch.testing.assert_close(graph(point)[0],torch.tensor(wanted,dtype=torch.float64),atol=1e-14,rtol=1e-14)

    def test_or_of_properties_not_and(self):
        clauses=scalar_bad_set_clauses(3)
        self.assertEqual(len(clauses),3)
        output=[1.,-1.,2.]
        bad=any(all(sum(c*y for c,y in zip(row,output))<=b for row,b in zip(C,rhs))
                for C,rhs in clauses)
        self.assertTrue(bad)
        self.assertTrue(all(len(C)==len(rhs)==1 for C,rhs in clauses))

    def test_whole_box_keeps_same_variable_gate(self):
        model=self.model();props=[{'q':[1.,-1.,0.],'c':.125}]
        graph=GenBaBGuardedProperties(model,(0,1),props,guarded=False)
        self.assertEqual(graph.guard_pairs,())
        outputs=[float(graph(torch.tensor([[x]],dtype=torch.float64))[0,0].detach()) for x in (-1.,0.,1.)]
        self.assertNotEqual(outputs[0],outputs[1])


if __name__=='__main__':unittest.main()
