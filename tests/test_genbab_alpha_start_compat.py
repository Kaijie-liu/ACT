"""Regression for a final alpha omitted after zero initial backward A."""
import ast
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


@unittest.skipUnless(os.environ.get('MOE_DIAGNOSTIC_NATIVE')=='1','isolated native control')
class AlphaStartCompatibility(unittest.TestCase):
    def test_zero_path_final_allocation_and_unchanged_existing_specs(self):
        import torch
        import auto_LiRPA.backward_bound as B
        from auto_LiRPA.operators.relu import BoundRelu
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        # Compile the unchanged author's function in a private namespace. No
        # imported class/function is patched, and no second library is loaded.
        source=(Path('/data1/Kane/MOE/baselines/adaptations/genbab_cpu_20261022_r3')/
            'auto_LiRPA/auto_LiRPA/backward_bound.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
                  and n.name=='get_alpha_crown_start_nodes')
        namespace=dict(B.get_alpha_crown_start_nodes.__globals__)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged-r3-function>','exec'),namespace)
        old=namespace['get_alpha_crown_start_nodes'];new=B.get_alpha_crown_start_nodes
        class Net:
            bound_opts={}
            def __getitem__(self,k):return self.nodes[k]
        net=Net();middle=SimpleNamespace(name='middle',lower=torch.zeros(1,3))
        final=SimpleNamespace(name='final',output_shape=(1,9))
        activation=SimpleNamespace(name='relu',used=True)
        net.nodes={'final':final};net.backward_from={'relu':[middle]}
        c=torch.eye(9).unsqueeze(0)
        before=old(net,activation,c=c,final_node_name='final')
        after=new(net,activation,c=c,final_node_name='final')
        self.assertEqual(before,[('middle',torch.Size([3]),None,False)])
        self.assertEqual(after,before+[('final',9,None,True)])
        inp=SimpleNamespace(lower=torch.full((1,4),-1.),upper=torch.ones(1,4))
        relu=BoundRelu(inputs=[inp],options={});relu.shape=(4,)
        relu.init_d=torch.full((2,1,1,4),.5)
        relu.init_opt_parameters(before)
        with self.assertRaises(KeyError):
            relu.select_alpha_by_idx(torch.ones(9,1,4),None,None,final)
        relu.init_opt_parameters(after)
        selected,lookup=relu.select_alpha_by_idx(torch.ones(9,1,4),None,None,final)
        self.assertEqual(tuple(selected.shape),(2,9,1,4));self.assertIsNone(lookup)
        self.assertTrue(bool(((selected>=0)&(selected<=1)).all()))
        net.backward_from['relu']=[middle,final]
        self.assertEqual(old(net,activation,c=c,final_node_name='final'),
                         new(net,activation,c=c,final_node_name='final'))
        net.backward_from['relu']=[]
        self.assertEqual(new(net,activation,c=c,final_node_name='final'),[])
        net.backward_from['relu']=[middle];activation.used=False
        self.assertEqual(new(net,activation,c=c,final_node_name='final'),before)

    def test_mixed_property_graph_keeps_pruning_and_real_negative(self):
        import torch
        from b2_observed_worker import verify_graph
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        class Mixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Match the registered model's affine -> ReLU topology. The
                # first control's bare-input ReLU hit a separate BBPS input-A
                # limitation; retain that failed control, not a backend patch.
                self.first=torch.nn.Linear(1,1)
                with torch.no_grad():
                    self.first.weight.fill_(1.);self.first.bias.zero_()
            def forward(self,x):
                p=.1*torch.relu(self.first(x))-.25
                f=p+torch.relu(-.18-p)
                return torch.cat((x*0+.2,f),1)
        graph=Mixed().eval();lo=torch.tensor([[-1.]]);hi=-lo
        # Exact second-property minimum is -.18. A positive terminal would be
        # an error; UNKNOWN is a valid test outcome without a new success gate.
        self.assertAlmostEqual(float(graph(lo)[0,1]),-.18)
        record,_=verify_graph(graph,lo,hi,2,min(float(os.environ['MOE_CONTROL_DEADLINE']),time.monotonic()+8))
        self.assertIn(record['status'],('unknown','falsified'))
        self.assertEqual(len(record['actual_raw_spec']),2)


if __name__=='__main__':unittest.main()
