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
    def test_branch_cache_preparation_uses_native_strict_stop(self):
        import torch
        from auto_LiRPA.utils import stop_criterion_batch_any
        import genbab_cpu_ready as b
        # Execute the installed build method's actual cache-preparation branch
        # privately; no library function is replaced. Check its calls against
        # the public native strict predicate, including vector OR clauses/RHS.
        tree=ast.parse((b.COMPAT/'complete_verifier/beta_CROWN_solver.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LiRPANet')
        build=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='build')
        node=next(n for n in build.body if isinstance(n,ast.If) and any(
            isinstance(t,ast.Call) and isinstance(t.func,ast.Name) and t.func.id=='precompute_A'
            for t in ast.walk(n)))
        code=compile(ast.Module(body=[node],type_ignores=[]),'<installed-cache-branch>','exec')
        for lower,rhs,expected in [([[0.]],[[0.]],True),([[-.1]],[[0.]],True),
                ([[.1]],[[0.]],False),([[.1]],[[.2]],True),
                ([[-.1]],[[-.2]],False),([[0.,.1]],[[0.,0.]],False),
                ([[0.,0.]],[[0.,0.]],True)]:
            for enabled,return_a in [(True,True),(False,True),(True,False)]:
                calls=[];a={};x=object()
                env=dict(self=SimpleNamespace(nonlinear_split=enabled,return_A=return_a,net=object()),
                    lb=torch.tensor(lower),stop_criterion_func=stop_criterion_batch_any(torch.tensor(rhs)),
                    total_batches=1,A=a,batch_x=x,batch_lb={'n':torch.tensor([0.])},
                    batch_ub={'n':torch.tensor([1.])},
                    precompute_A=lambda *args,**kw:calls.append((args,kw)))
                exec(code,env)
                self.assertEqual(len(calls),int(expected and enabled and return_a))
                if calls:
                    self.assertIs(calls[0][0][1],a);self.assertIs(calls[0][0][2],x)

    def test_zero_bound_graph_stays_unaccepted_and_reaches_bab(self):
        import torch
        from b2_observed_worker import verify_graph
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        class Zero(torch.nn.Module):
            def __init__(self):
                super().__init__();self.affine=torch.nn.Linear(1,1)
                with torch.no_grad():self.affine.weight.fill_(1.);self.affine.bias.zero_()
            def forward(self,x):
                p=.1*torch.relu(self.affine(x))-.25
                return p+torch.relu(-p)
        graph=Zero().eval();lo=torch.tensor([[-1.]]);hi=-lo
        torch.testing.assert_close(graph(torch.tensor([[-1.],[0.],[1.]])),torch.zeros(3,1),rtol=0,atol=0)
        record,_=verify_graph(graph,lo,hi,1,
            min(float(os.environ['MOE_CONTROL_DEADLINE']),time.monotonic()+6))
        self.assertIn(record['status'],('unknown','falsified'))
        self.assertEqual(record['stage'],'nonlinear_bab')

    def test_root_input_scores_match_explicit_identity_cache(self):
        import torch
        from auto_LiRPA.bound_ops import BoundInput, BoundRelu
        from heuristics.nonlinear.bbps import NonlinearBranching
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        # Compare native backward ReLU relaxations for multiple specs, batches,
        # signs and input shapes. The oracle is the unchanged R5 routine with
        # an explicit identity A cache, compiled privately (no monkeypatch).
        import heuristics.nonlinear.bbps as B
        source=(Path('/data1/Kane/MOE/baselines/adaptations/genbab_cpu_20261026_r5')/
            'complete_verifier/heuristics/nonlinear/bbps.py').read_text()
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)
                 and n.name=='NonlinearBranching')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)
                  and n.name=='_fast_backward_propagation')
        namespace=dict(B.__dict__)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<r5-bbps>','exec'),namespace)
        old=namespace['_fast_backward_propagation']
        for shape in ((2,4),(2,2,2)):
            root=BoundInput('input',torch.zeros(shape),options={})
            root.name='/input';root.output_shape=shape
            lo=torch.tensor([[-2.,-1.,0.,1.],[-3.,-1.,0.,2.]]).reshape(shape)
            hi=torch.tensor([[-1.,2.,0.,2.],[-2.,3.,0.,3.]]).reshape(shape)
            root.lower,root.upper=lo,hi
            relu=BoundRelu(inputs=[root],options={});relu.name='/relu'
            relu.output_shape=shape;relu.shape=shape[1:];relu.opt_stage=None
            final=SimpleNamespace(name='final',output_shape=(2,3))
            class Model:
                input_name=['/input'];final_name='final'
                def __getitem__(self,key):return final
            heuristic=SimpleNamespace(model=Model(),roots=[root],net=SimpleNamespace(A_saved={}))
            lA=torch.arange(24,dtype=torch.float64).reshape((2,3,*shape[1:]))/7-1.5
            args=({'/relu':lA},{'/input':lo},{'/input':hi},root,[relu])
            with self.assertRaises(KeyError):old(heuristic,*args)
            actual=NonlinearBranching._fast_backward_propagation(heuristic,*args)
            identity=torch.eye(4).reshape((1,4,*shape[1:]))
            heuristic.net.A_saved={'/input':{'/input':dict(lA=identity,uA=identity,
                lbias=torch.zeros(1,4),ubias=torch.zeros(1,4),unstable_idx=None)}}
            expected=old(heuristic,*args)
            for x,y in zip(actual,expected):
                if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
                else:self.assertEqual(x,y)
            self.assertEqual(tuple(actual[0].shape),(3,2,4,4))
            # A non-root predecessor with missing saved A must still fail.
            heuristic.roots=[SimpleNamespace(output_shape=shape)]
            heuristic.net.A_saved={}
            with self.assertRaises(KeyError):
                NonlinearBranching._fast_backward_propagation(heuristic,*args)

    def test_bare_input_relu_reaches_bab_without_input_cache(self):
        import torch
        from b2_observed_worker import verify_graph
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        class Bare(torch.nn.Module):
            def forward(self,x):
                p=.1*torch.relu(x)-.25
                return torch.cat((x*0+.2,p+torch.relu(-.18-p)),1)
        graph=Bare().eval();lo=torch.tensor([[-1.]]);hi=-lo
        self.assertAlmostEqual(float(graph(lo)[0,1]),-.18)
        record,_=verify_graph(graph,lo,hi,2,
            min(float(os.environ['MOE_CONTROL_DEADLINE']),time.monotonic()+8))
        self.assertIn(record['status'],('unknown','falsified'))
        self.assertEqual(len(record['actual_raw_spec']),2)

    def test_missing_stable_start_is_exact_without_new_parameters(self):
        import torch
        from auto_LiRPA.operators.relu import BoundRelu
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        inp=SimpleNamespace(lower=torch.tensor([[-2.,1.,0.,-1.],[-3.,2.,0.,-2.]]),
                            upper=torch.tensor([[-1.,2.,0.,0.],[-1.,3.,0.,0.]]))
        relu=BoundRelu(inputs=[inp],options={});relu.shape=(4,);relu.opt_stage='opt'
        relu.alpha_indices=(torch.empty(0,dtype=torch.long),)
        relu.alpha={'existing':torch.empty(2,9,2,0)};relu.alpha_lookup_idx={'existing':None}
        old=SimpleNamespace(name='existing',output_shape=(2,9))
        new=SimpleNamespace(name='missing',output_shape=(2,9))
        for indices in (None,torch.tensor([0,2])):
            n=9 if indices is None else len(indices)
            a=torch.ones(n,2,4)
            expected=relu._backward_relaxation(a,-a,inp,old,indices)
            actual=relu._backward_relaxation(a,-a,inp,new,indices)
            for x,y in zip(expected,actual):
                if isinstance(x,torch.Tensor):
                    x,y=torch.broadcast_tensors(x,y);torch.testing.assert_close(x,y,rtol=0,atol=0)
                else:self.assertEqual(x,y)
            self.assertEqual(set(relu.alpha),{'existing'})
            self.assertEqual(set(relu.alpha_lookup_idx),{'existing'})
        # A missing start with a genuinely unstable CURRENT bound must still
        # fail; zero-sized stored alpha alone is not sufficient evidence.
        inp.lower[0,1]=-1.
        with self.assertRaises(KeyError):
            relu._backward_relaxation(a,-a,inp,new,None)

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
