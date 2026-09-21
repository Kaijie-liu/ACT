import copy
import itertools
import unittest
from unittest.mock import patch
import torch
from torch import nn

from act.back_end.moe.multilayer import RoutedLayer, HistoryPlan, verify_multilayer_box
from act.back_end.moe.multilayer_audit import audit_multilayer_result
from act.util.device_manager import initialize_device


def affine(w, b):
    m = nn.Linear(len(w[0]), len(w), dtype=torch.float64)
    with torch.no_grad():
        m.weight.copy_(torch.tensor(w, dtype=torch.float64))
        m.bias.copy_(torch.tensor(b, dtype=torch.float64))
    return m.eval()


def example(unsafe=False):
    first = RoutedLayer(affine([[1.],[-1.]], [0.,0.]),
        [affine([[1.]], [0.]), affine([[-1.]], [0.])])
    second = RoutedLayer(affine([[1.],[-1.]], [-.5,.5]),
        [affine([[-1.]] if unsafe else [[1.]], [.25] if unsafe else [1.]),
         affine([[-1.]], [2.])])
    return nn.Sequential(first, second, affine([[1.],[0.]], [0.,0.])).double().eval()


class MultilayerControls(unittest.TestCase):
    def setUp(self):
        initialize_device('cpu','float64')
        torch.set_num_threads(1)
        self.center = torch.tensor([[0.]], dtype=torch.float64)
        self.kwargs = dict(center=self.center, lower=self.center-1, upper=self.center+1,
            rows=torch.tensor([[1.,-1.]],dtype=torch.float64), thresholds=torch.zeros(1,dtype=torch.float64), total_seconds=30.)

    def test_full_graph_history_grid_differential(self):
        model = example()
        plan = HistoryPlan(model, self.center)
        self.assertEqual(plan.total_histories,4)
        for history in plan.histories():
            static, score_rows = plan.compile(history)
            observed = 0
            for value in [-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.]:
                x = torch.tensor([[value]], dtype=torch.float64)
                out = static(x)
                legal = all(all(out[0,indices[i]] >= out[0,indices[j]]
                    for i in selected for j in range(len(indices)) if j not in selected)
                    for selected,indices in zip(history,score_rows))
                if legal:
                    self.assertTrue(torch.equal(out[:,:2], model(x)))
                    observed += 1
            self.assertGreater(observed,0)

    def test_two_layer_route_changing_positive(self):
        result = verify_multilayer_box(example(), **self.kwargs)
        self.assertEqual(result['status'],'POSITIVE',result)
        self.assertEqual(len(result['records']),4)
        self.assertTrue(result['complete'])
        self.assertEqual(result['evidence_grade'],'HZ_POLICY_ACCEPTED')
        self.assertFalse(result['source_complete'])

    def test_nonclean_history_full_model_unsafe(self):
        result = verify_multilayer_box(example(True), **self.kwargs)
        self.assertEqual(result['status'],'UNSAFE_REPLAYED',result)
        x = torch.tensor(result['witness'])
        self.assertTrue((x.abs() <= 1).all())
        self.assertLess(example(True)(x)[0,0],0)

    def test_history_cap_never_accepts_partial_coverage(self):
        result = verify_multilayer_box(example(), max_histories=3, **self.kwargs)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual(result['records'],[])
        self.assertFalse(result['complete'])

    def test_previous_outputs_constrain_later_route(self):
        model = example()
        model[1].router = affine([[1.],[-1.]], [2.,0.])
        result = verify_multilayer_box(model, **self.kwargs)
        self.assertEqual(result['status'],'POSITIVE',result)
        self.assertEqual(sum(r['status']=='EXCLUDED' for r in result['records']),2)

    def test_repeated_module_calls_are_distinct_sites(self):
        layer = RoutedLayer(affine([[1.],[-1.]],[0.,0.]),
            [affine([[1.]],[1.]), affine([[-1.]],[1.])]).eval()
        class Shared(nn.Module):
            def __init__(self):
                super().__init__(); self.layer=layer
            def forward(self,x):
                return self.layer(self.layer(x))
        plan = HistoryPlan(Shared().eval(), self.center)
        self.assertEqual(len(plan.sites),2)
        self.assertNotEqual(plan.sites[0].name,plan.sites[1].name)
        self.assertEqual(plan.sites[0].target,plan.sites[1].target)

    def test_ties_are_all_obligations_not_clean_topk(self):
        model = example()
        model[0].router = affine([[0.],[0.]], [0.,0.])
        plan = HistoryPlan(model,self.center)
        self.assertEqual(len(list(plan.histories())),4)
        with self.assertRaises(ValueError):
            plan.compile(((0,),))

    def test_bad_mode_dtype_and_dimensions(self):
        with self.assertRaises(ValueError):
            verify_multilayer_box(example().train(),**self.kwargs)
        with self.assertRaises(ValueError):
            verify_multilayer_box(example().float(),**self.kwargs)
        with self.assertRaises(ValueError):
            verify_multilayer_box(example(),**{**self.kwargs,'rows':torch.ones(1,3)})

    def test_weighted_intermediate_top2_and_residual(self):
        for mode in ['selected_softmax','raw_epsilon']:
            first = example()[0]
            router = affine([[.25],[-.25],[0.]], [2.,2.,1.])
            weighted = RoutedLayer(router, [affine([[0.]],[1.]),affine([[0.]],[2.]),affine([[0.]],[3.])],k=2,mode=mode)
            model = nn.Sequential(first,weighted,affine([[1.],[0.]],[0.,0.])).eval()
            plan = HistoryPlan(model,self.center)
            for history in plan.histories():
                compiled, score_rows=plan.compile(history)
                for val in [-.8,.2,.8]:
                    x=torch.tensor([[val]],dtype=torch.float64)
                    out=compiled(x)
                    if all(all(out[0,rr[i]]>=out[0,rr[j]] for i in selected for j in range(len(rr)) if j not in selected)
                           for selected,rr in zip(history,score_rows)):
                        self.assertTrue(torch.allclose(out[:,:2],model(x),atol=1e-12,rtol=0))
            result=verify_multilayer_box(model,**self.kwargs)
            if mode == 'selected_softmax':
                self.assertEqual(result['status'],'POSITIVE',(mode,result))
            else:
                self.assertEqual(result['status'],'UNSUPPORTED',result)
                self.assertIn('backend_joint_HZ_not_retained',result['reason'])

    def test_zero_ste_cannot_silently_become_weight_one(self):
        layer=RoutedLayer(affine([[1.],[1.]],[0.,-2.]),[affine([[0.]],[1.]),affine([[0.]],[1.])],mode='nonzero_ste')
        model=nn.Sequential(layer,affine([[1.],[0.]],[0.,0.])).eval()
        result=verify_multilayer_box(model,**{**self.kwargs,'center':torch.tensor([[.5]])})
        self.assertNotEqual(result['status'],'POSITIVE',result)

    def test_prefix_definedness_not_hidden_by_later_guard(self):
        first=RoutedLayer(affine([[1.],[0.]],[1.,-2.]),
            [affine([[0.]],[2.]),affine([[0.]],[2.])],mode='nonzero_ste')
        second=RoutedLayer(affine([[1.],[-1.]],[-1.,1.]),
            [affine([[0.]],[1.]),affine([[0.]],[-1.])])
        model=nn.Sequential(first,second,affine([[1.],[0.]],[0.,0.])).eval()
        result=verify_multilayer_box(model,**self.kwargs)
        self.assertNotEqual(result['status'],'POSITIVE',result)

    def test_audit_missing_duplicate_property_binding_mutations(self):
        result=verify_multilayer_box(example(),**self.kwargs)
        request_id=result['request_id']
        self.assertEqual(audit_multilayer_result(result,expected_request_id=request_id)['status'],'PASS')
        mutations=[]
        bad=copy.deepcopy(result); bad['records'].pop(); mutations.append(bad)
        bad=copy.deepcopy(result); bad['records'][-1]=bad['records'][0]; mutations.append(bad)
        bad=copy.deepcopy(result); bad['records'][0]['properties']=[]; mutations.append(bad)
        bad=copy.deepcopy(result); bad['records'][0]['properties'][0]['lower']=-1.; mutations.append(bad)
        bad=copy.deepcopy(result); bad['identity']['properties']=2; mutations.append(bad)
        bad=copy.deepcopy(result); bad['source_complete']=True; mutations.append(bad)
        for bad in mutations:
            self.assertEqual(audit_multilayer_result(bad,expected_request_id=request_id)['status'],'FAIL')

    def test_unknown_feasibility_is_not_excluded(self):
        from types import SimpleNamespace
        with patch('act.back_end.solver.solver_hz.hz_check_feasibility',return_value=SimpleNamespace(status='unknown')):
            result=verify_multilayer_box(example(),**self.kwargs)
        self.assertEqual(result['status'],'POSITIVE')
        self.assertTrue(all(r['feasibility']=='unknown' and r['status']=='ACCEPTED' for r in result['records']))

    def test_rng_progress_isolation_and_multiple_properties(self):
        model=example()
        plan=HistoryPlan(model,self.center)
        state=torch.random.get_rng_state().clone()
        plan.compile(next(plan.histories()))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        result=verify_multilayer_box(model,**{**self.kwargs,'rows':torch.tensor([[1.,-1.],[2.,0.]]),
            'thresholds':torch.tensor([0.,0.]),'progress':lambda r:r['records'].clear()})
        self.assertEqual(result['status'],'POSITIVE')
        self.assertEqual(len(result['records']),4)
        self.assertTrue(all(len(r['properties'])==2 for r in result['records']))

    def test_residual_relu_convolution_history(self):
        def conv(weight,bias):
            m=nn.Conv2d(1,1,1,dtype=torch.float64)
            with torch.no_grad(): m.weight.fill_(weight); m.bias.fill_(bias)
            return m
        def router():
            return nn.Sequential(nn.Flatten(),affine([[1.],[-1.]],[0.,0.]))
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.a=RoutedLayer(router(),[conv(1.,2.),conv(-1.,2.)])
                self.relu=nn.ReLU()
                self.b=RoutedLayer(router(),[conv(.25,1.),conv(-.25,1.)])
                self.output=nn.Sequential(nn.Flatten(),affine([[1.],[0.]],[0.,0.]))
            def forward(self,x):
                a=self.relu(self.a(x))
                return self.output(self.b(a)+x)
        x=torch.zeros(1,1,1,1,dtype=torch.float64)
        result=verify_multilayer_box(Residual().eval(),**{**self.kwargs,'center':x,'lower':x-.5,'upper':x+.5})
        self.assertEqual(result['status'],'POSITIVE',result)

    def test_tie_only_bad_branch_must_not_prove(self):
        # Actual torch tie pick may be safe; ANY_LEGAL_TOPK includes both.
        layer=RoutedLayer(affine([[0.],[0.]],[1.,1.]),[affine([[0.]],[1.]),affine([[0.]],[-1.])])
        model=nn.Sequential(layer,affine([[1.],[0.]],[0.,0.])).eval()
        result=verify_multilayer_box(model,**self.kwargs)
        self.assertNotEqual(result['status'],'POSITIVE')

    def test_deadline_and_nonoptimal_bound_are_not_positive(self):
        result=verify_multilayer_box(example(),**{**self.kwargs,'total_seconds':1e-8})
        self.assertEqual(result['status'],'TIMEOUT')
        from types import SimpleNamespace
        fake=SimpleNamespace(status='timeout',minimum=100.,solver_status=1,solver_bound_kind='incumbent',elapsed=.0,candidate_input=None)
        with patch('act.back_end.solver.solver_hz.hz_minimize_output',return_value=fake):
            result=verify_multilayer_box(example(),**self.kwargs)
        self.assertEqual(result['status'],'UNKNOWN')

    def test_batch_dependent_bn_rejected(self):
        model=example()
        model[0].router=nn.Sequential(nn.BatchNorm1d(1,track_running_stats=False),model[0].router).eval()
        result=verify_multilayer_box(model,**self.kwargs)
        self.assertEqual(result['status'],'UNSUPPORTED')

    def test_author_raw_dispatch_contract_is_fail_closed(self):
        layer=RoutedLayer(affine([[0.],[0.]],[1.,.5]),[affine([[0.]],[1.]),affine([[0.]],[1.])],mode='raw')
        model=nn.Sequential(layer,affine([[1.],[0.]],[0.,0.])).eval()
        # Contract gate control; author-class execution has its separate source
        # compatibility protocol, not a synthetic claim of author reproduction.
        with patch('act.back_end.moe.multilayer._author_layer',side_effect=lambda m:type(m) is RoutedLayer and m.mode=='raw'):
            result=verify_multilayer_box(model,**self.kwargs)
        self.assertEqual(result['status'],'UNSUPPORTED')
        self.assertEqual(result['reason'],'author_raw_dispatch_definedness_not_supported')


if __name__=='__main__':
    unittest.main()
