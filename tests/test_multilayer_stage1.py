"""New census/denominator/budget integration controls; no trained checkpoint."""
import copy
import json
import math
import time
import unittest
from unittest.mock import patch
import torch
from torch import nn
from act.pipeline.moe.multilayer_study import (coverage,wilson,endpoint,affine_interval,
    pair_interval,interval_census,public_pgd)
from act.pipeline.moe.multilayer_training import TrainableSequential
from act.back_end.moe.multilayer import RoutedLayer
from act.back_end.moe.multilayer_external import verify_histories


def linear(w,b):
    layer=nn.Linear(len(w[0]),len(w),dtype=torch.float64)
    with torch.no_grad():layer.weight.copy_(torch.tensor(w));layer.bias.copy_(torch.tensor(b))
    return layer.eval()


def model():
    router=linear([[0.],[0.],[0.]],[0.,0.,0.])
    a=RoutedLayer(router,[linear([[1.]],[2.]),linear([[-1.]],[2.]),linear([[0.]],[2.])],k=2,mode='selected_softmax')
    b=RoutedLayer(copy.deepcopy(router),[linear([[0.],[0.]],[1.,0.]) for _ in range(3)],k=2,mode='selected_softmax')
    return TrainableSequential([a,b]).eval()


class Controls(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_external_active_prefix_and_unknown_traversal(self):
        x=torch.zeros(1,1,dtype=torch.float64);events=[]
        kw=dict(center=x,lower=x-1,upper=x+1,rows=torch.tensor([[1.,-1.]],dtype=torch.float64),
            thresholds=torch.zeros(1,dtype=torch.float64),pruning=True,deadline=time.monotonic()+20,
            publish=lambda r:events.append(copy.deepcopy(r)))
        def bounds(*_):return {}
        def verifier(*_):return dict(status='unknown'),None
        result=verify_histories(model(),**kw,verify_graph=verifier,bound_router=bounds)
        cov=coverage(result,'genbab_pruned',9)
        self.assertTrue(cov['traversal_complete']);self.assertFalse(cov['proof_complete'])
        self.assertEqual(cov['returned_unproved'],9)
        self.assertTrue(any((r['active'] or {}).get('phase')=='prefix_CROWN' and
            r['attempted_prefix_queries']>r['prefix_queries'] for r in events))
    def test_prefix_timeout_is_incomplete_not_zero_work(self):
        x=torch.zeros(1,1,dtype=torch.float64)
        def timeout(*_):raise TimeoutError('fixed')
        value=verify_histories(model(),center=x,lower=x-1,upper=x+1,
            rows=torch.tensor([[1.,-1.]],dtype=torch.float64),thresholds=torch.zeros(1,dtype=torch.float64),
            pruning=True,deadline=time.monotonic()+5,verify_graph=lambda *_:None,bound_router=timeout)
        cov=coverage(value,'genbab_pruned',9)
        self.assertEqual(cov['prefix_attempts'],1);self.assertEqual(cov['prefix_returns'],0)
        self.assertEqual(cov['never_started'],9);self.assertEqual(value['status'],'TIMEOUT')
    def test_act_counter_convention_and_bad_count(self):
        records=[dict(status='EXCLUDED',phase='complete'),dict(status='UNKNOWN',phase='complete'),
            dict(status='UNKNOWN',phase='property')]
        c=coverage(dict(records=records),'act',9)
        self.assertEqual((c['never_started'],c['in_flight'],c['returned_unproved']),(6,1,1))
        with self.assertRaises(ValueError):coverage(dict(pruned_histories=-1),'genbab_pruned',9)
    def test_wilson_and_unsafe_union(self):
        self.assertEqual(wilson(0,60)['decision'],'STOP_NO_BOTTLENECK')
        self.assertEqual(wilson(60,60)['decision'],'ENTER_STAGE2')
        self.assertEqual(wilson(0,0)['decision'],'GRAY_UNDEFINED')
        rows=[dict(model='m',index=i,coverage=dict(traversal_complete=False)) for i in range(60)]
        value=endpoint(rows,{('m',i) for i in range(60)})
        self.assertEqual(value['adjusted']['n'],0);self.assertEqual(value['raw']['n'],60)
    def test_interval_and_weighted_range(self):
        layer=nn.Sequential(linear([[1.,-2.],[-1.,1.]],[.25,-.5]),nn.ReLU())
        lo=torch.tensor([[-1.,-.5]],dtype=torch.float64);hi=-lo
        a,b=affine_interval(layer,lo,hi)
        for p in (lo,hi,torch.zeros_like(lo)):
            self.assertTrue(bool(((layer(p)>=a)&(layer(p)<=b)).all()))
        for d in (-1.,0.,1.):
            rlo=torch.tensor([[-1.,0.]],dtype=torch.float64);rhi=torch.tensor([[1.,0.]],dtype=torch.float64)
            outlo,outhi=pair_interval(0,1,(a,b),(a-1,b+1),rlo,rhi)
            lam=torch.sigmoid(torch.tensor(d,dtype=torch.float64));v=lam*a+(1-lam)*(a-1)
            self.assertTrue(bool(((v>=outlo-1e-12)&(v<=outhi+1e-12)).all()))
    def test_tie_census_and_deadline(self):
        x=torch.zeros(1,1,dtype=torch.float64)
        census=interval_census(model(),x,x-1,x+1,time.monotonic()+10,0)
        self.assertEqual(census['candidate_upper'],9);self.assertTrue(census['traversal_complete'])
        partial=interval_census(model(),x,x-1,x+1,time.monotonic()-1,0)
        self.assertEqual(partial['candidate_upper'],9);self.assertFalse(partial['traversal_complete'])
    def test_public_attack_zero_budget_not_proof(self):
        x=torch.zeros(1,1,dtype=torch.float64)
        result=public_pgd(model(),dict(center=x,lower=x-1,upper=x+1,
            rows=torch.tensor([[1.,-1.]],dtype=torch.float64),thresholds=torch.zeros(1,dtype=torch.float64)),
            time.monotonic()-1,0)
        self.assertFalse(result['completed_fixed_search']);self.assertIsNone(result['witness'])
    def test_serialized_external_coverage_and_missing_rejected(self):
        from multilayer_stage1 import external_audit
        raw=[([[1.,0.]],[0.]),([[0.,1.]],[0.])]
        value=dict(status='POSITIVE',histories=[dict(history=[[0,1]],outcome=dict(status='verified',actual_raw_spec=raw))],
            exclusions=[],prefix_bounds=[],verified_histories=1,pruned_histories=0,returned_histories=1)
        self.assertEqual(external_audit(json.loads(json.dumps(value)),[2],2)['status'],'PASS')
        bad=copy.deepcopy(value);bad['histories'][0]['outcome']['actual_raw_spec']=raw[:1]
        with self.assertRaises(ValueError):external_audit(bad,[2],2)
        bad=copy.deepcopy(value);bad['histories']=[]
        with self.assertRaises(ValueError):external_audit(bad,[2],2)
    def test_full_shape_count_no_allocation(self):
        expected={(1,8):28,(2,8):784,(3,8):21952,(2,16):14400}
        for (layers,experts),value in expected.items():
            self.assertEqual(math.comb(experts,2)**layers,value)
        self.assertEqual(9*20*3+3*20,600)


if __name__=='__main__':unittest.main()
