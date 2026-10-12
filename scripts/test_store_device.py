"""Finite CPU and simulated-device controls; never initialize a CUDA driver."""
import copy
import math
import os
from pathlib import Path
import time
import tempfile
from types import SimpleNamespace
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import torch

from act.back_end.moe import store_device as d
from scoped_source.hz_request_store import identity, compact
from scoped_source.check_store_endpoints import EndpointWork

UUID='GPU-00000000-0000-0000-0000-000000000001'


class Controls(unittest.TestCase):
    def setUp(self):
        self.guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('physical CUDA forbidden'))
        self.lazy=self.guard.start(); self.addCleanup(self.guard.stop)
        self.env=patch.dict(os.environ,CUDA_VISIBLE_DEVICES=UUID)
        self.env.start(); self.addCleanup(self.env.stop)
        self.work=EndpointWork(time.monotonic()+30)
        self.args=dict(store='1'*64,request='2'*64,contract='3'*64,work=self.work)
        self.ctx=dict(schema=d.SCHEMA,store='1'*64,request='2'*64,contract='3'*64,
                      deadline=self.work.deadline,invocation='finite-control',gpu_uuid=UUID,
                      allocator_limit_bytes=d.ALLOCATOR_LIMIT)

    def tearDown(self): self.lazy.assert_not_called()

    def runtime(self):
        return d.Runtime('cuda:0',self.ctx,identity(self.ctx),**self.args)

    def test_context_before_initialization(self):
        bads=[None,{},dict(self.ctx,store='9'*64),dict(self.ctx,request='9'*64),
              dict(self.ctx,contract='9'*64),dict(self.ctx,schema='old'),
              dict(self.ctx,deadline=self.work.deadline+1),dict(self.ctx,deadline=True),
              dict(self.ctx,allocator_limit_bytes=2**29),dict(self.ctx,allocator_limit_bytes=True),
              dict(self.ctx,invocation=''),dict(self.ctx,gpu_uuid='GPU-foreign')]
        with patch.object(d,'_initialize_cuda') as init:
            for ctx in bads:
                with self.subTest(ctx=ctx),self.assertRaises(ValueError):
                    d.Runtime('cuda:0',ctx,identity(ctx),**self.args)
            with self.assertRaises(ValueError): d.Runtime('cuda:0',self.ctx,'0'*64,**self.args)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),self.assertRaises(ValueError): self.runtime()
            with self.assertRaises(ValueError): d.Runtime('cuda:1',None,None,**self.args)
            init.assert_not_called()

    def test_cpu_rejects_context_and_explicit_meta_allocation(self):
        with self.assertRaises(ValueError): d.Runtime('cpu',self.ctx,identity(self.ctx),**self.args)
        with self.assertRaises(ValueError): d.Runtime('cpu',None,'1'*64,**self.args)
        with torch.device('meta'):
            runtime=d.Runtime('cpu',None,None,**self.args)
            x=torch.tensor([[1.]],device='cpu',dtype=torch.float64)
            y,z=runtime.move((x,x))
            self.assertEqual(y.device.type,'cpu')
            self.assertEqual(runtime.read(y,z),([[1.]],[[1.]]))

    def test_capture_uses_exact_bytes_not_live_alias(self):
        before=copy.deepcopy(self.ctx); original=d.compact
        def mutated(value):
            raw=original(value); value['store']='9'*64; return raw
        with patch.object(d,'compact',side_effect=mutated):
            got=d.capture('cuda:0',self.ctx,identity(before),**self.args)
        self.assertEqual(got,before); self.assertNotEqual(got,self.ctx)
        self.ctx=copy.deepcopy(before)
        with patch.object(d,'compact',side_effect=mutated),self.assertRaises(ValueError):
            d.capture('cuda:0',self.ctx,identity(dict(before,store='9'*64)),**self.args)

    def test_initializer_failure_no_fallback(self):
        for error in (RuntimeError('init'),torch.cuda.OutOfMemoryError('allocation')):
            with patch.object(d,'_initialize_cuda',side_effect=error) as init,self.assertRaises(type(error)):
                self.runtime()
            init.assert_called_once_with(UUID)

    def test_runtime_initializes_once_and_private_context(self):
        with patch.object(d,'_initialize_cuda',return_value={'gpu_uuid':UUID}) as init,patch.object(d,'_synchronize'):
            runtime=self.runtime(); self.ctx['invocation']='changed'
            runtime.sync(); runtime.sync()
            self.assertEqual(runtime.context['invocation'],'finite-control')
            init.assert_called_once_with(UUID)

    def test_transfer_exception_no_readback(self):
        x=torch.tensor([1.],device='cpu',dtype=torch.float64)
        with patch.object(d,'_initialize_cuda',return_value={}),patch.object(d,'_synchronize'):
            runtime=self.runtime()
            with patch.object(torch.Tensor,'to',side_effect=torch.cuda.OutOfMemoryError('transfer')):
                with self.assertRaises(torch.cuda.OutOfMemoryError): runtime.move((x,))
            self.assertEqual(runtime.transfers,0)

    def test_sync_failure_and_readback_exception(self):
        runtime=d.Runtime('cpu',None,None,**self.args)
        x=torch.zeros((1,2),dtype=torch.float64,device='cpu')
        with patch.object(d,'_synchronize',side_effect=RuntimeError('sync')):
            with self.assertRaises(RuntimeError): runtime.move((x,))
            with self.assertRaises(RuntimeError): runtime.read(x,x)
        with patch.object(d,'_readback',side_effect=RuntimeError('D2H')):
            with self.assertRaises(RuntimeError): runtime.read(x,x)

    def test_readback_late_never_returns_candidate(self):
        runtime=d.Runtime('cpu',None,None,**self.args)
        x=torch.zeros((1,1),dtype=torch.float64,device='cpu')
        def late(*args):
            self.work.deadline=time.monotonic()-1
            return [[0.]],[[0.]]
        with patch.object(d,'_readback',side_effect=late),self.assertRaises(TimeoutError): runtime.read(x,x)

    def test_partial_nonfinite_and_wrong_dimensions(self):
        runtime=d.Runtime('cpu',None,None,**self.args)
        x=torch.zeros((1,2),dtype=torch.float64,device='cpu')
        for y,z in (([],[]),([[0.],[0.]],[[0.]]),([[0.,0.],[0.]],[[0.],[0.]]),
                    ([[math.nan],[0.]],[[0.],[0.]]),([[0.],[0.]],[[math.inf],[0.]])):
            with self.subTest(y=y,z=z),patch.object(d,'_readback',return_value=(y,z)),self.assertRaises(ValueError):
                runtime.read(x,x)

    def test_expired_before_initialization(self):
        self.work.deadline=time.monotonic()-1; self.ctx['deadline']=self.work.deadline
        with patch.object(d,'_initialize_cuda') as init,self.assertRaises(TimeoutError): self.runtime()
        init.assert_not_called()

    def test_post_readback_sync_failure(self):
        runtime=d.Runtime('cpu',None,None,**self.args)
        x=torch.zeros((1,1),dtype=torch.float64,device='cpu')
        from act.back_end.moe import batched_support_device as low
        with patch.object(low,'_synchronize',side_effect=RuntimeError('post D2H sync')):
            with self.assertRaises(RuntimeError): runtime.read(x,x)

    def test_late_optimization_failure_cannot_return_best(self):
        from act.back_end.moe import store_endpoints as ep
        torch.set_num_threads(1)
        store=SimpleNamespace(identity='1'*64,request='2'*64,factors=(0,),
                              common=SimpleNamespace(eq=(),ine=()),templates=())
        for fault in ('post_optimization','nonfinite_iteration'):
            native=ep.Native(store,self.work); tasks=[SimpleNamespace(query_id='q',slot='s')]
            original=native.runtime.sync; calls=[]
            def sync():
                calls.append(1)
                if len(calls)==4 and fault=='post_optimization': raise RuntimeError('after best')
                return original()
            with patch.object(native.runtime,'sync',side_effect=sync),\
                 patch.object(ep,'objectives',return_value=[(F(1),{0:F(1)})]),\
                 patch.object(ep,'evaluate_tile',side_effect=AssertionError('must not accept')) as accept:
                if fault=='nonfinite_iteration':
                    with patch.object(torch.sparse,'mm',return_value=torch.full((1,1),math.nan,device='cpu')):
                        with self.assertRaisesRegex(ValueError,'finite candidate'): native.propose(tasks,[native.common]*4)
                else:
                    with self.assertRaisesRegex(RuntimeError,'after best'): native.propose(tasks,[native.common]*4)
                accept.assert_not_called()

    def test_final_report_failure_keeps_only_partial_tiles(self):
        from act.back_end.moe import store_endpoints as ep
        task=SimpleNamespace(pair=(0,1),slot='x')
        plan=SimpleNamespace(contract='3'*64,tasks=[task],tiles=lambda _:iter([[task]]))
        fake=SimpleNamespace(blocks=lambda _:(),propose=lambda *_:[dict(slot='x',candidate=None)],
                             runtime=SimpleNamespace(report=lambda:(_ for _ in ()).throw(RuntimeError('report'))))
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp,patch.object(ep,'Native',return_value=fake):
            root=Path(tmp)/'proof'
            with self.assertRaisesRegex(RuntimeError,'report'): ep.produce(root,None,plan,tile_width=8,work=self.work)
            self.assertTrue((root/'00000.json').is_file())
            self.assertFalse((root/'manifest.json').exists())


def run(report):
    from scoped_proof.io import save
    class Result(unittest.TextTestResult):
        def __init__(self,*a,**k): super().__init__(*a,**k); self.outcomes=[]
        def startTest(self,t): self.state='PASS'; super().startTest(t)
        def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
        def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
        def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
        def stopTest(self,t): self.outcomes.append(dict(name=t.id(),status=self.state)); super().stopTest(t)
    from scripts.test_store_endpoints_units import Endpoints as EndpointControls
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Controls),
                             unittest.defaultTestLoader.loadTestsFromTestCase(EndpointControls)])
    expected=suite.countTestCases()
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    save(report,dict(expected=expected,tests=result.testsRun,outcomes=result.outcomes,
                     physical_cuda=False,all_pass=all(r['status']=='PASS' for r in result.outcomes)))
    return int(not(result.wasSuccessful() and result.testsRun==expected and not result.skipped))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    raise SystemExit(run(a.report))
