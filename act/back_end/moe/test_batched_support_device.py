"""Fixed CPU reference + simulated device failures. Never initialize CUDA."""
import copy
from fractions import Fraction as F
import math
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.moe import batched_support as old, batched_support_device as new
from act.back_end.moe.check_batched_support import check_batch
from act.back_end.solver.solver_hz import SparseHZono
from scoped_source.rowwise_bound import identity
from scripts.hz_batch_support_worker import fixture

UUID = "GPU-00000000-0000-0000-0000-000000000000"
CASES = ("guarded_two_sides_positive","guarded_nonpositive","equality_coupled",
         "private_binary_relaxation","constant_objective","zero_width_shared_private")


def source(case):
    if case in CASES[:4]: return fixture(case)
    hz, queries, context = fixture("guarded_two_sides_positive")
    if case == "constant_objective":
        queries=[{"id":s,"q":[0],"offset":"1/4","side":s} for s in ("min","max")]
    elif case == "zero_width_shared_private":
        # xi0=xi1=0, with two independent private binary factors still in [-1,1].
        hz=SparseHZono(c=np.array([.25,.25,.25,0.]),
            Gc=sp.csr_matrix([[1.,0.],[1.,0.],[0.,1.],[0.,0.]]),
            Gb=sp.csr_matrix([[1.,0.],[0.,1.],[0.,0.],[0.,0.]]),
            Ac=sp.csr_matrix([[0.,1.]]), Ab=sp.csr_matrix((1,2)), b=np.zeros(1),
            Auc=sp.csr_matrix([[1.,0.],[-1.,0.]]), Aub=sp.csr_matrix((2,2)),ub=np.zeros(2),frame_id=314)
        queries=[{"id":key+"-"+s,"q":q,"offset":offset,"side":s}
                 for key,q,offset in (("private-difference",[1,-1,0,0],0),("first-expert",[1,0,0,0],0),
                                      ("zero-factor",[0,0,1,0],0),("constant",[0,0,0,1],"1/4"))
                 for s in ("min","max")]
    else: raise ValueError("fixed case")
    return hz,queries,context


class DeviceCandidateControls(unittest.TestCase):
    observations={}
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(1)
        if torch.cuda.is_initialized(): raise RuntimeError("CPU control process has already initialized CUDA")
        cls.cuda_block=patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("physical CUDA forbidden"))
        cls.blocked_init=cls.cuda_block.start()

    @classmethod
    def tearDownClass(cls):
        cls.blocked_init.assert_not_called()
        cls.cuda_block.stop(); torch.set_num_threads(cls.threads)

    def setUp(self): self.end=time.monotonic()+30

    def batch(self,case=CASES[0],queries=None):
        hz,qs,ctx=source(case)
        return old.prepare_batch(hz,qs if queries is None else queries,context=ctx,deadline=self.end)

    def propose(self,batch,**kw):
        return new.propose_batch(batch,expected_batch_sha256=identity(batch),deadline=self.end,**kw)

    def checked(self,batch,candidate):
        return check_batch(batch,candidate,expected_batch_sha256=identity(batch),deadline=self.end)

    def context(self,batch):
        return {"schema":new.CONTEXT_SCHEMA,"batch_sha256":identity(batch),"gpu_uuid":UUID,
                "deadline":self.end,"allocator_limit_bytes":new.ALLOCATOR_LIMIT,"invocation":"stub-not-resource-admission"}

    def test_six_cpu_reference_cases_same_original_exact_bounds(self):
        for case in CASES:
            batch=self.batch(case); anchor=identity(batch)
            prior=old.propose_batch(batch,expected_batch_sha256=anchor,deadline=self.end)
            candidate=self.propose(batch); a=self.checked(batch,prior); b=self.checked(batch,candidate)
            self.assertEqual(a["results"],b["results"],case)
            self.assertEqual(candidate["entries"],prior["entries"],case)
            self.assertFalse(b["network_or_complete_moe_proof"])
            self.assertEqual(candidate["device"],"cpu")
            self.assertIsNone(candidate["hardware"]); self.assertIsNone(candidate["allocator_memory"])
            self.assertGreaterEqual(candidate["cost_seconds"]["other"],0)
            self.assertAlmostEqual(sum(v for k,v in candidate["cost_seconds"].items() if k!="total"),candidate["cost_seconds"]["total"])
            self.observations[case]={"batch":batch,"v1":prior,"v2":candidate,"checked_v1":a,"checked_v2":b}
        self.assertLessEqual(F(self.observations["guarded_nonpositive"]["checked_v2"]["results"][0]["bound"]),-F(1,4))
        self.assertEqual([F(r["bound"]) for r in self.observations["private_binary_relaxation"]["checked_v2"]["results"]],[-2,2])
        mixed=self.observations["zero_width_shared_private"]["checked_v2"]
        self.assertEqual(mixed["n_relaxed_binaries"],2)
        self.assertEqual([F(r["bound"]) for r in mixed["results"][:2]],[-2,2])
        self.assertEqual([F(r["bound"]) for r in mixed["results"][-2:]],[F(1,4)]*2)

    def test_objective_order_and_single_column_keep_bindings(self):
        batch=self.batch(); baseline={r["id"]:r for r in self.checked(batch,self.propose(batch))["results"]}
        queries=source(CASES[0])[1]
        for qs in ([q] for q in queries):
            b=self.batch(queries=qs); r=self.checked(b,self.propose(b))["results"][0]
            self.assertEqual(r["lp_sha256"],baseline[r["id"]]["lp_sha256"])
            self.assertLessEqual(abs(F(r["bound"])-F(baseline[r["id"]]["bound"])),F(1,10**10))
        b=self.batch(queries=list(reversed(queries)))
        self.assertEqual(self.checked(b,self.propose(b))["results"],list(reversed(list(baseline.values()))))

    def test_cpu_explicit_allocation_under_meta_default(self):
        b=self.batch()
        with torch.device("meta"): candidate=self.propose(b)
        self.assertEqual(len(self.checked(b,candidate)["results"]),4)

    def test_reject_device_context_before_gpu_initialization(self):
        b=self.batch(); ctx=self.context(b)
        with patch.object(new,"_initialize_cuda",side_effect=AssertionError("unauthorized initialization")) as init:
            for dev in ("cuda","cuda:1","mps",None):
                with self.assertRaises(ValueError): self.propose(b,device=dev)
            for supplied,digest in ((None,None),(ctx,None),(ctx,"0"*64)):
                with self.assertRaises(ValueError): self.propose(b,device="cuda:0",cuda_context=supplied,expected_context_sha256=digest)
            for key,value in (("batch_sha256","0"*64),("deadline",self.end+1),("gpu_uuid","GPU-bad"),
                              ("allocator_limit_bytes",new.ALLOCATOR_LIMIT+1),("invocation","")):
                wrong=dict(ctx); wrong[key]=value
                with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=UUID):
                    with self.assertRaises(ValueError): self.propose(b,device="cuda:0",cuda_context=wrong,expected_context_sha256=identity(wrong))
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES="0"):
                with self.assertRaises(ValueError): self.propose(b,device="cuda:0",cuda_context=ctx,expected_context_sha256=identity(ctx))
            wrong=dict(ctx,gpu_uuid="GPU-"+"0"*36)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=wrong["gpu_uuid"]):
                with self.assertRaises(ValueError): self.propose(b,device="cuda:0",cuda_context=wrong,expected_context_sha256=identity(wrong))
            with self.assertRaises(ValueError): self.propose(b,cuda_context=ctx,expected_context_sha256=identity(ctx))
            init.assert_not_called()

    def test_expired_context_and_wrong_batch_reject_before_device(self):
        b=self.batch(); ctx=self.context(b)
        with patch.object(new,"_initialize_cuda",side_effect=AssertionError("GPU touched")) as init:
            with self.assertRaises(TimeoutError):
                new.propose_batch(b,expected_batch_sha256=identity(b),deadline=time.monotonic()-1,device="cuda:0",
                                  cuda_context=ctx,expected_context_sha256=identity(ctx))
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=UUID):
                b["context"]["request"]="wrong"
                with self.assertRaises(ValueError): self.propose(b,device="cuda:0",cuda_context=ctx,expected_context_sha256=identity(ctx))
            init.assert_not_called()

    def test_initialization_and_oom_stubs_are_not_hardware_success(self):
        b=self.batch(); ctx=self.context(b); records=[]
        for error in (RuntimeError("stub initialization failure"),torch.cuda.OutOfMemoryError("stub allocator OOM")):
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=UUID),patch.object(new,"_initialize_cuda",side_effect=error) as init:
                with self.assertRaises(type(error)):
                    self.propose(b,device="cuda:0",cuda_context=ctx,expected_context_sha256=identity(ctx))
                self.assertEqual(init.call_count,1)
            records.append({"error":type(error).__name__,"injection_reached":True,"physical_cuda":False})
        self.observations["device_failures_stub"]={"records":records}

    def test_sync_exception_never_returns_old_best(self):
        b=self.batch(); original=new._synchronize; seen=[]
        def sync(device):
            seen.append(device)
            if len(seen)==3: raise RuntimeError("stub sync failure after optimization")
            original(device)
        with patch.object(new,"_synchronize",side_effect=sync):
            with self.assertRaises(RuntimeError): self.propose(b)
        self.assertEqual(seen,["cpu"]*3)
        self.observations["sync_failure_stub"]={"boundaries":seen,"injection_reached":True,"physical_cuda":False}

    def test_readback_then_expired_never_returns_candidate(self):
        b=self.batch(); original=new._readback; original_clock=new.clock; expired=[False]
        def readback(*a,**kw):
            result=original(*a,**kw); expired[0]=True; return result
        def clock(end):
            tick=original_clock(end)
            def checked():
                if expired[0]: raise TimeoutError("stub late readback")
                tick()
            return checked
        with patch.object(new,"clock",side_effect=clock),patch.object(new,"_readback",side_effect=readback):
            with self.assertRaises(TimeoutError): self.propose(b)
        self.assertTrue(expired[0])
        self.observations["late_readback_stub"]={"injection_reached":True,"physical_cuda":False}

    def test_partial_nonfinite_and_alias_pollution_reject(self):
        b=self.batch(); real=new._candidate_columns
        for returned in (([],[],None,None),([[math.nan,0]]*4,[[]]*4,None,None),([[math.inf,0]]*4,[[]]*4,None,None)):
            with patch.object(new,"_candidate_columns",return_value=returned):
                with self.assertRaises((ValueError,OverflowError)): self.propose(b)
        def pollute(*a,**kw):
            result=real(*a,**kw); b["context"]["guard"]="changed"; return result
        with patch.object(new,"_candidate_columns",side_effect=pollute):
            with self.assertRaises(ValueError): self.propose(b)

    def test_independent_checker_does_not_trust_device_label_or_claim(self):
        b=self.batch(); candidate=self.propose(b)
        for key,value in (("inequality_dual",[1,0]),("claimed_lower_bound","999"),("lp_sha256","0"*64)):
            wrong=copy.deepcopy(candidate); wrong["device"]="cuda:0"; wrong["entries"][0]["certificate"][key]=value
            with self.assertRaises(ValueError): self.checked(b,wrong)
        partial=copy.deepcopy(candidate); partial["entries"].pop()
        with self.assertRaises(ValueError): self.checked(b,partial)
        self.observations["partial_rejected"]={"batch":b,"candidates":partial,"required":4,"received":3}

    def test_overflow_and_nonfinite_iteration_never_publish(self):
        b=self.batch()
        with patch.object(torch.sparse,"mm",side_effect=lambda a,x: torch.full((a.shape[0],x.shape[1]),math.inf,device="cpu",dtype=torch.float64)):
            with self.assertRaises(ValueError): self.propose(b)
        # Caller-owned source/goal validation rejects malformed values before kernels.
        b["queries"][0]["constant"]="1/0"
        with patch.object(new,"_candidate_columns",side_effect=AssertionError("kernel touched")) as kernel:
            with self.assertRaises((ValueError,ZeroDivisionError)): self.propose(b)
            kernel.assert_not_called()

    def test_cuda_context_mutation_rejected_after_stub(self):
        b=self.batch(); ctx=self.context(b)
        def columns(*a,**kw):
            ctx["invocation"]="changed"
            return [[0.,0.]]*4,[[]]*4,None,None
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=UUID),patch.object(new,"_candidate_columns",side_effect=columns):
            with self.assertRaises(ValueError): self.propose(b,device="cuda:0",cuda_context=ctx,expected_context_sha256=identity(ctx))

    def test_archive_requires_all_cases_tests_and_sources(self):
        from scripts.run_hz_device_candidates import inventory,protocol,test_names,FILES
        summary={"test_names":test_names(),"tests":len(test_names()),"failures":0,"errors":0,"skipped":0,
                 "physical_cuda":False,"native_solves":0,"real_requests":0}
        bindings=dict.fromkeys(FILES,"separately-hash-checked")
        observations=dict.fromkeys(protocol()["normal_cases"]+["device_failures_stub","sync_failure_stub","late_readback_stub","partial_rejected"])
        inventory(summary,bindings,observations)
        for key in ("tests","test_names","bindings","observations","physical_cuda","skipped"):
            s,b,o=copy.deepcopy((summary,bindings,observations))
            if key=="tests": s[key]-=1
            elif key=="test_names": s[key].pop()
            elif key=="bindings": b.pop(next(iter(b)))
            elif key=="observations": o.pop(next(iter(o)))
            elif key=="skipped": s[key]=1
            else: s[key]=True
            with self.assertRaises(ValueError): inventory(s,b,o)

    def test_fixed_source_identity_not_just_case_labels(self):
        from scripts.run_hz_device_candidates import expected_batch
        expected={case:identity(expected_batch(case)) for case in CASES}
        self.assertEqual(len(set(expected.values())),len(CASES))
        for case in CASES: self.assertEqual(identity(self.batch(case)),expected[case])
        self.assertNotEqual(expected["guarded_two_sides_positive"],expected["guarded_nonpositive"])

    def test_nested_fault_inventory_and_execution_claims_cannot_disappear(self):
        from scripts.run_hz_device_candidates import fault_inventory,proposal_metadata
        b=self.batch(); c=self.propose(b); item={"v1":old.propose_batch(b,expected_batch_sha256=identity(b),deadline=self.end),"v2":c}
        proposal_metadata(item)
        for key,value in (("dtype","float32"),("iterations",1),("device","cuda:0"),("hard_budget_supervision",True),
                          ("cost_scope","complete MoE request"),("execution_context_sha256","0"*64)):
            wrong=copy.deepcopy(item); wrong["v2"][key]=value
            with self.assertRaises(ValueError): proposal_metadata(wrong)
        wrong=copy.deepcopy(item); wrong["v2"]["cost_seconds"]["total"]=math.nan
        with self.assertRaises(ValueError): proposal_metadata(wrong)
        partial=copy.deepcopy(c); partial["entries"].pop()
        obs={"device_failures_stub":{"records":[{"error":e,"injection_reached":True,"physical_cuda":False} for e in ("RuntimeError","OutOfMemoryError")]},
             "sync_failure_stub":{"boundaries":["cpu"]*3,"injection_reached":True,"physical_cuda":False},
             "late_readback_stub":{"injection_reached":True,"physical_cuda":False},
             "partial_rejected":{"batch":b,"candidates":partial,"required":4,"received":3}}
        fault_inventory(obs,identity(b))
        mutations=[lambda x:x["device_failures_stub"].update(records=[]),
                   lambda x:x["device_failures_stub"]["records"].pop(),
                   lambda x:x["device_failures_stub"]["records"][1].update(error="RuntimeError"),
                   lambda x:x["sync_failure_stub"]["boundaries"].pop(),
                   lambda x:x["late_readback_stub"].update(injection_reached=False),
                   lambda x:x["partial_rejected"]["batch"]["context"].update(request="wrong")]
        for mutation in mutations:
            wrong=copy.deepcopy(obs); mutation(wrong)
            with self.assertRaises(ValueError): fault_inventory(wrong,identity(b))


if __name__=="__main__": unittest.main()
