"""S2 secondary comparisons and shared terminal audit; no trained examples."""
import copy
from pathlib import Path
import sys
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
from act.pipeline.moe.architecture_s2_statistics import crossed_medians,verification_summary
from architecture_s1_execution import Artifacts
from architecture_s1_audit import audit_request
from architecture_s2_audit import check_secondary,check_primary,analyze
from act.pipeline.moe.architecture_s2_evaluation import pack_arrays
from scoped_proof.io import sha

CONTROL_ROOT=None


class StatisticsControls(unittest.TestCase):
    def test_crossed_median_matches_manual_common_draws(self):
        values=np.arange(24,dtype=float).reshape(4,2,3)+1
        actual=crossed_medians(values,resamples=31,seed=17,chunk=7)
        sr,ir=(np.random.default_rng(s) for s in np.random.SeedSequence(17).spawn(2))
        expected=[]
        for _ in range(31):
            s=sr.integers(0,2,size=2);i=ir.integers(0,3,size=3)
            expected.append(np.median(values[:,s][:,:,i],axis=(1,2)))
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual,crossed_medians(values,resamples=31,seed=17,chunk=1))

    def test_positive_solved_and_all_costs_are_distinct(self):
        states=np.full((4,5,4),'UNKNOWN',dtype='<U20');costs=np.ones(states.shape)
        states[:,:,0]='POSITIVE'
        states[2,:,1]='POSITIVE';states[2,:,2]='UNSAFE_REPLAYED'
        states[0,:,1]='TIMEOUT';costs[0,:,1]=299
        states[2,:,3]='ERROR';costs[2,:,3]=.1
        value=verification_summary(states,costs,expected_inputs=4,resamples=200)
        p=value['comparisons']['M_minus_D-act']
        self.assertEqual((p['gained'],p['lost'],p['positive_intersection']),(5,0,5))
        self.assertEqual(p['rate_difference'],.25);self.assertEqual(p['solved_difference'],.5)
        self.assertEqual(value['arms']['D-act']['summed_request_seconds'],1510)
        self.assertEqual(value['arms']['M']['states']['ERROR'],5)
        self.assertEqual(p['all_request_cost']['median_m'],1)
        self.assertEqual(p['all_request_cost']['median_comparator'],1)
        # Slow failures must affect mean cost even if medians happen to match.
        self.assertLess(p['all_request_cost']['mean_difference'],-70)
        from architecture_s1_execution import encoded
        self.assertGreater(len(encoded(value)),0)  # No NumPy scalar/NaN serialization failure.

    def test_missing_arm_is_not_a_gain_or_interval(self):
        states=np.full((4,5,4),'UNKNOWN',dtype='<U20');costs=np.ones(states.shape)
        states[2,:,1]='POSITIVE';states[0,0,1]='NOT_EXECUTED';costs[0,0,1]=np.nan
        value=verification_summary(states,costs,expected_inputs=4,resamples=100)
        p=value['comparisons']['M_minus_D-act']
        self.assertFalse(value['complete']);self.assertEqual(p['gained'],4)
        self.assertEqual(p['denominator'],20);self.assertEqual(p['paired_executed'],19)
        self.assertIsNone(p['rate_difference_interval95']);self.assertIsNone(p['all_request_cost'])
        costs[0,0,1]=0
        with self.assertRaises(ValueError):verification_summary(states,costs,expected_inputs=4)

    def test_timeouts_not_zero_cost_or_unsafe(self):
        states=np.full((4,5,4),'TIMEOUT',dtype='<U20');costs=np.full(states.shape,299.)
        value=verification_summary(states,costs,expected_inputs=4,resamples=100)
        self.assertEqual(value['arms']['M']['solved_fraction'],0)
        self.assertEqual(value['arms']['M']['median_all_request_seconds'],299)
        costs[0,0,0]=np.nan
        incomplete=verification_summary(states,costs,expected_inputs=4,resamples=100)
        self.assertFalse(incomplete['complete_cost'])
        self.assertEqual(incomplete['arms']['D-act']['missing_executed_cost'],1)


class IdentityControls(unittest.TestCase):
    def test_full_roster_prefix_does_not_create_success_or_zero_time_calls(self):
        from architecture_s2_study import ARMS,all_requests
        from architecture_s1_execution import digest,read_record
        root=CONTROL_ROOT/'full_prefix';store=Artifacts.create(root)
        pixels=dict(pixels=np.zeros((10000,3,32,32),dtype=np.uint8),labels=np.zeros(10000,dtype=np.int64))
        input_sha=store.write(root/'inputs.json.gz',pack_arrays(pixels))
        models=[dict(id=f'{a}_seed{s}',arm=a,seed=s,family='moe' if a in ('M','M-reg') else 'dense')
                for a in ARMS for s in range(5)]
        queries=all_requests(models)
        spec=dict(study='S2A',models=models,source_files={},inputs=dict(path=str(root/'inputs.json.gz'),sha256=input_sha),
            request_catalog_sha256=digest(queries),bootstrap=dict(resamples=10000,seed=20261031),
            artifact_bytes=128*2**20,preparation_seconds=1.)
        mh=store.write(root/'manifest.json',spec)
        store.write(root/'start.json.gz',dict(manifest_sha256=mh))
        store.write(root/'queue_complete.json.gz',dict(status='PREFIX_STOP',source_manifest_sha256=mh,
            request_catalog_sha256=digest(queries),settled=0,not_executed=24020,
            states={s:0 for s in ('POSITIVE','UNSAFE_REPLAYED','UNKNOWN','TIMEOUT','ERROR','AUXILIARY_COMPLETE')},
            summed_request_seconds=0.,missing_request_cost=0,queue_wall_seconds=.1))
        value=analyze(root)
        self.assertEqual(value['status'],'PREFIX_AUDIT')
        self.assertEqual(value['registered_decision'],'PENDING_COMPLETE_S2A_EVIDENCE')
        self.assertEqual(value['online_states'],{'NOT_EXECUTED':24020})
        self.assertEqual(value['by_mode']['complete']['states']['NOT_EXECUTED'],4000)
        self.assertEqual(value['by_mode']['incomplete']['states']['NOT_EXECUTED'],20000)
        self.assertIsNone(value['primary'])
        self.assertIsNone(value['secondary']['complete']['comparisons']['M_minus_D-act']['all_request_cost'])
        store.write(root/'test_summary.json.gz',value)
        self.assertEqual(read_record(root/'test_summary.json.gz')['registered_decision'],value['registered_decision'])

    def test_same_engine_properties_labels_and_heuristics(self):
        spec=dict(backend_sources_sha256='engine',complete_configuration={'bab.branching.method':'nonlinear'})
        model=dict(family='dense');query=dict(mode='complete')
        value=dict(backend_sources_sha256='engine',family='dense',properties=9,label=3,
            clean_correct=True,clean_prediction=3,
            result=dict(status='POSITIVE',evaluation=dict(family='dense',
                explicit_complete_configuration=spec['complete_configuration']),
                histories=[dict(outcome=dict(branching='nonlinear'))]))
        check_secondary(value,query,model,spec,3)
        for key,wrong in [('backend_sources_sha256','other'),('properties',8),('label',4),('family','moe')]:
            changed=copy.deepcopy(value);changed[key]=wrong
            with self.assertRaises(ValueError):check_secondary(changed,query,model,spec,3)
        changed=copy.deepcopy(value);changed['result']['histories'][0]['outcome']['branching']='kfsb'
        with self.assertRaises(ValueError):check_secondary(changed,query,model,spec,3)
        incomplete=copy.deepcopy(value);incomplete['result']['histories'][0]['outcome']=dict(
            bound_propagation='CROWN-IBP',complete_verifier=None)
        check_secondary(incomplete,dict(mode='incomplete'),model,spec,3)
        incomplete['result']['histories'][0]['outcome']['complete_verifier']='bab'
        with self.assertRaises(ValueError):check_secondary(incomplete,dict(mode='incomplete'),model,spec,3)

    def test_primary_transport_and_no_late_promotion(self):
        root=CONTROL_ROOT/'transport';store=Artifacts.create(root)
        q=dict(id='primary',mode='primary',model='M_seed0',radius=2)
        folder=root/q['id'];folder.mkdir();now=time.monotonic()
        request=dict(q,manifest_sha256='manifest',started_at=now,hard_deadline=now+300,worker_deadline=now+288)
        store.write(folder/'request.json.gz',request)
        ah=store.write(folder/'arrays.json.gz',{'synthetic_transport_only':True})
        result=dict(status='AUXILIARY_COMPLETE',evidence_grade='IBP_NUMERICAL_POLICY',formal_SAFE=False,
                    evaluation=dict(mode='primary',family='moe',source_complete_proof=False))
        value=dict(result=result,coverage=None,clean_correct=None,properties=9,array_sha256=ah)
        oh=store.write(folder/'outcome.json.gz',dict(request_sha256=sha(folder/'request.json.gz'),clock=now+1,value=value))
        store.write(folder/'published.json.gz',dict(sha256=oh,clock=now+1.1))
        store.write(folder/'session.json.gz',dict(status='COMPLETED',returncode=0,execution_error=None,
            cleanup_error=None,cleanup_status='LEADER_REAPED_NO_LIVE_SESSION'))
        terminal=dict(id=q['id'],started_at=now,hard_deadline=now+300,received_at=now+2,status='AUXILIARY_COMPLETE',
            accepted=True,clean_correct=None,coverage=None,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        th=store.write(folder/'terminal.json.gz',terminal)
        store.write(root/'primary.ledger.json.gz',dict(id=q['id'],status='AUXILIARY_COMPLETE',accepted=True,
            seconds=2.1,terminal_sha256=th))
        row,actual=audit_request(root,q,'manifest')
        self.assertEqual(row['status'],'AUXILIARY_COMPLETE');self.assertEqual(actual,value)
        with self.assertRaises(ValueError):audit_request(root,dict(q,model='M_seed1'),'manifest')
        # Separate fixture, original accepted record untouched.
        import shutil
        late=CONTROL_ROOT/'late';shutil.copytree(root,late)
        path=late/'primary.ledger.json.gz'
        import gzip,json
        ledger=dict(id=q['id'],status='TIMEOUT',accepted=False,seconds=301,terminal_sha256=th,terminal_overrun=True)
        path.write_bytes(gzip.compress(json.dumps(ledger).encode(),mtime=0))
        row,actual=audit_request(late,q,'manifest')
        self.assertEqual(row['status'],'TIMEOUT');self.assertIsNone(actual)

    def test_primary_arrays_and_missing_route_metadata(self):
        root=CONTROL_ROOT/'primary_arrays';store=Artifacts.create(root);folder=root/'primary';folder.mkdir()
        n=10000;labels=np.arange(n,dtype=np.int64)%10
        router=np.zeros((n,8),dtype=np.float32);router[:,:2]=1
        mask=np.zeros((n,4),dtype=np.uint8);mask[:,0]=1
        arrays=dict(label=labels,prediction=labels,clean=np.ones(n,dtype=bool),certified=np.ones(n,dtype=bool),
            lower_margin=np.ones((n,9),dtype=np.float32),upper_margin=np.ones((n,9),dtype=np.float32),
            router_lower=router,router_upper=router,pair_mask=mask,route_count=np.ones(n,dtype=np.uint8))
        ah=store.write(folder/'arrays.json.gz',pack_arrays(arrays))
        model=dict(family='moe',checkpoint_sha256='weights')
        spec=dict(inputs=dict(sha256='inputs'),primary_torch_version='synthetic',primary_batch_size=5000,
                  source_files={'act/pipeline/moe/route_envelope_ibp.py':'bounds'})
        value=dict(family='moe',checkpoint_sha256='weights',input_sha256='inputs',count=n,torch='synthetic',
            bound_source_sha256='bounds',array_sha256=ah,clean=n,certified=n,single_route=n,
            batches=[dict(start=i,stop=i+5000,count=5000,clean=5000,certified=5000,seconds=1) for i in (0,5000)])
        actual=check_primary(root,dict(id='primary'),value,model,spec,labels)
        np.testing.assert_array_equal(actual['certified'],arrays['certified'])
        changed=copy.deepcopy(value);changed['single_route']=9999
        with self.assertRaises(ValueError):check_primary(root,dict(id='primary'),changed,model,spec,labels)
        changed=copy.deepcopy(value);changed['batches']=changed['batches'][:1]
        with self.assertRaises(ValueError):check_primary(root,dict(id='primary'),changed,model,spec,labels)
        with self.assertRaises(ValueError):check_primary(root,dict(id='primary'),value,model,spec,labels[::-1])
