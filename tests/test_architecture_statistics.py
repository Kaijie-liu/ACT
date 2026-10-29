"""Input-block statistics and terminal audit controls; no solver invocation."""
import copy
from pathlib import Path
import time
import unittest
import numpy as np
from act.pipeline.moe.architecture_statistics import model_summary,paired,s1_decision
from architecture_s1_execution import Artifacts,audit_value,read_record,measured_invoke
from architecture_s1_audit import audit_request
from scoped_proof.io import sha,save

CONTROL_ROOT=None


def arms():
    a=[[dict(index=i,status='POSITIVE' if i<2 else 'UNKNOWN',seconds=2.,clean_correct=i<3)
        for i in range(4)] for _ in range(3)]
    b=[[dict(index=i,status='POSITIVE' if i<1 else 'TIMEOUT',seconds=4.,clean_correct=i<2)
        for i in range(4)] for _ in range(3)]
    return a,b


class Controls(unittest.TestCase):
    def test_input_block_not_call_bootstrap(self):
        a,b=arms();result=paired(a,b,count=4,resamples=200,seed=29)
        indices=np.random.default_rng(29).integers(0,4,size=(200,4))
        manual=np.array([0.,1.,0.,0.])[indices].mean(axis=1)
        np.testing.assert_equal(result['intervals']['rate_difference'],np.quantile(manual,[.025,.975]))
        self.assertEqual((result['gained'],result['lost'],result['positive_intersection']),(3,0,3))
        self.assertEqual(result['rate_difference'],.25)
        self.assertEqual(result['median_seconds']['ratio'],.5)
        self.assertEqual(result['shared_clean_correct']['pairs'],6)
        self.assertEqual(result['decision'],'ENTER_S2')

    def test_fixed_point_gate_and_zero_certificate_disclosure(self):
        a,b=arms()
        for group in (a,b):
            for rows in group:
                for row in rows:row['status']='UNKNOWN'
        result=paired(a,b,count=4,resamples=100,seed=29)
        self.assertTrue(result['both_certified_rates_zero'])
        self.assertEqual(result['decision'],'ENTER_S2')  # Literal registered screen, not a C2 claim.
        for rows in b:
            for row in rows:row['seconds']=2.
        self.assertEqual(paired(a,b,count=4,resamples=100)['decision'],'STOP_S1_MATCH')

    def test_losses_fail_and_unexecuted_are_not_dropped(self):
        a,b=arms()
        reverse=paired(b,a,count=4,resamples=100)
        self.assertEqual(reverse['decision'],'STOP_S1_MATCH')
        a[1][3].update(status='NOT_EXECUTED',seconds=None)
        missing=paired(a,b,count=4,resamples=100)
        self.assertEqual(missing['denominator'],12)
        self.assertEqual(missing['not_executed_arm_items'],1)
        self.assertTrue(missing['decision'].startswith('PENDING'))
        self.assertIsNone(missing['intervals'])
        with self.assertRaises(ValueError):
            paired([a[0][:-1],a[1],a[2]],b,count=4,resamples=100)

    def test_route_unknown_and_cost_denominators(self):
        rows=arms()[0][0]
        rows[0]['route_candidate_upper']=1;rows[1]['route_candidate_upper']=3
        rows[2].update(status='ERROR',seconds=.1)
        rows[3].update(status='TIMEOUT',seconds=299.)
        result=model_summary(rows,4)
        self.assertEqual(result['states']['ERROR'],1)
        self.assertEqual(result['states']['TIMEOUT'],1)
        self.assertEqual(result['route_unknown'],2)
        self.assertEqual(result['numeric_single_candidate_fraction_of_all'],.25)
        self.assertEqual(result['all_request_seconds'],303.1)

    def test_primary_completion_gate(self):
        good=dict(decision='ENTER_S2');bad=dict(decision='STOP_S1_MATCH')
        self.assertEqual(s1_decision(good,bad,all_queue_items_settled=False),'PENDING_S1_COMPLETION')
        self.assertEqual(s1_decision(good,bad,all_queue_items_settled=True),'ENTER_S2')
        self.assertEqual(s1_decision(bad,bad,all_queue_items_settled=True),'STOP_S1_RETURN_TO_REDUCTION_PAPER')

    def test_prelaunch_error_is_charged(self):
        result=measured_invoke(CONTROL_ROOT/'absent_parent',dict(id='failed'),CONTROL_ROOT/'no_manifest.json',
            seconds=300,timing_reserve=(12,10,5))
        self.assertEqual(result['status'],'ERROR');self.assertFalse(result['accepted'])
        self.assertGreaterEqual(result['seconds'],0)
        self.assertLess(result['seconds'],1)
        self.assertEqual(result['controller_error']['kind'],'FileNotFoundError')

    def _record(self,name,*,overrun=False,missing_property=False):
        root=CONTROL_ROOT/name;store=Artifacts.create(root)
        query=dict(id='job',mode='complete',model='dense',index=0,radius=2)
        folder=root/'job';folder.mkdir();start=time.monotonic()
        request=dict(**query,started_at=start,hard_deadline=start+300,worker_deadline=start+288,
            manifest_sha256='manifest')
        store.write(folder/'request.json.gz',request)
        result=dict(status='POSITIVE',expected_histories=1,attempted_histories=1,returned_histories=1,
            verified_histories=1,pruned_histories=0,attempted_prefix_queries=0,prefix_queries=0,
            prefix_bounds=[],exclusions=[],histories=[dict(history=[],outcome=dict(status='verified',
                actual_raw_spec=[[[[1.]], [0.]]]))],complete=True,formal_SAFE=False,
            evidence_grade='ALPHA_BETA_CROWN_NUMERICAL_FILTER',
            evaluation=dict(mode='complete',family='dense',source_complete_proof=False))
        cov=audit_value(result,family='dense',experts=[],properties=1)
        if missing_property:result['histories'][0]['outcome']['actual_raw_spec']=[]
        value=dict(result=result,coverage=cov,family='dense',experts=[],properties=1,clean_correct=True)
        h=store.write(folder/'outcome.json.gz',dict(request_sha256=sha(folder/'request.json.gz'),
            clock=start+1,value=value))
        store.write(folder/'published.json.gz',dict(sha256=h,clock=start+1.1))
        store.write(folder/'session.json.gz',dict(status='COMPLETED',returncode=0,execution_error=None,
            cleanup_error=None,cleanup_status='LEADER_REAPED_NO_LIVE_SESSION'))
        terminal=dict(id='job',started_at=start,hard_deadline=start+300,received_at=start+2,
            status='POSITIVE',accepted=True,clean_correct=True,coverage=cov,
            files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        store.write(folder/'terminal.json.gz',terminal)
        store.write(root/'job.ledger.json.gz',dict(id='job',status='TIMEOUT' if overrun else 'POSITIVE',
            accepted=not overrun,seconds=301 if overrun else 2.1,
            terminal_sha256=sha(folder/'terminal.json.gz'),terminal_overrun=overrun))
        return root,query

    def test_full_terminal_and_no_late_upgrade(self):
        root,q=self._record('normal')
        row,value=audit_request(root,q,'manifest')
        self.assertEqual(row['status'],'POSITIVE');self.assertIsNotNone(value)
        root,q=self._record('late',overrun=True)
        row,value=audit_request(root,q,'manifest')
        self.assertEqual(row['status'],'TIMEOUT');self.assertIsNone(value)

    def test_missing_property_and_wrong_binding_rejected(self):
        root,q=self._record('missing',missing_property=True)
        with self.assertRaises(ValueError):audit_request(root,q,'manifest')
        root,q=self._record('identity')
        with self.assertRaises(ValueError):audit_request(root,dict(q,radius=8),'manifest')
        with self.assertRaises(ValueError):audit_request(root,q,'other_manifest')
