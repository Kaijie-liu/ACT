import copy
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.test_conv_sign_lp import SignEvidenceTests
from scripts.check_conv_request_sign_lp import aggregate,expected_scope,property_vector,TRUSTED
from scripts.conv_request_sign_lp_contract import publication_gate


def fixture(center=1.):
    rec,cert,_,_=SignEvidenceTests().example(center)
    rec['source']['frame_id']='frame'
    from act.back_end.solver.lp_certificate import identity
    rec['source_sha256']=identity(rec['source'])
    job={'parent_request_sha256':'parent','case':{'expected_pairs':[[0,3]]},
         'parent_request':{'sample':{'dataset_index':16,'label':5,'center':'x','lower':'l','upper':'u'},
                           'epsilon':2/255,'subject':{'model_state':'model'}}}
    req=expected_scope(job);prop={'classes':10,'clean_prediction':5,'kind':'TOP1_ROBUST'}
    scope={'request_id':'request','model_state':'model','property':prop,'frame_id':'frame',
           'lower':'l','upper':'u','gate':'selected_softmax_top2','tie_policy':'ANY_LEGAL_TOPK',
           'numerical_policy':{'safe_positive_margin':1e-7}}
    snapshot={'request_id':'request','identity':{'model_state':'model','center':'x','lower':'l','upper':'u','property':prop},
              'scope':scope,'feasible_route_sets':[[0,3]],'route_sets_exact':True,'branches':[]}
    rows=[{'pair':[0,3],'property_index':i,'competitor':c,'q':property_vector(5,c),'constant':0,
           'kind':'lp','proposal_status':'PROPOSED','export':'export','certificate':'cert',
           'source_sha256':rec['source_sha256']} for i,c in enumerate(j for j in range(10) if j!=5)]
    manifest={'schema':'CONV_REQUEST_SIGN_LP_R1_MANIFEST','request':req,'trusted_base':TRUSTED,
              'positive_threshold':1e-7,'generation_complete':True,'obligations':rows,
              'routes':{'feasible':[[0,3]],'infeasible':[list(p) for p in itertools.combinations(range(4),2) if p!=(0,3)],
                        'unresolved':[],'exact':True}}
    return manifest,snapshot,job,{'export':rec,'cert':cert}


class RequestEvidenceTests(unittest.TestCase):
    def check(self,args):return aggregate(*args[:3],args[3].__getitem__)

    def test_complete_and_no_solver(self):
        with patch('scipy.optimize.linprog',side_effect=AssertionError('solver called')):
            r=self.check(fixture())
        self.assertEqual(r['positive_obligations'],9)
        self.assertEqual(r['status'],'CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_F0_LOWERING')
        self.assertFalse(r['production_SAFE_verdict_changed']);self.assertFalse(r['deployed_float_SAFE'])

    def test_nonpositive_is_unknown(self):
        r=self.check(fixture(0.));self.assertEqual(r['status'],'UNKNOWN');self.assertEqual(r['positive_obligations'],0)

    def test_missing_duplicate_wrong_property(self):
        for change in ('missing','duplicate','competitor','q','constant'):
            a=fixture();rows=a[0]['obligations']
            if change=='missing':rows.pop()
            elif change=='duplicate':rows[-1]=copy.deepcopy(rows[0])
            elif change=='competitor':rows[0]['competitor']=2
            elif change=='q':rows[0]['q'][5]=0
            else:rows[0]['constant']=1
            with self.assertRaises(ValueError):self.check(a)

    def test_route_coverage(self):
        a=fixture();a[0]['routes']['infeasible'].pop()
        with self.assertRaises(ValueError):self.check(a)
        a=fixture();a[0]['routes']['unresolved'].append(a[0]['routes']['infeasible'].pop())
        a[0]['routes']['exact']=False
        self.assertEqual(self.check(a)['reason'],'INCOMPLETE_ROUTE_COVERAGE')

    def test_generation_and_proposal_incomplete(self):
        for status in ('PENDING','UNAVAILABLE'):
            a=fixture();a[0]['obligations'][0]['proposal_status']=status
            self.assertEqual(self.check(a)['positive_obligations'],8)
            self.assertEqual(self.check(a)['status'],'UNKNOWN')
        a=fixture();a[0]['generation_complete']=False
        self.assertEqual(self.check(a)['status'],'UNKNOWN')

    def test_frame_certificate_and_request(self):
        for change in ('frame','claim','request'):
            a=fixture()
            if change=='frame':a[1]['scope']['frame_id']='different'
            elif change=='claim':a[3]['cert']['claimed_lower_bound']='100'
            else:a[0]['request']['dataset_index']=98
            with self.assertRaises(ValueError):self.check(a)

    def test_reuse_reconstructed_and_mutations(self):
        from act.pipeline.moe.scoped_f0_proofs import facts_from_branches,reuse_property
        a=fixture();lo=[0.]*10;hi=[1.]*10;lo[5]=3.;hi[5]=4.
        a[1]['branches']=[{'candidate':e,'proof_output_bounds':{'lower':lo,'upper':hi}} for e in (0,3)]
        facts=facts_from_branches(a[1]['branches'],a[1]['scope'])
        for i,row in enumerate(a[0]['obligations']):
            row.update(kind='reused',proof=reuse_property(facts,a[1]['scope'],(0,3),i))
        self.assertEqual(self.check(a)['reused_positive'],9)
        for change in ('one','expert','scope','bound','source'):
            b=copy.deepcopy(a);proof=b[0]['obligations'][0]['proof'];f=proof['proof_sources'][0]
            if change=='one':proof['proof_sources'].pop()
            elif change=='expert':f['expert']=2
            elif change=='scope':f['scope']=dict(f['scope'],frame_id='wrong')
            elif change=='bound':f['lower_bound']=100.
            else:f['source_interval']={'lower':[0]*10,'upper':[1]*10}
            with self.assertRaises(ValueError):self.check(b)

    def test_stdlib_only_complete_check(self):
        code=('import json,sys;from scripts.check_conv_sign_lp import isolate;isolate();'
              'from scripts.check_conv_request_sign_lp import aggregate; a=json.load(sys.stdin);'
              'r=aggregate(*a[:3],a[3].__getitem__);'
              'assert not any(x in sys.modules for x in ("torch","numpy","scipy"));print(json.dumps(r))')
        p=subprocess.run([sys.executable,'-S','-c',code],input=json.dumps(fixture()),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(p.stdout)['positive_obligations'],9)

    def test_remote_gate(self):
        ref='refs/heads/feat/moe-route-verification'
        with patch('scripts.conv_request_sign_lp_contract.git',side_effect=['abc','abc\t'+ref]):
            self.assertEqual(publication_gate()['remote_head'],'abc')
        with patch('scripts.conv_request_sign_lp_contract.git',side_effect=['abc','def\t'+ref]):
            with self.assertRaises(ValueError):publication_gate()
        with patch('scripts.conv_request_sign_lp_contract.git',side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):publication_gate()

    def test_capture_visits_all_properties_without_weighted_solver(self):
        from contextlib import ExitStack
        from act.back_end.moe.hz_routing import TopKSetReport
        from scripts import run_conv_request_sign_lp as run
        import act.pipeline.moe.paired_monolithic as mono
        a=fixture();request=a[2]['parent_request'];request['config']={'path':'config'}
        request['subject'].update(checkpoint='cp',checkpoint_sha256='cp-hash')
        job={**a[2],'competitors':[i for i in range(10) if i!=5],'protocol':{'positive_threshold':1e-7}}
        config={'support':{'lp_neurons':0,'milp_neurons':0,'lp_time_limit':1,'milp_time_limit':1},
                'solver':{'margin_support_seconds':1,'difference_support_seconds':1}}
        pairs=((0,3),);routes=TopKSetReport(pairs,tuple(p for p in itertools.combinations(range(4),2) if p not in pairs),(),(),True)
        internal={'route_sets':routes,'program':SimpleNamespace(output_width=10,experts=list(range(4))),
                  'router':SimpleNamespace(output_hz='router',input_hz='input'),'output_spec':'spec'}
        budget=SimpleNamespace(check=lambda *a:None,limit=lambda *a:1)
        original=mono._run_monolithic
        with tempfile.TemporaryDirectory(dir=run.ROOT/'data/moe/results') as tmp,ExitStack() as stack:
            directory=Path(tmp);run.save(directory/'job.json',job)
            def fake_verify(*args,**kw):
                run.save(kw['journal_path'],[]);kw['common_fact_callback']({'payload':a[1]})
                return mono._run_monolithic(model=None,center=None,clean_prediction=5,internal=internal,config=config,budget=budget)
            stack.enter_context(patch.object(run,'validate_job',return_value=job))
            stack.enter_context(patch.object(run,'read',return_value=config))
            stack.enter_context(patch('act.pipeline.moe.external_pair_worker.load',return_value=(None,{'center':None})))
            stack.enter_context(patch('scripts.budget_contract_v2.verify_v2',side_effect=fake_verify))
            stack.enter_context(patch.object(mono,'linear_safety_rows',return_value=[(property_vector(5,i),0) for i in job['competitors']]))
            for name in ('condition_topk_set','guarded_input_topk_set'):
                stack.enter_context(patch.object(mono,name,return_value=SimpleNamespace(hz='hz')))
            stack.enter_context(patch.object(mono,'shared_input_pair_propagation',return_value=SimpleNamespace(joint='joint')))
            stack.enter_context(patch.object(mono,'compute_weighted_top2_gate_range',return_value='gates'))
            build=stack.enter_context(patch.object(mono,'build_weighted_top2_f0',return_value=SimpleNamespace(output_hz='out')))
            stack.enter_context(patch('act.back_end.solver.hz_lp_export.export',return_value=a[3]['export']))
            solver=stack.enter_context(patch.object(mono,'solve_monolithic_weighted_top2_f0',side_effect=AssertionError('property solver called')))
            run.capture_worker(directory)
            m=json.loads((directory/'generation.json').read_text())
            self.assertTrue(m['generation_complete']);self.assertEqual(len(m['obligations']),9)
            self.assertEqual(build.call_count,9);solver.assert_not_called()
        self.assertIs(mono._run_monolithic,original)


if __name__=='__main__':unittest.main()
