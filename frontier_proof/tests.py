"""Synthetic real-intake integration, fail-closed reception and total cost controls."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from frontier_proof import supervisor
from frontier_proof.audit import audit
from frontier_proof.contract import (accepted_routes, output_inputs, phases, policy,
                                     route_inputs, SCHEMA)
from scoped_proof.evidence import POSITIVE
from scoped_proof.io import load, save, sha, PYTHON, ROOT
from scoped_proof.tests import make_fixture
from source_enclosure.format import identity


def fixture(root, mode='checked_frontier', prunable=True, bias=2.):
    spec, _, _, _ = make_fixture(root, bias=bias)
    if prunable:
        import torch
        from act.back_end.moe.factory import OutputMoEFactoryConfig, build_output_moe
        from scoped_source.capture import capture
        # Use the final pair so that retained global property indices have gaps.
        raw = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=False)
        key = next(k for k in raw['state_dict'] if k.startswith('router.') and k.endswith('bias'))
        raw['state_dict'][key] = torch.tensor([0., 2., 3.], dtype=torch.float64)
        torch.save(raw, root/'checkpoint.pt')
        spec['checkpoint']['sha256'] = sha(root/'checkpoint.pt')
        model = build_output_moe(OutputMoEFactoryConfig(**raw['factory_config'])).double().eval()
        model.load_state_dict(raw['state_dict'], strict=True)
        center = torch.zeros((1, 2), dtype=torch.float64)
        doc = capture(model, center, label=2, radius='1/8', margin='1/100', clip=['-1','1'], deadline=time.monotonic()+30)
        spec['scope'] = {k:doc['request'][k] for k in spec['scope']}
    spec['proof_policy'] = {'schema':SCHEMA,'mode':mode,'router_proposal':'FINAL_AFFINE_DUAL_ONLY',
                            'parse_cache':False,'checker_cache':False}
    return spec


def mutate(path, value):
    """Only fresh synthetic copies are deliberately corrupted."""
    path.unlink()
    save(path, value)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        cls.root = Path(cls.temp.name)
        assets = cls.root/'assets'; assets.mkdir()
        cls.spec = fixture(assets)
        cls.good = cls.root/'good'
        cls.outcome = supervisor.supervise(cls.good, cls.spec, budget=30)
        if cls.outcome['status'] != POSITIVE:
            raise AssertionError('\n'.join(p.read_text() for p in cls.good.glob('*.log')))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self):
        self.work = self.root/self._testMethodName; self.work.mkdir()

    def copy_good(self):
        target = self.work/'copy'; shutil.copytree(self.good, target); return target

    def test_complete_prunable_same_object_both_arms(self):
        spec = copy.deepcopy(self.spec); spec['proof_policy']['mode'] = 'exhaustive'
        other = self.work/'exhaustive'
        # Empty guarded LPs have no dual proposal on the unchanged exhaustive
        # path: it does not manufacture an infeasibility certificate.
        self.assertEqual(supervisor.supervise(other, spec, budget=30)['status'], 'NOT_CLOSED')
        self.assertEqual(load(other/'source.json'), load(self.good/'source.json'))
        old = load(other/'construction.json'); new = load(self.good/'construction.json')
        self.assertEqual(new['pairs'], [p for p in old['pairs'] if p['pair']==[1,2]])
        self.assertEqual(audit(other)['checked_bounds'], 2)
        review = audit(self.good)
        self.assertEqual((review['required'],review['discharged_by_exclusion'],review['checked_bounds']), (6,4,2))
        self.assertEqual(sorted(p.name for p in (self.good/'candidates').glob('*.json')), ['0004.json','0005.json'])

    def test_ties_keep_every_pair_and_property(self):
        spec=fixture(self.work,prunable=False); folder=self.work/'ties'
        self.assertEqual(supervisor.supervise(folder,spec,budget=30)['status'],POSITIVE)
        report=audit(folder)
        self.assertEqual((report['checked_excluded_pairs'],report['checked_bounds']), (0,6))
        spec=copy.deepcopy(spec);spec['proof_policy']['mode']='exhaustive'
        other=self.work/'exhaustive_ties'
        self.assertEqual(supervisor.supervise(other,spec,budget=30)['status'],POSITIVE)
        self.assertEqual(audit(other)['checked_bounds'],6)

    def test_nonpositive_never_becomes_unsafe_or_complete_proof(self):
        spec=fixture(self.work,bias=-1.);folder=self.work/'negative'
        self.assertEqual(supervisor.supervise(folder,spec,budget=30)['status'],'NOT_CLOSED')
        self.assertFalse(audit(folder)['complete_output_positive_proof'])
        self.assertEqual(len(load(folder/'evidence_check.json')['nonpositive']),2)

    def test_fresh_solver_free_audit_and_no_overwrite(self):
        code=("import sys,json; from frontier_proof.audit import audit; "
              "r=audit(sys.argv[1]); assert not any(n.split('.')[0] in ('torch','numpy','scipy','highspy','act') for n in sys.modules); "
              "assert 'checked_route_frontier.build' not in sys.modules; print(json.dumps(r))")
        run=subprocess.run([PYTHON,'-S','-c',code,str(self.good)],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertTrue(json.loads(run.stdout)['complete_output_positive_proof'])
        with self.assertRaises(FileExistsError):supervisor.supervise(self.good,self.spec,budget=30)

    def test_route_candidate_identity_and_complete_inventory(self):
        for key in ('invocation','source_sha256','prefix_sha256','higher','lower','index'):
            folder=self.work/key;shutil.copytree(self.good,folder)
            path=folder/'route_candidates/0000.json';entry=load(path);entry['context'][key]='wrong';mutate(path,entry)
            with self.assertRaises(ValueError):route_inputs(folder,time.monotonic()+10)
        folder=self.copy_good();(folder/'route_candidates/0000.json').unlink()
        with self.assertRaises(ValueError):route_inputs(folder,time.monotonic()+10)

    def test_route_check_source_and_property_binding(self):
        folder=self.copy_good();receipt=load(folder/'route_check.json')
        receipt['context']['request_sha256']='wrong';mutate(folder/'route_check.json',receipt)
        with self.assertRaises(ValueError):accepted_routes(folder,time.monotonic()+10)
        folder=self.work/'other';shutil.copytree(self.good,folder)
        doc=load(folder/'source.json');doc['request']['label']=1;mutate(folder/'source.json',doc)
        with self.assertRaises(ValueError):route_inputs(folder,time.monotonic()+10)

    def test_missing_or_excluded_output_cannot_close(self):
        folder=self.copy_good();(folder/'candidates/0005.json').unlink()
        with self.assertRaises(ValueError):audit(folder)
        folder=self.work/'extra';shutil.copytree(self.good,folder)
        save(folder/'candidates/0000.json',load(folder/'candidates/0004.json'))
        with self.assertRaises(ValueError):audit(folder)

    def test_no_route_completion_is_partial_not_executable(self):
        folder=self.copy_good();(folder/'route_complete.json').unlink()
        with self.assertRaises(ValueError):accepted_routes(folder,time.monotonic()+10)
        _,_,rows,context=route_inputs(folder,time.monotonic()+10,require_complete=False)
        self.assertEqual(len(rows),6);self.assertIsNone(context['completion'])

    def test_partial_route_then_exception_stops_before_construction(self):
        def dispatch(phase,folder,deadline):
            if phase!='route_propose':return supervisor.command(phase,folder,deadline,self.spec)
            code=("from pathlib import Path; import frontier_proof.worker as w; original=w.save\n"
                  "def stop(path,obj):\n if Path(path).name=='0001.json':raise RuntimeError('partial router publication')\n return original(path,obj)\n"
                  f"w.save=stop;w.work('route_propose',Path({str(folder)!r}),{deadline!r})")
            return [PYTHON,'-S','-c',code]
        folder=self.work/'partial';out=supervisor.supervise(folder,self.spec,budget=30,command_factory=dispatch)
        self.assertEqual(out['status'],'ERROR');self.assertFalse((folder/'construction.json').exists())
        report=audit(folder);self.assertEqual(report['route_bounds_received'],1)
        self.assertFalse(report['complete_output_positive_proof'])

    def test_partial_output_then_exception_cannot_promote(self):
        def dispatch(phase,folder,deadline):
            if phase!='propose':return supervisor.command(phase,folder,deadline,self.spec)
            code=("from pathlib import Path;import scoped_proof.worker as p;import frontier_proof.worker as w;original=p.propose;count=0\n"
                  "def stop(lp,seconds):\n global count\n count+=1\n if count>1:raise RuntimeError('partial output')\n return original(lp,seconds)\n"
                  f"p.propose=stop;w.work('propose',Path({str(folder)!r}),{deadline!r})")
            return [PYTHON,'-c',code]
        folder=self.work/'partial';out=supervisor.supervise(folder,self.spec,budget=30,command_factory=dispatch)
        self.assertEqual(out['status'],'ERROR');report=audit(folder)
        self.assertEqual(report['checked_bounds'],1);self.assertFalse(report['complete_output_positive_proof'])
        self.assertEqual(load(folder/'evidence_check.json')['missing'],[5])

    def test_deadline_kills_only_owned_children_and_keeps_partial(self):
        def dispatch(phase,folder,deadline):
            code=("import subprocess,time;from pathlib import Path;p=subprocess.Popen(['sleep','10']);"
                  f"Path({str(folder/'child.pid')!r}).write_text(str(p.pid));"
                  f"Path({str(folder/'partial.txt')!r}).write_text('not evidence');time.sleep(10)")
            return [PYTHON,'-S','-c',code]
        folder=self.work/'timeout';out=supervisor.supervise(folder,self.spec,budget=.4,command_factory=dispatch)
        self.assertEqual(out['status'],'TIMEOUT');self.assertFalse(out['complete_output_positive_proof'])
        self.assertTrue((folder/'partial.txt').exists());audit(folder)
        pid=int((folder/'child.pid').read_text());stat=Path(f'/proc/{pid}/stat')
        if stat.exists():self.assertEqual(stat.read_text().rsplit(')',1)[1].split()[0],'Z')

    def test_parent_exception_rss_and_missing_evidence_costed(self):
        for mode in ('exception','rss','missing'):
            def dispatch(*args):
                if mode=='exception':raise RuntimeError('dispatch failure')
                return [PYTHON,'-S','-c','import time;time.sleep(.05)']
            folder=self.work/mode
            out=supervisor.supervise(folder,self.spec,budget=2,rss_limit=1 if mode=='rss' else 8*2**30,command_factory=dispatch)
            self.assertEqual(out['status'],'RESOURCE_LIMIT' if mode=='rss' else 'ERROR')
            # Fake successful workers have no proof artifacts: structural audit must reject.
            if mode=='missing':
                with self.assertRaises(ValueError):audit(folder)
            else:self.assertGreater(audit(folder)['end_to_end_seconds'],0)

    def test_late_positive_cannot_override_deadline(self):
        def dispatch(phase,folder,deadline):
            if phase!='aggregate':return supervisor.command(phase,folder,deadline,self.spec)
            code=("import time;from pathlib import Path;from frontier_proof.worker import work;"
                  f"work('aggregate',Path({str(folder)!r}),{deadline!r});time.sleep(20)")
            return [PYTHON,'-S','-c',code]
        folder=self.work/'late';out=supervisor.supervise(folder,self.spec,budget=9,command_factory=dispatch)
        self.assertEqual(out['status'],'TIMEOUT');self.assertTrue((folder/'result_candidate.json').exists())
        self.assertFalse(audit(folder)['complete_output_positive_proof'])

    def test_full_ledger_and_single_deadline_not_renewed(self):
        report=audit(self.good);cost=load(self.good/'cost.json');inv=load(self.good/'invocation.json')
        self.assertEqual(list(report['stage_seconds']),list(phases(self.spec)))
        self.assertAlmostEqual(cost['end_to_end_seconds'],cost['stage_seconds']+cost['overhead_seconds'])
        self.assertEqual(inv['budget_seconds'],30)
        self.assertAlmostEqual(inv['deadline_monotonic']-inv['started_monotonic'],30)
        folder=self.copy_good();ledger=load(folder/'cost.json');ledger['stage_seconds']=0;mutate(folder/'cost.json',ledger)
        with self.assertRaises(ValueError):audit(folder)

    def test_receipt_overrun_cannot_accept_and_budget_bound(self):
        original=supervisor.save
        def delayed(path,value):
            if path.name=='receipt.json':time.sleep(.7)
            return original(path,value)
        folder=self.work/'overrun'
        with patch.object(supervisor,'save',side_effect=delayed),patch.object(supervisor,'accept',return_value={'status':POSITIVE}):
            out=supervisor.supervise(folder,self.spec,budget=.6,command_factory=lambda *a:[PYTHON,'-S','-c','pass'])
        self.assertEqual(out['status'],'TIMEOUT');self.assertTrue((folder/'publication_timeout.json').exists())
        for budget in (0,301,float('nan'),True):
            with self.assertRaises(ValueError):supervisor.supervise(self.work/'bad',self.spec,budget=budget)

    def test_invalid_policy_and_checker_deadline_refused(self):
        for key,value in (('mode','automatic'),('parse_cache',True),('checker_cache',True),('router_proposal','SEARCH')):
            spec=copy.deepcopy(self.spec);spec['proof_policy'][key]=value
            with self.assertRaises(ValueError):policy(spec)
        with self.assertRaises(TimeoutError):route_inputs(self.good,time.monotonic()-1)
        with self.assertRaises(TimeoutError):output_inputs(self.good,self.spec['scope'],load(self.good/'construction.json'),
                                                        load(self.good/'invocation.json')['invocation'],time.monotonic()-1)

    def batch(self, failed=False):
        root=self.work/'batch';root.mkdir()
        common=copy.deepcopy(self.spec);common.pop('proof_policy')
        common['limits']={'whole_pipeline_seconds':30,'sampled_group_rss_bytes':8*2**30,'cpu_threads':2}
        calls=[];rows=[]
        for mode in ('exhaustive','checked_frontier'):
            p={**self.spec['proof_policy'],'mode':mode};calls.append({'id':mode,'proof_policy':p})
            rows.append({'id':mode,'status':'NOT_STARTED_RESOURCE','seconds':None,'launched':False})
        cfg={'common':common,'calls':calls,'output':str(root)}
        if failed:
            rows[1]={'id':'checked_frontier','status':'SUPERVISOR_ERROR','seconds':.1,'launched':True,
                     'error':'injected','complete_cost_receipt':False}
        save(root/'launch.json',{'config':cfg,'config_sha256':identity(cfg)})
        for row in rows:save(root/(row['id']+'_batch_terminal.json'),row)
        save(root/'execution.json',{'rows':rows,'config_sha256':identity(cfg),'no_retries':True})
        return root

    def test_batch_resource_refusals_keep_both_denominators(self):
        from frontier_proof.batch_audit import review
        report=review(self.batch())
        self.assertEqual((report['required_terminals'],report['missing_costs']), (2,2))
        self.assertTrue(all(r['seconds'] is None and not r['positive'] for r in report['rows']))
        self.assertFalse(report['same_source_verified']);self.assertEqual(report['audit'],'PASS')

    def test_batch_missing_duplicate_changed_terminal_rejected(self):
        from frontier_proof.batch_audit import review
        root=self.batch();execution=load(root/'execution.json')
        for rows in (execution['rows'][:1],execution['rows']*2,list(reversed(execution['rows']))):
            mutate(root/'execution.json',{**execution,'rows':rows})
            with self.assertRaises(ValueError):review(root)
        mutate(root/'execution.json',execution)
        terminal=load(root/'exhaustive_batch_terminal.json');terminal['seconds']=0
        mutate(root/'exhaustive_batch_terminal.json',terminal)
        with self.assertRaises(ValueError):review(root)

    def test_batch_supervisor_failure_is_explicit_audit_gap(self):
        from frontier_proof.batch_audit import review
        report=review(self.batch(failed=True))
        self.assertEqual(report['audit'],'INCOMPLETE');self.assertEqual(report['issues'],1)
        self.assertEqual(report['required_terminals'],2)
        self.assertFalse(report['frontier_only_positive'])

    def test_complete_batch_same_source_and_cost_review(self):
        from frontier_proof.batch_audit import review
        root=self.work/'batch';root.mkdir()
        common=copy.deepcopy(self.spec);common.pop('proof_policy')
        common['limits']={'whole_pipeline_seconds':30,'sampled_group_rss_bytes':8*2**30,'cpu_threads':2}
        calls=[{'id':mode,'proof_policy':{**self.spec['proof_policy'],'mode':mode}}
               for mode in ('exhaustive','checked_frontier')]
        cfg={'common':common,'calls':calls,'output':str(root)}
        save(root/'launch.json',{'config':cfg,'config_sha256':identity(cfg)})
        rows=[]
        for call in calls:
            out=supervisor.supervise(root/call['id'],{**common,'proof_policy':call['proof_policy']},budget=30)
            row={'id':call['id'],**out,'launched':True};rows.append(row)
            save(root/(call['id']+'_batch_terminal.json'),row)
        save(root/'execution.json',{'rows':rows,'config_sha256':identity(cfg),'no_retries':True})
        result=review(root)
        self.assertEqual(result['audit'],'PASS');self.assertTrue(result['same_source_verified'])
        self.assertTrue(result['frontier_only_positive']);self.assertGreater(result['observed_call_seconds'],0)
        # An accounting mutation cannot turn a 30-second control into a 300-second run.
        inv=load(root/'checked_frontier/invocation.json');inv['budget_seconds']=300
        mutate(root/'checked_frontier/invocation.json',inv)
        with self.assertRaises(ValueError):review(root)

    def test_accept_rejects_exclusion_and_guarantee_overclaim(self):
        for mode in ('zero','direction','coverage','native'):
            folder=self.work/mode;folder.mkdir()
            result=load(self.good/'evidence_check.json');header=load(self.good/'result_candidate.json')
            if mode=='zero':result['rows'][0]['exclusion']['checked_lower_bound']='0'
            if mode=='direction':result['rows'][0]['exclusion']['higher']=0
            if mode=='coverage':result['rows'].pop()
            if mode=='native':result['native_float_proof']=True;header['native_float_proof']=True
            header['evidence_check']=save(folder/'evidence_check.json',result)
            save(folder/'result_candidate.json',header)
            with self.assertRaises(ValueError):
                supervisor.accept(folder,self.spec['scope'],load(self.good/'invocation.json')['invocation'],'checked_frontier')


if __name__=='__main__': unittest.main()
