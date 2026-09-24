"""Full synthetic checkpoint-to-proof controls; never reads a real request."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from scoped_proof.io import ROOT, PYTHON, save, load, sha
from scoped_proof.tests import make_fixture
from scoped_proof.evidence import POSITIVE
from source_enclosure.format import identity
from scoped_parse_proof.contract import SCHEMA, LIMITS, policy, validate_receipt
from scoped_parse_proof.supervisor import supervise, command
from scoped_parse_proof.audit import audit


def with_policy(spec, mode):
    return {**spec, 'construction_policy': {'schema':SCHEMA, 'mode':mode,
        'cache_limits':dict(LIMITS), 'checker_cache':False}}


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        self.root=Path(self.temp.name)
        self.base,_,_,_=make_fixture(self.root)
        self.spec=with_policy(self.base,'cached')

    def tearDown(self): self.temp.cleanup()

    def run_request(self, name='run', spec=None, **kw):
        root=self.root/name
        result=supervise(root,spec or self.spec,budget=kw.pop('budget',30),**kw)
        return root,result

    def test_two_complete_arms_identical_source_matrices_and_bounds(self):
        signatures=[]
        for mode in ('uncached','cached'):
            root,result=self.run_request(mode,with_policy(self.base,mode))
            self.assertEqual(result['status'],POSITIVE, '\n'.join(p.read_text() for p in root.glob('*.log')))
            checked=audit(root); self.assertEqual(checked['checked_bounds'],6)
            signatures.append((sha(root/'source.json'),sha(root/'construction.json'),
                [r['checked_lower_bound'] for r in load(root/'evidence_check.json')['rows']]))
            receipt=load(root/'construction_receipt.json')
            self.assertEqual(receipt['report']['parser']['enabled'],mode=='cached')
            self.assertFalse(load(root/'evidence_check.json')['native_float_proof'])
            fresh=subprocess.run([PYTHON,'-S','-m','scoped_parse_proof.audit',str(root)],
                cwd=ROOT,text=True,capture_output=True,timeout=15)
            self.assertEqual(fresh.returncode,0,fresh.stderr)
            self.assertTrue(json.loads(fresh.stdout)['construction_receipt_checked'])
        self.assertEqual(*signatures)

    def test_nonpositive_complete_stays_not_closed(self):
        spec,_,_,_=make_fixture(self.root,bias=-1.)
        root,result=self.run_request(spec=with_policy(spec,'cached'))
        self.assertEqual(result['status'],'NOT_CLOSED')
        checked=load(root/'evidence_check.json')
        self.assertEqual(len(checked['nonpositive']),6); self.assertFalse(checked['missing'])
        self.assertEqual(audit(root)['effective_status'],'NOT_CLOSED')

    def test_policy_is_explicit_and_no_checker_cache_or_capacity_tuning(self):
        for key,value in [('mode','reference'),('checker_cache',True),('cache_limits',{}),('schema','old')]:
            bad=copy.deepcopy(self.spec);bad['construction_policy'][key]=value
            with self.assertRaises(ValueError):policy(bad)
            _,result=self.run_request(key,spec=bad,budget=2)
            self.assertEqual(result['status'],'ERROR')

    def test_one_absolute_budget_and_cost_closure(self):
        root,result=self.run_request()
        self.assertEqual(result['status'],POSITIVE)
        inv=load(root/'invocation.json'); t=load(root/'terminal.json'); c=load(root/'cost.json')
        self.assertEqual(inv['deadline_monotonic']-inv['started_monotonic'],30)
        self.assertEqual(inv['deadline_monotonic']-inv['work_deadline_monotonic'],2)
        self.assertAlmostEqual(c['stage_seconds']+c['overhead_seconds'],c['end_to_end_seconds'])
        self.assertLessEqual(c['end_to_end_seconds'],result['seconds'])
        for row in t['stages']:
            self.assertTrue(row['cleanup_included'])
            if row['phase']=='propose':
                begin=inv['started_monotonic']+row['start_seconds']
                self.assertAlmostEqual(row['deadline_monotonic'],(begin+inv['work_deadline_monotonic'])/2,places=5)
            else:self.assertEqual(row['deadline_monotonic'],inv['work_deadline_monotonic'])
        with self.assertRaises(FileExistsError):supervise(root,self.spec,budget=30)

    def test_construct_hard_cutoff_keeps_events_no_receipt_or_bound(self):
        def dispatch(phase,folder,deadline):
            if phase!='construct':return command(phase,folder,deadline)
            code=("import time; from pathlib import Path; import scoped_parse_proof.worker as w; "
                "from source_construction_lab import build as b; old=b.construct\n"
                "def delayed(*a,**k):\n orig=k['emit']\n def event(v):\n  orig(v)\n  if v.get('operation')=='pair_guard' and v['event']=='ENTER': time.sleep(20)\n k['emit']=event\n return old(*a,**k)\n"
                f"b.construct=delayed; w.work('construct',Path({str(folder)!r}),{deadline!r})")
            return [PYTHON,'-S','-c',code]
        root,result=self.run_request(budget=8,command_factory=dispatch)
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertTrue((root/'source.json').exists())
        self.assertIn('pair_guard',(root/'construct_events.jsonl').read_text())
        self.assertFalse((root/'construction_receipt.json').exists())
        self.assertFalse((root/'candidates').exists());audit(root)

    def test_construct_exception_keeps_partial_no_success(self):
        def dispatch(phase,folder,deadline):
            if phase!='construct':return command(phase,folder,deadline)
            code=("from pathlib import Path; import scoped_parse_proof.worker as w; from source_construction_lab import build as b\n"
                "def fail(*a,**k):\n k['emit']({'event':'CONTROL_PARTIAL','operation':'source_prefix'})\n raise RuntimeError('controlled construction exception')\n"
                f"b.construct=fail; w.work('construct',Path({str(folder)!r}),{deadline!r})")
            return [PYTHON,'-S','-c',code]
        root,result=self.run_request(command_factory=dispatch)
        self.assertEqual(result['status'],'ERROR');self.assertFalse(result['complete_output_positive_proof'])
        self.assertIn('CONTROL_PARTIAL',(root/'construct_events.jsonl').read_text());audit(root)

    def test_partial_proposal_timeout_and_exception_checked_not_promoted(self):
        for kind in ('exception','timeout'):
            def dispatch(phase,folder,deadline):
                if phase!='propose':return command(phase,folder,deadline)
                action="raise RuntimeError('after first')" if kind=='exception' else 'time.sleep(20)'
                code=("import time; from pathlib import Path; import scoped_proof.worker as w; old=w.propose; n=0\n"
                    "def limited(lp,s):\n global n\n n+=1\n"
                    f" if n>1: {action}\n return old(lp,s)\n"
                    f"w.propose=limited; w.work('propose',Path({str(folder)!r}),{deadline!r})")
                return [PYTHON,'-c',code]
            root,result=self.run_request(kind,budget=10,command_factory=dispatch)
            self.assertEqual(result['status'],'ERROR' if kind=='exception' else 'NOT_CLOSED')
            self.assertFalse(result['complete_output_positive_proof'])
            evidence=load(root/'evidence_check.json')
            self.assertEqual(evidence['checked_bounds'],1);self.assertEqual(len(evidence['missing']),5)
            self.assertFalse(evidence['proposal_complete']);audit(root)

    def test_late_complete_evidence_does_not_override_cutoff(self):
        def dispatch(phase,folder,deadline):
            if phase!='aggregate':return command(phase,folder,deadline)
            return [PYTHON,'-S','-c',"import time; from pathlib import Path; from scoped_proof.worker import work; "
                f"work('aggregate',Path({str(folder)!r}),{deadline!r}); time.sleep(20)"]
        root,result=self.run_request(budget=8,command_factory=dispatch)
        self.assertEqual(result['status'],'TIMEOUT');self.assertFalse(result['complete_output_positive_proof'])
        self.assertTrue((root/'result_candidate.json').exists());audit(root)

    def test_missing_receipt_and_wrong_binding_fail_before_source_check(self):
        def dispatch(phase,folder,deadline):
            if phase!='source_check':return command(phase,folder,deadline)
            (folder/'construction_receipt.json').rename(folder/'unaccepted_receipt.json')
            return command(phase,folder,deadline)
        root,result=self.run_request(command_factory=dispatch)
        self.assertEqual(result['status'],'ERROR');self.assertFalse((root/'candidates').exists())
        # A deleted completed receipt is a structural audit error, not a harmless partial.
        with self.assertRaises(FileNotFoundError):audit(root)
        receipt=load(root/'unaccepted_receipt.json')
        for key,value in [('invocation','other'),('spec_sha256','bad'),('policy_sha256','bad')]:
            p=self.root/key;shutil.copytree(root,p)
            save(p/'construction_receipt.json',{**receipt,key:value})
            with self.assertRaises(ValueError):validate_receipt(p)

    def test_corrupt_matrix_cannot_borrow_completed_receipt(self):
        root,result=self.run_request();self.assertEqual(result['status'],POSITIVE)
        altered=self.root/'altered';shutil.copytree(root,altered)
        bundle=load(altered/'construction.json');bundle['pairs'].pop()
        (altered/'construction.json').unlink();save(altered/'construction.json',bundle)
        with self.assertRaises(ValueError):audit(altered)

    def test_rss_refusal_and_parent_dispatch_exception_are_costed(self):
        for name in ('rss','dispatch'):
            def dispatch(*args):
                if name=='dispatch':raise RuntimeError('dispatch failure')
                return [PYTHON,'-S','-c','import time; time.sleep(2)']
            root,result=self.run_request(name,budget=2,rss_limit=1 if name=='rss' else 8*2**30,command_factory=dispatch)
            self.assertEqual(result['status'],'RESOURCE_LIMIT' if name=='rss' else 'ERROR')
            self.assertGreater(audit(root)['end_to_end_seconds'],0)

    def test_receipt_publication_overrun_is_not_free(self):
        from scoped_proof import supervisor as original
        real_save=original.save
        def slow(path,value):
            if Path(path).name=='receipt.json':time.sleep(.7)
            return real_save(path,value)
        with patch.object(original,'save',side_effect=slow),patch.object(original,'accept',return_value={'status':POSITIVE}):
            root,result=self.run_request(budget=.6,command_factory=lambda *a:[PYTHON,'-S','-c','pass'])
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertTrue((root/'publication_timeout.json').exists())
        self.assertGreater(load(root/'cost.json')['end_to_end_seconds'],.6)

    def batch(self, name, complete=False):
        root=self.root/name;root.mkdir()
        cfg={'output':str(root),'common':self.base,'calls':[
            {'id':m,'construction_policy':with_policy(self.base,m)['construction_policy']}
            for m in ('cached','uncached')]}
        save(root/'launch.json',{'config':cfg,'config_sha256':identity(cfg)})
        rows=[]
        for call in cfg['calls']:
            if complete and call['id']=='cached':
                result=supervise(root/'cached',self.spec,budget=30)
                self.assertEqual(result['status'],POSITIVE)
                row={'id':'cached',**result,'launched':True}
            else:row={'id':call['id'],'status':'NOT_STARTED_RESOURCE','seconds':None,'launched':False}
            rows.append(row);save(root/(call['id']+'_batch_terminal.json'),row)
        save(root/'execution.json',{'config_sha256':identity(cfg),'rows':rows,'no_retries':True})
        return root

    def test_batch_audit_complete_proof_and_not_started_denominator(self):
        from scoped_parse_proof.batch_audit import review
        result=review(self.batch('batch',complete=True))
        self.assertEqual(result['required_terminals'],2);self.assertEqual(result['missing_costs'],1)
        self.assertFalse(result['all_constructions_identical'])
        self.assertIsNone(result['rows'][1]['cost_seconds'])
        self.assertGreater(result['observed_call_seconds'],0)

    def test_batch_missing_duplicate_and_changed_terminal_rejected(self):
        from scoped_parse_proof.batch_audit import review
        for kind in ('missing','duplicate','changed'):
            root=self.batch(kind)
            self.assertEqual(review(root)['missing_costs'],2)
            data=load(root/'execution.json')
            if kind=='missing':data['rows'].pop()
            if kind=='duplicate':data['rows'][1]=data['rows'][0]
            if kind=='changed':data['rows'][0]['seconds']=0.
            (root/'execution.json').unlink();save(root/'execution.json',data)
            with self.assertRaises(ValueError):review(root)

    def test_dispatch_preserves_other_workers_and_absolute_deadline(self):
        for phase in ('intake','construct','source_check','propose','aggregate'):
            cmd=command(phase,self.root,123.5)
            expected='scoped_parse_proof.worker' if phase in ('construct','source_check') else 'scoped_proof.worker'
            self.assertEqual(cmd[cmd.index('-m')+1],expected)
            self.assertEqual(cmd[-2:],['--deadline','123.5'])
            self.assertEqual('-S' in cmd,phase in ('construct','source_check','aggregate'))


if __name__=='__main__':unittest.main()
