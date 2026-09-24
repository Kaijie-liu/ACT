import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from checked_route_frontier.fixtures import analytic
from scoped_proof.io import PYTHON, load, save
from source_enclosure.format import identity
from shared_route_residual import study
from shared_route_residual.review import audit_one


def small(phase,root,deadline):
    # Execution controls use small fixtures; timing study uses the unmodified worker.
    code="from checked_route_frontier.fixtures import analytic;import shared_route_residual.fixtures as f;f.fixture=lambda kind:analytic();from shared_route_residual.worker import work;from pathlib import Path;work(%r,Path(%r),%r)"%(phase,str(root),deadline)
    return [PYTHON,'-S','-c',code]


class SupervisionControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/data1/Kane/MOE');self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def run_call(self,name,**kw):
        return study.run_one(self.root/name,'prunable','shared',budget=kw.pop('budget',5.),
                             expected_source_sha256=identity(analytic()),**kw)

    def test_complete_cost_and_relocation(self):
        r=self.run_call('a',commands=small);self.assertEqual(r['status'],'COMPLETED_ROUTER_SEGMENT')
        original=audit_one(self.root/'a');(self.root/'a').rename(self.root/'moved')
        self.assertEqual(original,audit_one(self.root/'moved'))
        self.assertFalse(original['result']['complete_output_positive_proof'])

    def test_timeout_preserves_partial_source_and_never_accepts(self):
        def command(phase,root,deadline):
            return [PYTHON,'-S','-c',"from pathlib import Path;from scoped_proof.io import save;import time;save(Path(%r)/'partial.json',{'partial':True});time.sleep(10)"%str(root)]
        r=self.run_call('a',commands=command,budget=.4)
        self.assertEqual(r['status'],'TIMEOUT');self.assertTrue((self.root/'a'/'partial.json').exists())
        self.assertIsNone(audit_one(self.root/'a')['result']);self.assertLess(r['seconds'],2.)

    def test_exception_and_missing_receipt_rejected(self):
        for mode,code in [('error','raise RuntimeError("control")'),('empty','pass')]:
            r=self.run_call(mode,commands=lambda *args:[PYTHON,'-S','-c',code])
            self.assertEqual(r['status'],'ERROR');self.assertIsNone(audit_one(self.root/mode)['result'])

    def test_check_exception_preserves_candidates(self):
        def command(phase,root,deadline):
            return small(phase,root,deadline) if phase=='build' else [PYTHON,'-S','-c','raise RuntimeError("check control")']
        r=self.run_call('a',commands=command)
        self.assertEqual(r['status'],'ERROR');self.assertTrue((self.root/'a'/'candidates.json').exists())
        self.assertIsNone(audit_one(self.root/'a')['result'])

    def test_wrong_run_missing_bounds_and_claim_rejected(self):
        self.run_call('a',commands=small);root=self.root/'a';plan=load(root/'plan.json');real=study.load
        for mode in ('run','margin','pair','claim','candidate'):
            bad=copy.deepcopy(real(root/'check.json'))
            if mode=='run':bad['invocation']='other'
            if mode=='margin':bad['result']['bounds'].pop()
            if mode=='pair':bad['result']['pairs'].pop()
            if mode=='claim':bad['complete_output_positive_proof']=True
            if mode=='candidate':bad['candidate_sha256']='0'*64
            with patch.object(study,'load',side_effect=lambda path:bad if path.name=='check.json' else real(path)):
                with self.subTest(mode=mode),self.assertRaises(ValueError):study.accept(root,plan)

    def test_actual_cost_mutations_rejected(self):
        self.run_call('a',commands=small);root=self.root/'a'
        from shared_route_residual import review
        original=load(root/'cost.json');real=review.load
        for mode in ('total','stage','overhead','rss','status','hash','budget'):
            bad=copy.deepcopy(original)
            if mode=='total':bad['end_to_end_seconds']=0.
            if mode=='stage':bad['stage_seconds']=0.
            if mode=='overhead':bad['overhead_seconds']=-1.
            if mode=='rss':bad['sampled_peak_rss']=0
            if mode=='status':bad['status']='SAFE'
            if mode=='hash':bad['terminal_sha256']='0'*64
            if mode=='budget':bad['budget_seconds']=301
            with patch.object(review,'load',side_effect=lambda path:bad if path.name=='cost.json' else real(path)):
                with self.subTest(mode=mode),self.assertRaises(ValueError):audit_one(root)

    def test_terminal_publication_deadline_blocks_acceptance(self):
        real=study.save
        def delayed(path,value):
            if path.name=='terminal.json':time.sleep(1.)
            return real(path,value)
        with patch.object(study,'save',side_effect=delayed):
            r=self.run_call('a',commands=small,budget=.6)
        self.assertEqual(r['status'],'TIMEOUT');self.assertIsNone(audit_one(self.root/'a')['result'])

    def test_no_overwrite_or_policy_extension(self):
        self.run_call('a',commands=small)
        with self.assertRaises(FileExistsError):self.run_call('a',commands=small)
        for budget in (0,301,float('inf'),True):
            with self.assertRaises(ValueError):self.run_call('bad',budget=budget)


if __name__=='__main__':unittest.main()
