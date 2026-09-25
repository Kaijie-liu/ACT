"""Tiny synthetic controls only, never the two frozen profiling fixtures."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from source_enclosure.format import identity
from source_construction_lab.fixtures import document
from source_cost_supervised import audit, supervisor


class Controls(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get('SOURCE_COST_CONTROL_ROOT', '/data1/Kane/MOE'))
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=self._testMethodName + '_', dir=parent))
        fixture = dict(experts=2, classes=2, width=1, depth=0, seed=91)
        self.spec = {'id': 'tiny_control', 'fixture': fixture, 'source_sha256': identity(document(**fixture))}

    def call(self, **kwargs):
        result = supervisor.supervise(self.root / 'call', self.spec, **kwargs)
        save(self.root / 'returned.json', result)
        return result

    def factory(self, code, phase='profile'):
        def command(which, root, deadline):
            if which == phase:
                return [PYTHON, '-S', '-c', code, str(root)]
            return [PYTHON, '-S', '-m', 'source_cost_supervised.' + ('worker' if which == 'profile' else 'audit'), str(root)]
        return command

    def review(self, result): return audit.review(self.root / 'call', result)

    def test_complete_relocated_fresh_solver_free_recheck(self):
        r = self.call(); self.assertEqual(r['status'], 'COMPLETED')
        v = audit.review(self.root / 'call', r, recheck=True)
        self.assertTrue(v['source_rechecked']); self.assertFalse(v['complete_output_positive_proof'])
        import shutil
        shutil.copytree(self.root / 'call', self.root / 'relocated')
        code = '''import sys
from pathlib import Path
from scoped_proof.io import load
def forbid(event,args):
    if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','act','highspy') or args[0] in ('residual_proof.build','checked_route_frontier.build','shared_route_residual.propose','source_construction_lab.fixtures')): raise ImportError(args[0])
sys.addaudithook(forbid)
from source_cost_supervised.audit import review
out=review(Path(sys.argv[1]),load(Path(sys.argv[2])),recheck=True)
assert out['audit']=='PASS' and out['source_rechecked'] and not out['complete_output_positive_proof']
'''
        out = subprocess.run([PYTHON, '-S', '-c', code, str(self.root/'relocated'), str(self.root/'returned.json')],
                             cwd=ROOT, env=supervisor.ENV, capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_construct_cutoff_keeps_censored_component(self):
        code = '''import sys,time
from residual_proof import build
from source_cost_supervised.worker import run
build.network=lambda *a,**k:time.sleep(30)
run(sys.argv[1])'''
        r = self.call(budget=.8, command_factory=self.factory(code))
        self.assertEqual(r['status'], 'TIMEOUT')
        v = self.review(r)
        self.assertTrue(any(x['name']=='expert_propagation' and x['seconds'] is None for x in v['journal']['open']))
        self.assertFalse((self.root/'call/received.json').exists())

    def test_publication_cutoff_preserves_partial(self):
        code = '''import sys,time
from pathlib import Path
from source_cost_controls import profile
from source_cost_supervised.worker import run
def pause(path,*a,**k):
    Path(str(path)+'.partial').write_text('partial')
    time.sleep(30)
profile.publish=pause
run(sys.argv[1])'''
        r = self.call(budget=.8, command_factory=self.factory(code))
        self.assertEqual(r['status'], 'TIMEOUT'); v = self.review(r)
        self.assertTrue((self.root/'call/profile/source.json.partial').exists())
        self.assertTrue(any(x['name']=='publish_R1' for x in v['journal']['open']))

    def test_source_check_cutoff_keeps_unaccepted_files(self):
        code = '''import sys,time
from residual_proof import check
from source_cost_supervised.worker import run
check.check_network=lambda *a,**k:time.sleep(30)
run(sys.argv[1])'''
        r = self.call(budget=.8, command_factory=self.factory(code))
        self.assertEqual(r['status'], 'TIMEOUT'); v = self.review(r)
        self.assertTrue((self.root/'call/profile/construction.json').exists())
        self.assertTrue(any(x['name']=='expert_check' for x in v['journal']['open']))

    def test_exception_keeps_terminal_and_error_event(self):
        code = '''import sys
from residual_proof import build
from source_cost_supervised.worker import run
def fail(*a,**k):raise RuntimeError('injected construction failure')
build.network=fail
run(sys.argv[1])'''
        r = self.call(command_factory=self.factory(code)); self.assertEqual(r['status'], 'ERROR')
        v = self.review(r)
        self.assertTrue(any(x['event']=='ERROR' for x in v['journal']['finished']))

    def test_receiver_cutoff_does_not_accept_complete_profile(self):
        r = self.call(budget=.9, command_factory=self.factory('import time;time.sleep(30)', phase='receive'))
        self.assertEqual(r['status'], 'TIMEOUT'); self.review(r)
        self.assertTrue((self.root/'call/candidate.json').exists())
        self.assertFalse((self.root/'call/received.json').exists())

    def test_receiver_exception(self):
        r = self.call(command_factory=self.factory("raise RuntimeError('receiver failed')", phase='receive'))
        self.assertEqual(r['status'], 'ERROR'); self.review(r)

    def test_missing_candidate(self):
        r = self.call(command_factory=self.factory('pass'))
        self.assertEqual(r['status'], 'ERROR'); self.review(r)

    def test_wrong_generated_source(self):
        self.spec['source_sha256'] = '0'*64
        r = self.call(); self.assertEqual(r['status'], 'ERROR'); self.review(r)

    def test_resource_limit(self):
        r = self.call(rss_limit=1)
        self.assertEqual(r['status'], 'RESOURCE_LIMIT'); self.review(r)

    def test_owned_descendant_cleanup(self):
        code = '''import subprocess,sys
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
Path(sys.argv[1],'descendant.txt').write_text(str(p.pid))'''
        r = self.call(command_factory=self.factory(code)); self.assertEqual(r['status'], 'ERROR')
        self.review(r)
        pid = int((self.root/'call/descendant.txt').read_text())
        path = Path('/proc')/str(pid)/'stat'
        if path.exists(): self.assertEqual(path.read_text().rsplit(')',1)[1].split()[0], 'Z')

    def test_late_ledger_invalidates_success(self):
        original = supervisor.save
        def slow(path, value):
            if Path(path).name == 'cost.json': time.sleep(1.1)
            return original(path, value)
        with patch.object(supervisor, 'save', slow): r = self.call(budget=1.)
        self.assertEqual(r['status'], 'TIMEOUT'); self.review(r)

    def test_late_terminal_invalidates_success(self):
        original = supervisor.save
        def slow(path, value):
            if Path(path).name == 'terminal.json': time.sleep(1.1)
            return original(path, value)
        with patch.object(supervisor, 'save', slow): r = self.call(budget=1.)
        self.assertEqual(r['status'], 'TIMEOUT'); self.review(r)

    def test_bad_budget_and_no_overwrite(self):
        for budget in (0, -1, 301, float('nan'), True):
            with self.assertRaises(ValueError): self.call(budget=budget)
        self.call()
        with self.assertRaises(FileExistsError): self.call()

    def test_cost_mutations(self):
        r = self.call(); root=self.root/'call'
        inv=load(root/'invocation.json'); cost=load(root/'cost.json'); term=load(root/'terminal.json')
        mutations = []
        for key in ('seconds_before_ledger','overhead_seconds','budget_seconds','work_deadline_monotonic'):
            bad=copy.deepcopy(cost);bad[key]=-1;mutations.append((bad,term,r))
        for key in ('seconds','start_seconds','end_seconds','deadline_monotonic'):
            bad=copy.deepcopy(cost);bad['stages'][0][key]=-1;mutations.append((bad,term,r))
        bad=copy.deepcopy(cost);bad['stages'][0]['cleanup_included']=False;mutations.append((bad,term,r))
        bad=copy.deepcopy(cost);bad['stages'].pop();mutations.append((bad,term,r))
        late=dict(r,seconds_including_terminal=301);mutations.append((cost,term,late))
        for args in mutations:
            with self.assertRaises(ValueError): audit.cost_check(inv,*args)

    def test_candidate_binding_size_and_hash_mutations(self):
        r=self.call();self.assertEqual(r['status'],'COMPLETED');root=self.root/'call'
        inv=load(root/'invocation.json'); path=root/'candidate.json'; original=load(path)
        values=[]
        for key,value in [('invocation','other'),('spec_sha256','0'*64),('source_sha256','0'*64),
                          ('complete_output_positive_proof',True),('native_solver_calls',1)]:
            bad=copy.deepcopy(original);bad[key]=value;values.append(bad)
        bad=copy.deepcopy(original);bad['records']['construction.json']['bytes']+=1;values.append(bad)
        bad=copy.deepcopy(original);bad['records']['construction.json']['sha256']='0'*64;values.append(bad)
        for bad in values:
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): audit.receive(root,inv,self.spec,deadline=time.monotonic()+30)
        path.write_text(json.dumps(original))

    def test_missing_output_obligation_rejected(self):
        self.call();root=self.root/'call'; inv=load(root/'invocation.json')
        path=root/'profile/report.json'; report=load(path);report['source_check']['output_obligations']=0
        path.write_text(json.dumps(report))
        candidate=load(root/'candidate.json');candidate['records']['report.json']={'sha256':sha(path),'bytes':path.stat().st_size}
        (root/'candidate.json').write_text(json.dumps(candidate))
        with self.assertRaises(ValueError): audit.receive(root,inv,self.spec,deadline=time.monotonic()+30)

    def test_journal_corruption_and_partial_tail(self):
        self.call();root=self.root/'call';inv=load(root/'invocation.json');path=root/'journal.jsonl'
        raw=path.read_bytes(); lines=raw.splitlines(True)
        for value in (lines[1]+lines[0]+b''.join(lines[2:]), raw.replace(inv['invocation'].encode(),b'wrong',1)):
            path.write_bytes(value)
            with self.assertRaises(ValueError): audit.journal_check(path,inv,complete=True,deadline=time.monotonic()+30)
        path.write_bytes(raw+b'{')
        with self.assertRaises(ValueError): audit.journal_check(path,inv,complete=True,deadline=time.monotonic()+30)
        self.assertTrue(audit.journal_check(path,inv,complete=False,deadline=time.monotonic()+30)['truncated_tail'])


if __name__ == '__main__': unittest.main()
