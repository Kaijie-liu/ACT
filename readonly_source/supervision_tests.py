"""New-view hard cutoff, failures, partial evidence and whole segment controls."""
from pathlib import Path
import time
import unittest

from parsed_source_reuse import supervision_tests as prior
from parsed_source_reuse.check import clone
from readonly_source.execution import supervise, receive


class SupervisionControls(prior.SupervisionControls):
    def setUp(self):
        super().setUp(); self.method['representation'] = 'readonly'

    run_call = clone(prior.SupervisionControls.run_call, {'supervise':supervise, 'receive':receive})

    def test_complete_modes_same_result_full_cost_and_fresh_scope(self):
        results = []
        for mode in ('none','copy','readonly'):
            self.method.update(representation=mode, enabled=mode!='none')
            r = self.run_call(mode); self.assertEqual(r['status'], 'COMPLETED')
            p = prior.load(self.root/mode/'parse_stats.json')['stats']['parser']
            self.assertEqual(p['scope'][0], prior.load(self.root/mode/'invocation.json')['invocation'])
            self.assertEqual(p['hits'] > 0, mode!='none')
            results.append(prior.load(self.root/mode/'candidate.json')['result'])
        self.assertEqual(results[0],results[1]); self.assertEqual(results[1],results[2])

    def test_hard_cutoff_and_partial_stats_not_accepted(self):
        def hook(phase, root, deadline, args):
            if phase != 'profile': return args
            code = '''import sys,time
from pathlib import Path
from readonly_source import execution,cache
root=Path(sys.argv[1]);old=cache.SourceParser._timed
def pause(self,name,fn):
    if name=='freeze':
        (root/'parse_stats.json.partial').write_text('partial seal')
        time.sleep(30)
    return old(self,name,fn)
cache.SourceParser._timed=pause
execution.work(root,sys.argv[2])'''
            return [prior.PYTHON,'-S','-c',code,str(root),prior.identity(self.method)]
        r = self.run_call(budget=.8, command_hook=hook); self.assertEqual(r['status'],'TIMEOUT')
        self.assertTrue((self.root/'call/parse_stats.json.partial').exists())
        self.assertFalse((self.root/'call/received.json').exists())

    def test_exception_retains_failed_stats(self):
        def hook(phase, root, deadline, args):
            if phase != 'profile': return args
            code = '''import sys
from pathlib import Path
from readonly_source import execution,cache
def fail(*a,**k):raise ValueError('injected exact parser failure')
cache.SourceParser.unpack.__globals__['reference_unpack']=fail
execution.work(Path(sys.argv[1]),sys.argv[2])'''
            return [prior.PYTHON,'-S','-c',code,str(root),prior.identity(self.method)]
        r = self.run_call(command_hook=hook); self.assertEqual(r['status'],'ERROR')
        self.assertEqual(prior.load(self.root/'call/parse_stats.json')['stats']['status'],'ERROR')

    def test_mode_identity_cannot_be_changed(self):
        self.method['representation'] = 'none'
        with self.assertRaises(ValueError): self.run_call('bad')
        self.method['representation'] = 'readonly'; self.run_call()
        folder=self.root/'call'; record=prior.load(folder/'parse_stats.json')
        # Rebound stats hash cannot authorize a different representation policy.
        record['stats']['parser']['policy']='EXACT_SOURCE_PARSE_REUSE_R1'
        (folder/'parse_stats.json').write_text(prior.json.dumps(record))
        candidate=prior.load(folder/'candidate.json')
        candidate['stats_record']={'sha256':prior.sha(folder/'parse_stats.json'),'bytes':(folder/'parse_stats.json').stat().st_size}
        (folder/'candidate.json').write_text(prior.json.dumps(candidate))
        with self.assertRaises(ValueError): receive(folder,prior.identity(self.method),deadline=time.monotonic()+30)


for name, fn in vars(prior.SupervisionControls).items():
    if name.startswith('test_') and name not in vars(SupervisionControls):
        replacements = {k:v for k,v in {'supervise':supervise, 'receive':receive}.items() if k in fn.__globals__}
        setattr(SupervisionControls, name, clone(fn, replacements))


if __name__ == '__main__': unittest.main()
