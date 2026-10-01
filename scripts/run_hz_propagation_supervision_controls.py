"""Execute and independently audit the frozen 18-call CPU propagation roster."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT, load, save, sha
from scripts import hz_propagation_supervised as api
from scripts.test_hz_propagation_supervised import PropagationSupervisionTests, roster

FILES = (*api.FILES, 'scripts/test_hz_propagation_supervised.py',
         'scripts/run_hz_propagation_supervision_controls.py',
         'docs/hz_propagation_supervision_design_20261001.md')


def names():
    return [t.id() for t in unittest.defaultTestLoader.loadTestsFromTestCase(PropagationSupervisionTests)]


def run(root):
    if not root.is_absolute() or not root.is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new project archive')
    root.mkdir(parents=True, exist_ok=False)
    bindings = {}
    for name in FILES:
        target = root/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target); bindings[name] = sha(target)
    save(root/'implementation.json', bindings)
    os.environ['HZ_PROPAGATION_CONTROL_ROOT'] = str(root)
    outcomes = []
    class Result(unittest.TextTestResult):
        def startTest(self, t):
            self.begin = time.monotonic(); self.outcome = 'PASS'; super().startTest(t)
        def addError(self, t, err): self.outcome='ERROR'; super().addError(t,err)
        def addFailure(self, t, err): self.outcome='FAIL'; super().addFailure(t,err)
        def addSkip(self, t, why): self.outcome='SKIP'; super().addSkip(t,why)
        def addExpectedFailure(self, t, err): self.outcome='EXPECTED_FAILURE'; super().addExpectedFailure(t,err)
        def addUnexpectedSuccess(self, t): self.outcome='UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
        def stopTest(self, t):
            outcomes.append({'test':t.id(), 'status':self.outcome, 'seconds':time.monotonic()-self.begin})
            super().stopTest(t)
    started = time.monotonic()
    with (root/'tests.log').open('x') as log:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PropagationSupervisionTests)
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    passed = result.wasSuccessful() and [o['test'] for o in outcomes]==names() and all(o['status']=='PASS' for o in outcomes)
    summary = {'status':'PASS' if passed else 'FAIL', 'outcomes':outcomes,
        'protocol_sha256':api.PROTOCOL_SHA, 'implementation_sha256':sha(root/'implementation.json'),
        'calls_sha256':sha(root/'calls.json') if (root/'calls.json').exists() else None,
        'tests_log_sha256':sha(root/'tests.log'), 'seconds':time.monotonic()-started,
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'native_solves':0,'gpu_executions':0,'real_requests':0,'complete_moe_proofs':0}
    save(root/'summary.json',summary)
    return summary


def audit(root):
    summary=load(root/'summary.json'); bindings=load(root/'implementation.json',summary['implementation_sha256'])
    if (set(bindings)!=set(FILES) or summary['protocol_sha256']!=api.PROTOCOL_SHA
            or summary['status']!='PASS' or [o['test'] for o in summary['outcomes']]!=names()
            or any(o['status']!='PASS' for o in summary['outcomes'])):
        raise ValueError('complete control/source inventory')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name)!=digest or sha(ROOT/name)!=digest:
            raise ValueError('execution/audit source version mismatch: '+name)
    load(root/'summary.json')
    if sha(root/'tests.log')!=summary['tests_log_sha256']:
        raise ValueError('test log changed')
    calls=load(root/'calls.json',summary['calls_sha256'])
    if set(calls)!={r[0] for r in roster()}:
        raise ValueError('missing frozen call')
    checked=0
    for name,case,fault,budget,expected in roster():
        if (load(root/(name+'_started.json'))!={'spec':api.specification(case,fault),'budget':budget,'expected':expected}
                or calls[name]!=load(root/(name+'_observed.json')) or calls[name]['result']['status']!=expected):
            raise ValueError('invocation/observation/result mismatch')
        report=api.audit(root/name,observation=calls[name],recheck=True)
        checked+=report['bounds_rechecked']
        if fault and fault not in ('launch_failure','low_rss','late_publish'):
            events=[json.loads(line) for p in (root/name).glob('*_events.jsonl') for line in p.read_text().splitlines()]
            if not any(e.get('event')=='FAULT_REACHED' and e.get('name')==fault for e in events):
                raise ValueError('fault timed out before its intended seam')
            if fault=='descendant':
                s=load(root/name/'produce_stage.json')
                created=[e for e in events if e.get('event')=='DESCENDANT_STARTED']
                if (len(created)!=1 or created[0]['pgid']!=s['pid'] or created[0]['pid']==s['pid']
                        or type(created[0]['pid']) is not int or created[0]['pid']<=0
                        or s['descendant_on_leader_exit'] is not True
                        or s['cleanup_status']!='LEADER_REAPED_NO_LIVE_GROUP'):
                    raise ValueError('descendant fault was not exercised and cleaned')
    ref=load(ROOT/'docs/hz_checked_propagation_20261001_r5.json')
    old=load(Path(ref['archive'])/'observations.json',ref['observations_sha256'])
    for name in api.protocol()['cases']:
        p=load(root/name/'payload.json')['propagation']
        if p['scope']!=old[name]['scope'] or p['output']!=old[name]['output']:
            raise ValueError('actual propagation differential failed')
        for a,b in zip(p['events'],old[name]['events']):
            if a['package'] is not None and a['package']['accepted']!=b['package']['accepted']:
                raise ValueError('support differential failed')
    if any(summary[k]!=0 for k in ('native_solves','gpu_executions','real_requests','complete_moe_proofs')):
        raise ValueError('scope overclaim')
    return {'status':'PASS','calls':len(calls),'controls':len(names()),'bounds_rechecked':checked,
            'complete_propagations':4,'complete_moe_proofs':0,'summary_sha256':sha(root/'summary.json'),
            'scope':'actual CPU propagation; exact used-HZ support, trusted network/guard/factor/legacy lowering'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','audit']); p.add_argument('directory',type=Path)
    a=p.parse_args(); report=(run if a.action=='run' else audit)(a.directory.resolve())
    print(json.dumps(report,sort_keys=True))
    raise SystemExit(0 if report['status']=='PASS' else 1)
