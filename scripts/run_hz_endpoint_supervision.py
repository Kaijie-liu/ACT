"""Execute the frozen controls once, or independently recheck the whole archive."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT,load,save,sha
from scoped_proof import device_lifecycle as engine
from scripts import hz_endpoint_supervised as policy


def names():
    return sorted('scripts.test_hz_endpoint_supervised.EndpointSupervisionTests.test_'+n
                  for n in policy.protocol()['controls'])


def validate_calls(calls):
    from scripts.test_hz_endpoint_supervised import roster
    if set(calls)!={n for n,_ in roster()}: raise ValueError('complete frozen call roster required')


def event_costs(root,terminal):
    """Nested diagnostics are not added to stage cost; open operations are censored."""
    result={}
    for stage in terminal['stages']:
        phase=stage['phase']; path=root/(phase+'_events.jsonl')
        if not path.exists():
            if stage['status']=='COMPLETED': raise ValueError('completed worker lacks cost events')
            result[phase]=[]; continue
        pending=None; rows=[]; previous=stage['start_seconds']
        for line in path.read_text().splitlines():
            e=json.loads(line); policy.finite(e['elapsed'])
            if not previous<=e['elapsed']<=stage['end_seconds']: raise ValueError('operation clock order')
            previous=e['elapsed']
            if e['event']=='ENTER':
                if pending is not None: raise ValueError('unexpected nested operation')
                pending=e
            elif e['event'] in ('EXIT','EXIT_ERROR'):
                policy.finite(e['seconds'])
                if (pending is None or e['operation']!=pending['operation']
                        or e['seconds']>e['elapsed']-pending['elapsed']+.01):
                    raise ValueError('operation duration/identity')
                rows.append({'operation':e['operation'],'seconds':e['seconds'],'status':e['event']})
                pending=None
        if pending is not None:
            if stage['status']=='COMPLETED': raise ValueError('unfinished operation in completed phase')
            rows.append({'operation':pending['operation'],'seconds':None,'status':'RIGHT_CENSORED',
                         'observed_lower_seconds':max(0.,stage['end_seconds']-pending['elapsed']-stage['cleanup_seconds'])})
        result[phase]=rows
    return result


def run(root):
    root.mkdir(parents=True,exist_ok=False); bindings=policy.sources()
    for name in bindings:
        target=root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save(root/'implementation.json',bindings)
    from scripts import test_hz_endpoint_supervised as tests
    tests.ROOT=root; tests.CALLS.clear()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.EndpointSupervisionTests)
    if [t.id() for t in suite]!=names(): raise ValueError('frozen test roster')
    begin=time.monotonic()
    with (root/'tests.log').open('x') as log:
        class Result(unittest.TextTestResult):
            outcomes=[]
            def startTest(self,t): self.state='PASS'; super().startTest(t)
            def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
            def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
            def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
            def addExpectedFailure(self,t,e): self.state='EXPECTED_FAILURE'; super().addExpectedFailure(t,e)
            def addUnexpectedSuccess(self,t): self.state='UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
            def stopTest(self,t): self.outcomes.append({'test':t.id(),'status':self.state}); super().stopTest(t)
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result).run(suite)
    save(root/'calls.json',tests.CALLS)
    summary={'status':'PASS' if result.wasSuccessful() and all(o['status']=='PASS' for o in result.outcomes) else 'FAIL',
             'tests':result.testsRun,'outcomes':result.outcomes,'calls':len(tests.CALLS),'seconds':time.monotonic()-begin,
             'protocol_sha256':policy.CONFIG_SHA,'implementation_sha256':sha(root/'implementation.json'),
             'calls_sha256':sha(root/'calls.json'),'tests_log_sha256':sha(root/'tests.log'),
             'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'native_solves':0,'cuda_calls':0,'real_requests':0,'source_complete_proofs':0}
    save(root/'summary.json',summary); return summary


def audit(root):
    from scripts.test_hz_endpoint_supervised import roster,fault_reached
    summary=load(root/'summary.json'); calls=load(root/'calls.json')
    bindings=load(root/'implementation.json'); validate_calls(calls)
    if (summary['status']!='PASS' or summary['tests']!=len(names()) or summary['calls']!=len(calls)
            or summary['outcomes']!=[{'test':n,'status':'PASS'} for n in names()]
            or summary['calls_sha256']!=sha(root/'calls.json') or summary['tests_log_sha256']!=sha(root/'tests.log')
            or summary['implementation_sha256']!=sha(root/'implementation.json') or bindings!=policy.sources()
            or summary['protocol_sha256']!=policy.CONFIG_SHA
            or any(summary[k]!=0 for k in ('native_solves','cuda_calls','real_requests','source_complete_proofs'))):
        raise ValueError('archive/source/test/scope identity')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name)!=digest: raise ValueError('snapshot changed')
    reference_report=load(ROOT/policy.protocol()['reference_report'],policy.REPORT_SHA)
    reference=reference_report['checked']; oldroot=Path(reference_report['root'])
    oldsummary=load(oldroot/'summary.json',reference_report['summary_sha256'])
    oldobservations=load(oldroot/'observations.json',oldsummary['observations_sha256'])
    if {p.name for p in root.iterdir() if p.is_dir()}!={'implementation',*calls}:
        raise ValueError('unregistered or missing execution directory')
    reports={}
    for i,(name,s) in enumerate(roster()):
        path=root/name; inv=load(path/'invocation.json')
        if (load(root/('observed_'+name+'.json'))!=calls[name]
                or load(root/('pending_after_'+name+'.json'))!={
                    'pending':[n for n,_ in roster()[i+1:]],'cleanup_confirmed':True}):
            raise ValueError('durable observation/pending inventory')
        if inv['spec']!=s: raise ValueError('call request changed')
        r=engine.audit(policy,path,observation=calls[name],recheck=True); fault_reached(path,s)
        expected=policy.protocol()['faults'][name][1] if s['control'] else policy.DONE
        if r['execution_status']!=expected: raise ValueError('unexpected fixed terminal')
        term=load(path/'terminal.json'); agg=None
        if term['accepted'] is not None:
            agg=term['accepted']['checked']['aggregation']
            if s['reference'] is not None:
                ref=reference[s['reference']]
                for key in ('status','required','positive','checked_endpoints','missing_endpoints'):
                    if agg[key]!=ref[key]: raise ValueError('changed reference outcome')
                if str(min(F(row['lower_bound']) for row in agg['results']))!=ref['minimum_bound']:
                    raise ValueError('changed exact lower bound')
                if agg['results']!=oldobservations[s['reference']]['checked']['results']:
                    raise ValueError('individual endpoint reference changed')
            elif (agg['status']!='UNKNOWN_MISSING_EVIDENCE' or agg['checked_endpoints']!=4 or agg['missing_endpoints']!=2):
                raise ValueError('partial case upgraded')
        prefix=policy.recheck_prefix(path,inv,time.monotonic()+30)
        reports[name]={**r,'obligations':agg,'offline_prefix_checked_endpoints':0 if prefix is None else prefix['checked_endpoints'],
                       'seconds':calls[name]['result']['seconds'],'stage_seconds':term['stage_seconds'],
                       'parent_seconds':term['parent_seconds'],'offline_prefix_is_online_acceptance':False,
                       'nested_operation_costs':event_costs(path,term)}
    if (root/'batch_stop.json').exists(): raise ValueError('stopped batch cannot pass full roster')
    return {'status':'PASS','root':str(root),'summary_sha256':sha(root/'summary.json'),
            'tests':len(names()),'calls':reports,'source_complete':False,'cuda_calls':0,'real_requests':0,
            'native_solves':0,'all_six_goal_gates_remain_open':True}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['run','audit'])
    parser.add_argument('root',type=Path); parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if not args.root.is_absolute() or not args.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('project result path required')
    result=run(args.root) if args.action=='run' else audit(args.root)
    if args.report: save(args.report,result)
    print({'status':result['status'],'tests':result['tests'],'root':str(args.root)})


if __name__=='__main__': main()
