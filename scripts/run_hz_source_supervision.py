"""Execute the fixed whole-source controls and audit all saved terminals."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT,load,save,sha
from scoped_proof import device_lifecycle as engine
from scripts import hz_source_supervised as policy
from scripts.run_hz_endpoint_supervision import event_costs as base_event_costs


def event_costs(root, terminal):
    """Retain inherited clocks and reject empty or unfinished successful logs."""
    costs=base_event_costs(root,terminal)
    for stage in terminal['stages']:
        if stage['status']!='COMPLETED': continue
        phase=stage['phase']
        events=[json.loads(line) for line in (root/(phase+'_events.jsonl')).read_text().splitlines()]
        finished=[e for e in events if e['event']=='WORKER_COMPLETE']
        final=[r for r in costs[phase] if r['operation']=='final_serialization']
        if (not events or len(finished)!=1 or events[-1]!=finished[0]
                or finished[0].get('phase')!=phase or len(final)!=1 or final[0]['status']!='EXIT'):
            raise ValueError('completed phase lacks final cost/finish events')
        if phase=='produce':
            imports=[r for r in costs[phase] if r['operation']=='numerical_imports']
            if len(imports)!=1 or imports[0]['status']!='EXIT':
                raise ValueError('completed producer lacks import cost')
    return costs


def names():
    return sorted('scripts.test_hz_source_supervised.SourceSupervisionTests.test_'+n
                  for n in policy.protocol()['controls'])


def validate_calls(calls):
    from scripts.test_hz_source_supervised import roster
    if set(calls)!={n for n,_ in roster()}: raise ValueError('complete frozen call roster required')


def run(root):
    root.mkdir(parents=True,exist_ok=False); bindings=policy.sources()
    for name in bindings:
        target=root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save(root/'implementation.json',bindings)
    from scripts import test_hz_source_supervised as tests
    tests.ROOT=root; tests.CALLS.clear()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.SourceSupervisionTests)
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
    summary={'status':'PASS' if result.wasSuccessful() and result.testsRun==len(names())
                and len(result.outcomes)==len(names()) and all(o['status']=='PASS' for o in result.outcomes) else 'FAIL',
             'tests':result.testsRun,'outcomes':result.outcomes,'calls':len(tests.CALLS),'seconds':time.monotonic()-begin,
             'protocol_sha256':policy.CONFIG_SHA,'implementation_sha256':sha(root/'implementation.json'),
             'calls_sha256':sha(root/'calls.json'),'tests_log_sha256':sha(root/'tests.log'),
             'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'exceptional_tests':{k:len(getattr(result,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')},
             'native_solves':0,'cuda_calls':0,'real_requests':0,'performance_claim':False}
    save(root/'summary.json',summary); return summary


def audit(root):
    from scripts.test_hz_source_supervised import roster,fault_reached
    summary=load(root/'summary.json'); calls=load(root/'calls.json')
    bindings=load(root/'implementation.json'); validate_calls(calls)
    if (summary['status']!='PASS' or summary['tests']!=len(names()) or summary['calls']!=len(calls)
            or summary['outcomes']!=[{'test':n,'status':'PASS'} for n in names()]
            or summary['exceptional_tests']!={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}
            or summary['calls_sha256']!=sha(root/'calls.json') or summary['tests_log_sha256']!=sha(root/'tests.log')
            or summary['implementation_sha256']!=sha(root/'implementation.json') or bindings!=policy.sources()
            or summary['protocol_sha256']!=policy.CONFIG_SHA or summary['performance_claim'] is not False
            or any(type(summary[k]) is not int or summary[k]!=0 for k in ('native_solves','cuda_calls','real_requests'))):
        raise ValueError('archive/source/test/scope identity')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name)!=digest: raise ValueError('snapshot changed')
    cfg=policy.protocol(); reference=load(ROOT/cfg['reference_report'],policy.REPORT_SHA)
    oldroot=Path(cfg['reference_root'])
    oldsummary=load(oldroot/'summary.json',reference['summary_sha256'])
    oldobs=load(oldroot/'observations.json',oldsummary['observations_sha256'])
    if {p.name for p in root.iterdir() if p.is_dir()}!={'implementation',*calls}:
        raise ValueError('unregistered or missing execution directory')
    reports={}
    for i,(name,s) in enumerate(roster()):
        path=root/name; inv=load(path/'invocation.json')
        pending=[n for n,_ in roster()[i+1:]]
        if (load(root/('observed_'+name+'.json'))!=calls[name]
                or load(root/('started_'+name+'.json'))!={'name':name,'spec_sha256':policy.identity(s),'pending':pending}
                or load(root/('pending_after_'+name+'.json'))!={'pending':pending,'cleanup_confirmed':True}):
            raise ValueError('durable started/observation/pending inventory')
        if inv['spec']!=s: raise ValueError('call request changed')
        audit_start=time.monotonic()
        r=engine.audit(policy,path,observation=calls[name],recheck=True); fault_reached(path,s)
        execution_recheck_seconds=time.monotonic()-audit_start
        expected=cfg['faults'][name][1] if s['control'] else policy.DONE
        if r['execution_status']!=expected: raise ValueError('unexpected fixed terminal')
        term=load(path/'terminal.json'); agg=None
        if term['accepted'] is not None:
            agg=term['accepted']['checked']['aggregation']
            ref=oldobs['partial_proof' if s['partial'] else s['source_case']]
            if agg!=ref['checked']: raise ValueError('source or individual exact reference bound changed')
            if policy.payload(path,inv,term['accepted']['inputs']['produce'])['package']!=ref['package']:
                raise ValueError('new execution did not retain same complete source/endpoint representation')
            if not s['partial'] and agg['missing_endpoints']!=0: raise ValueError('normal proof incomplete')
            if s['partial'] and (agg['required'],agg['checked_endpoints'],agg['missing_endpoints'])!=(18,15,3):
                raise ValueError('registered partial coverage changed')
        begin=time.monotonic()
        try:
            prefix=policy.recheck_prefix(path,inv,time.monotonic()+30)
        except ValueError as exc:
            if s['control']!='missing_endpoint' or str(exc)!='incomplete support certificate roster': raise
            prefix={'status':'REJECTED_INVALID_CANDIDATE_PREFIX','reason':str(exc),'checked_endpoints':0,'output_accepted':False}
        else:
            if s['control']=='missing_endpoint': raise ValueError('bad endpoint prefix unexpectedly accepted')
        reports[name]={**r,'obligations':agg,'offline_prefix':prefix,
                       'offline_prefix_check_seconds':time.monotonic()-begin,
                       'offline_execution_recheck_seconds':execution_recheck_seconds,
                       'seconds':calls[name]['result']['seconds'],'stage_seconds':term['stage_seconds'],
                       'parent_seconds':term['parent_seconds'],'offline_prefix_is_online_acceptance':False,
                       'final_publication_seconds':calls[name]['result']['seconds']-term['seconds'],
                       'observed_api_seconds':calls[name]['end']-calls[name]['begin'],
                       'outer_observation_seconds':(calls[name]['end']-calls[name]['begin']
                                                    -calls[name]['result']['seconds']),
                       'nested_operation_costs':event_costs(path,term)}
    if (root/'batch_stop.json').exists() or list(root.glob('api_error_*.json')):
        raise ValueError('stopped batch cannot pass full roster')
    return {'status':'PASS','root':str(root),'summary_sha256':sha(root/'summary.json'),
            'tests':len(names()),'calls':reports,'cuda_calls':0,'real_requests':0,'native_solves':0,
            'performance_claim':False,'deployed_float_SAFE':False,'all_six_goal_gates_remain_open':True}


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
