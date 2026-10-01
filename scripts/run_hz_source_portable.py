"""Frozen offline portability controls and independent archive receipt audit."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT,load,save,sha
from scripts import hz_source_portable as p


def names():
    return sorted('scripts.test_hz_source_portable.PortableSourceTests.test_'+n for n in p.protocol()['controls'])


def mutation_inventory(root):
    result={}
    for path in sorted((root/'mutations').rglob('*')):
        name=str(path.relative_to(root/'mutations'))
        if path.is_symlink(): result[name]={'kind':'symlink','target':os.readlink(path)}
        elif path.is_dir(): result[name]={'kind':'directory'}
        elif path.is_file(): result[name]={'kind':'file','bytes':path.stat().st_size,'sha256':sha(path)}
        else: raise ValueError('unexpected mutation artifact type')
    return result


def run(root):
    root.mkdir(parents=True,exist_ok=False); bindings=p.sources()
    for name in bindings:
        dest=root/'implementation'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    save(root/'implementation.json',bindings)
    from scripts import test_hz_source_portable as tests
    tests.ROOT=root; tests.OBS.clear()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.PortableSourceTests)
    if [t.id() for t in suite]!=names(): raise ValueError('test inventory')
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
        # Save child stages before assertions; any failure stops further work.
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result,failfast=True).run(suite)
    save(root/'calls.json',tests.OBS)
    save(root/'mutation_inventory.json',mutation_inventory(root))
    summary={'status':'PASS' if result.wasSuccessful() and result.testsRun==len(names()) and
             result.outcomes==[{'test':n,'status':'PASS'} for n in names()] else 'FAIL',
             'tests':result.testsRun,'outcomes':result.outcomes,'seconds':time.monotonic()-begin,
             'exceptional_tests':{k:len(getattr(result,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')},
             'implementation_sha256':sha(root/'implementation.json'),'calls_sha256':sha(root/'calls.json'),
             'tests_log_sha256':sha(root/'tests.log'),'protocol_sha256':p.CONFIG_SHA,
             'mutation_inventory_sha256':sha(root/'mutation_inventory.json'),
             'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
             'new_proposals':0,'native_solves':0,'cuda_calls':0,'real_requests':0}
    save(root/'summary.json',summary); return summary


def audit(root):
    summary=load(root/'summary.json'); calls=load(root/'calls.json'); bindings=load(root/'implementation.json')
    if (summary['status']!='PASS' or summary['tests']!=len(names())
            or summary['outcomes']!=[{'test':n,'status':'PASS'} for n in names()]
            or summary['exceptional_tests']!={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}
            or summary['implementation_sha256']!=sha(root/'implementation.json') or bindings!=p.sources()
            or summary['protocol_sha256']!=p.CONFIG_SHA or summary['calls_sha256']!=sha(root/'calls.json')
            or summary['tests_log_sha256']!=sha(root/'tests.log') or set(calls)!=set(p.protocol()['cases'])
            or any(type(summary[k]) is not int or summary[k]!=0 for k in ('new_proposals','native_solves','cuda_calls','real_requests'))):
        raise ValueError('frozen archive inventory/identity')
    for n,h in bindings.items():
        if sha(root/'implementation'/n)!=h: raise ValueError('saved implementation changed')
    if (root/'batch_stop.json').exists(): raise ValueError('stopped execution cannot pass')
    inventory=load(root/'mutation_inventory.json',summary['mutation_inventory_sha256'])
    if inventory!=mutation_inventory(root): raise ValueError('mutation archive inventory/identity')
    copies={'code_verify.py','code_code_scoped_source_hz_source_check.py','source_change',
            *('inventory_'+s for s in ('missing','extra','symlink','traversal','duplicate')),
            *('math_'+s for s in ('pair','property','factor','input')),'shadow'}
    child_names={'math_pair','math_property','math_factor','math_input','shadow','isolation','expired'}
    top={n for n in inventory if '/' not in n}
    if top!=copies|{n+suffix for n in child_names for suffix in ('.log','_stage.json')}:
        raise ValueError('fixed mutation roster')
    from scripts.test_hz_source_portable import MATH_ERRORS
    mutation_calls={}
    for name in sorted(child_names):
        log=root/'mutations'/(name+'.log'); stage=load(root/'mutations'/(name+'_stage.json'))
        pairroot=root/'mutations'/name if name.startswith('math_') else root/'weighted_sign'/'relocated'
        anchors=stage['receipt']['anchors']; success=name in ('shadow','isolation')
        p.validate_stage(pairroot,anchors,stage,log,success=success)
        if name.startswith('math_'):
            p.preflight(pairroot,anchors,time.monotonic()+30)
            if log.read_text().splitlines()[-1]!='ValueError: '+MATH_ERRORS[name[5:]]:
                raise ValueError('specific mathematical refusal not observed')
        elif name=='expired':
            if log.read_text().splitlines()[-1]!='TimeoutError: portable shared deadline':
                raise ValueError('specific expired refusal not observed')
        elif name=='shadow':
            p.receive(pairroot,anchors,stage,log,time.monotonic()+30,p.reference('weighted_sign')[2])
        elif load(log)['blocked']!=['numpy','scipy','torch','local_check','proof_format',
                                   'act.back_end.moe.batched_support','outside_read','write']:
            raise ValueError('isolation observations')
        mutation_calls[name]={'status':stage['status'],'seconds':stage['seconds'],
                              'cleanup_seconds':stage['cleanup_seconds'],'stdout_sha256':sha(log)}
    reports={}
    for name,obs in calls.items():
        path=root/name; terminal=load(path/'terminal.json',obs['result']['terminal_sha256'])
        observed=load(path/'observed.json'); stage=load(path/'stage.json',terminal['stage_sha256'])
        if observed!=obs['result'] or load(root/('observed_'+name+'.json'))!=obs:
            raise ValueError('durable observed API identity')
        start,end=terminal['start'],obs['end']
        if (observed['status']!='CHECKED_OFFLINE_RELOCATION' or terminal['status']!=observed['status']
                or terminal['error'] is not None or obs['start']>start or end>=terminal['deadline']
                or terminal['deadline']!=start+p.protocol()['normal_budget_seconds']
                or observed['start']!=start or not start<=observed['end']<=end
                or observed['seconds']!=observed['end']-start
                or abs(observed['seconds']-terminal['seconds_before_publication']-observed['terminal_publication_seconds'])>1e-8):
            raise ValueError('complete observation/deadline/cost')
        costs=terminal['cost_seconds']
        if set(costs)!={'archive_read','publication','relocation','preflight_and_child','receipt'}:
            raise ValueError('whole offline cost inventory')
        if (any(not isinstance(t,(int,float)) or not math.isfinite(t) or t<0 for t in costs.values())
                or sum(costs.values())>terminal['seconds_before_publication']
                or stage['seconds']>costs['preflight_and_child']): raise ValueError('cost decomposition')
        checker=load(path/'checker.log',terminal['result_file_sha256'])
        if checker!=terminal['result']: raise ValueError('saved checker stdout binding')
        anchors=checker['anchors']; case=p.protocol()['cases'][name]
        if any(anchors[k]!=case[k] for k in ('source_sha256','package_sha256')):
            raise ValueError('registered data anchor')
        original=p.preflight(path/'original',anchors,time.monotonic()+30)
        relocated=p.preflight(path/'relocated',anchors,time.monotonic()+30)
        if original!=relocated: raise ValueError('relocation changed bytes')
        checked=p.receive(path/'relocated',anchors,stage,path/'checker.log',time.monotonic()+30,p.reference(name)[2],terminal['deadline'])
        reports[name]={'anchors':anchors,'mathematical_result':checked['result'],
                       'bundle_bytes':checked['bundle_bytes'],'observed_seconds':end-obs['start'],
                       'cost_seconds':costs,'stage_seconds':stage['seconds'],'cleanup_seconds':stage['cleanup_seconds'],
                       'sampled_peak_rss':stage['sampled_peak_rss'],'terminal_sha256':sha(path/'terminal.json'),
                       'check_stdout_sha256':sha(path/'checker.log'),
                       'final_publication_seconds':observed['terminal_publication_seconds'],
                       'post_record_observation_seconds':end-observed['end'],
                       'loaded_modules':checked['loaded_modules'],'bundle_reads':checked['bundle_reads']}
    return {'status':'PASS','root':str(root),'summary_sha256':sha(root/'summary.json'),
            'tests':len(names()),'objects':reports,'mutation_calls':mutation_calls,
            'mutation_inventory_sha256':summary['mutation_inventory_sha256'],
            'new_proposals':0,'native_solves':0,'cuda_calls':0,
            'real_requests':0,'offline_recheck_only':True,'performance_claim':False,
            'audit_reexecutes_mathematical_checker':False,'all_six_goal_gates_remain_open':True}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['run','audit'])
    parser.add_argument('root',type=Path); parser.add_argument('--report',type=Path); a=parser.parse_args()
    if not a.root.is_absolute() or not a.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new project archive required')
    result=run(a.root) if a.action=='run' else audit(a.root)
    if a.report: save(a.report,result)
    print({'status':result['status'],'tests':result['tests'],'root':str(a.root)})


if __name__=='__main__': main()
