"""Frozen block/flat controls and saved-source/dual independent recheck."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT,load,save,sha
from source_enclosure.format import identity
from scripts.run_hz_templates import FILES as OLD_FILES
from scripts.run_hz_lifted_source import SOURCES,check_cost

DESIGN='docs/hz_block_support_design_20261002.md'
DESIGN_SHA='1802dc745fe5f0fdc4a5d33053f74fd786139e1ed95c51a6de8f1e7b7842cdea'
FREEZE='54f364afb4c5e43902f9049d282d376d38332298'
FILES=sorted(set(OLD_FILES)|{DESIGN,'act/back_end/moe/check_block_support.py','act/back_end/moe/block_support.py',
    'act/back_end/moe/check_block_endpoints.py','act/back_end/moe/block_endpoints.py',
    'scoped_source/hz_block_source.py','scoped_source/check_hz_block_source.py',
    'scoped_source/block_differential.py','scoped_source/test_hz_blocks.py','scripts/run_hz_blocks.py'})
TESTS=sorted('test_'+s for s in ('complete_sources','numeric_differential','candidate_crosscheck','binary_mapping',
    'shared_relation','common_prefix','private_spans','zero_dual_rows','property_offset','missing_evidence',
    'candidate_mutation','source_gate','no_joint','fixed_algorithm','producer_faults','checker_expiry',
    'checker_independence','cost_inventory'))


def run(root):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=FREEZE or sha(ROOT/DESIGN)!=DESIGN_SHA:raise ValueError('frozen block execution identity')
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic();bindings={n:sha(ROOT/n) for n in FILES}
    for n in FILES:
        dest=root/'implementation'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    save(root/'implementation.json',bindings)
    from scoped_source import test_hz_blocks as tests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.BlockTests)
    names=['scoped_source.test_hz_blocks.BlockTests.'+n for n in TESTS]
    if [t.id() for t in suite]!=names:raise ValueError('fixed block test roster')
    with (root/'tests.log').open('x') as log:
        class Result(unittest.TextTestResult):
            outcomes=[]
            def startTest(self,t):self.state='PASS';self.start=time.monotonic();super().startTest(t)
            def addError(self,t,e):self.state='ERROR';super().addError(t,e)
            def addFailure(self,t,e):self.state='FAIL';super().addFailure(t,e)
            def addSkip(self,t,e):self.state='SKIP';super().addSkip(t,e)
            def addExpectedFailure(self,t,e):self.state='EXPECTED_FAILURE';super().addExpectedFailure(t,e)
            def addUnexpectedSuccess(self,t):self.state='UNEXPECTED_SUCCESS';super().addUnexpectedSuccess(t)
            def stopTest(self,t):
                self.outcomes.append({'test':t.id(),'status':self.state,'start':self.start,'end':time.monotonic()});super().stopTest(t)
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result,failfast=True).run(suite)
    save(root/'observations.json',tests.OBS);end=time.monotonic()
    success=result.wasSuccessful() and [(r['test'],r['status']) for r in result.outcomes]==[(n,'PASS') for n in names]
    summary={'status':'PASS' if success else 'FAIL','head':head,'design_sha256':DESIGN_SHA,'tests':result.testsRun,
             'outcomes':result.outcomes,'exceptions':{k:len(getattr(result,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')},
             'start':start,'end':end,'seconds':end-start,'implementation_sha256':sha(root/'implementation.json'),
             'observations_sha256':sha(root/'observations.json'),'tests_log_sha256':sha(root/'tests.log'),
             'real_requests':0,'native_solves':0,'cuda_calls':0,'hard_supervision':False,'performance_claim':False}
    save(root/'summary.json',summary);return summary


def check_inventory(obs):
    if (set(obs)!={'cases','given','negatives','auxiliary','diagnostics','partial','differentials'}
            or set(obs['cases'])!=set(SOURCES) or set(obs['given'])!={'private','relation'}
            or set(obs['differentials'])!=set(SOURCES) or set(obs['diagnostics'])!={'given','differential'}
            or set(obs['auxiliary'])!={'expired_producer','mutated_source'}):
        raise ValueError('complete block archive inventory')


def finite_times(item,lower,upper):
    if (any(type(item[k]) not in (float,int) or not math.isfinite(item[k]) for k in ('start','end'))
            or not lower<=item['start']<=item['end']<=upper):raise ValueError('finite ordered clock')


def audit(root):
    from scoped_source.endpoint_source_controls import cases
    from scoped_source.check_hz_templates import check as old_check
    from scoped_source.check_hz_block_source import check
    from scoped_source.block_differential import compare_sources
    from scoped_source.test_hz_blocks import negative_queries,recheck_negative,check_given
    s=load(root/'summary.json');binding=load(root/'implementation.json',s['implementation_sha256'])
    obs=load(root/'observations.json',s['observations_sha256'])
    names=['scoped_source.test_hz_blocks.BlockTests.'+n for n in TESTS]
    if (s['status']!='PASS' or s['head']!=FREEZE or s['design_sha256']!=DESIGN_SHA or binding[DESIGN]!=DESIGN_SHA
            or set(binding)!=set(FILES) or s['tests']!=18
            or [(r['test'],r['status']) for r in s['outcomes']]!=[(n,'PASS') for n in names]
            or s['exceptions']!={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}
            or sha(root/'tests.log')!=s['tests_log_sha256']
            or any(type(s[k]) is not int or s[k]!=0 for k in ('real_requests','native_solves','cuda_calls'))
            or s['hard_supervision'] is not False or s['performance_claim'] is not False
            or type(s['seconds']) not in (int,float) or not math.isfinite(s['seconds']) or s['seconds']!=s['end']-s['start']):
        raise ValueError('frozen block suite completion/scope')
    finite_times(s,s['start'],s['end'])
    for n,digest in binding.items():
        if sha(ROOT/n)!=digest or sha(root/'implementation'/n)!=digest:raise ValueError('executed implementation identity')
    check_inventory(obs);last=s['start'];expected={n:d for n,d,_ in cases()};results={};total=0
    for i,name in enumerate(SOURCES):
        arms=obs['cases'][name]
        if set(arms)!={'materialized','block'}:raise ValueError('two fresh arms required')
        for arm in (('materialized','block') if i%2==0 else ('block','materialized')):
            item=arms[arm];finite_times(item,last,s['end']);last=item['end'];check_cost(item);total+=item['cost']['total']
            anchor,duties,endpoints=SOURCES[name]
            if item['source']!=expected[name] or identity(item['source'])!=anchor:raise ValueError('frozen source')
            value=(check if arm=='block' else old_check)(item['source'],item['package'],expected_source_sha256=anchor,deadline=time.monotonic()+300)
            if (value!=item['checked'] or value['required']!=duties or value['checked_endpoints']!=endpoints
                    or value['missing_endpoints']!=0 or name=='unsafe_tied' and value['positive']!=0):
                raise ValueError('complete fresh source/output evidence')
            if item['dispatch']!=['router']+[f'expert{j}' for j in range(expected[name]['request']['experts'])]:
                raise ValueError('same template propagation inventory')
            counts=item['boundaries']
            if set(counts)!={'joint','remap','export','direct_export','old_source_check'} or any(type(v) is not int or v<0 for v in counts.values()):
                raise ValueError('construction boundary inventory')
            if arm=='block' and any(counts.values()):raise ValueError('block path constructed joint/export/old checker')
            if arm=='materialized' and (counts['joint']!=len(item['package']['pairs']) or counts['remap']<=0 or counts['export']<=0 or counts['old_source_check']!=1):
                raise ValueError('materialized reference execution')
        differential=compare_sources(arms['materialized']['package'],arms['block']['package'],time.monotonic()+300)
        if differential!=obs['differentials'][name]:raise ValueError('full domain/candidate differential changed')
        results[name]={'arms':{a:{'checked':v['checked'],'cost_seconds':v['cost'],'serialized_bytes':v['serialized_bytes'],
                                 'construction_boundaries':v['boundaries']} for a,v in arms.items()},'differential':differential}
    test_end=last
    for r in s['outcomes']:finite_times(r,test_end,s['end']);test_end=r['end']
    windows={r['test'].rsplit('.',1)[1]:r for r in s['outcomes']}
    finite_times(obs['diagnostics']['given'],last,s['outcomes'][0]['start'])
    w=windows['test_numeric_differential']
    finite_times(obs['diagnostics']['differential'],w['start'],w['end'])
    crosschecks=[v for rows in obs['differentials'].values() for p in rows for v in p['crosschecks']]
    if len(crosschecks)!=84:raise ValueError('complete 84 crosschecks')
    for name,item in obs['given'].items():
        check_given(name,item)
    parent=obs['cases']['tied_partial_reuse']['block'];p=deepcopy(parent['package']);p['proof']['pairs'][-1]['candidates']=None
    partial=obs['partial']
    if partial['source']!=parent['source'] or partial['package']!=p:raise ValueError('specific partial omission')
    pv=check(partial['source'],p,expected_source_sha256=identity(partial['source']),deadline=time.monotonic()+300)
    if pv!=partial['checked'] or pv['status']!='UNKNOWN_MISSING_EVIDENCE':raise ValueError('partial evidence acceptance')
    negatives=negative_queries(obs)
    if set(negatives)!=set(obs['negatives']):raise ValueError('specific mutation inventory')
    for key,spec in negatives.items():
        if recheck_negative(spec)!=obs['negatives'][key]:raise ValueError('negative input/actual refusal changed')
    for name in ('expired_producer','mutated_source'):
        item=obs['auxiliary'][name];expired=name=='expired_producer';c=item['cost']
        finite_times(item,last,s['end']);last=item['end']
        w=windows['test_producer_faults'];finite_times(item,w['start'],w['end'])
        changed=deepcopy(expected['weighted_sign'])
        if not expired:changed['request']['radius']='1/2'
        if (item['source_sha256']!=SOURCES['weighted_sign'][0] or item['observed_source_sha256']!=identity(changed)
                or set(c)!=({'source_creation','total'} if expired else {'source_creation','construction','proposals','total'})
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in c.values())
                or c['total']!=item['end']-item['start'] or sum(v for k,v in c.items() if k!='total')>c['total']
                or item['deadline']!=item['start']+(-1 if expired else 300) or not expired and item['end']>=item['deadline']
                or item['injection_reached'] is not (not expired) or item['actual_candidate_calls']!=0
                or item['dispatch']!=([] if expired else ['router','expert0','expert1','expert2'])
                or item['error_type']!=('TimeoutError' if expired else 'ValueError')
                or item['error']!=('source construction/check deadline' if expired else 'source changed during generation')):
            raise ValueError('auxiliary failure/complete cost')
        total+=c['total']
    if total>s['seconds']:raise ValueError('suite undercounts requests')
    return {'status':'PASS','issues':0,'summary_sha256':sha(root/'summary.json'),'freeze_head':FREEZE,'design_sha256':DESIGN_SHA,
            'tests':18,'source_packages_checked':9,'cases':results,'given_hz_controls':2,'same_candidate_crosschecks':84,
            'negative_inputs_rechecked':len(negatives),'partial':pv,'suite_seconds':s['seconds'],
            'auxiliary_failures':obs['auxiliary'],'real_requests':0,'native_solves':0,'cuda_calls':0,
            'hard_supervision':False,'performance_claim':False,'all_six_goal_gates_remain_open':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','audit']);p.add_argument('root',type=Path)
    p.add_argument('--report',type=Path);a=p.parse_args()
    if not a.root.is_absolute() or not a.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('project result directory required')
    result=run(a.root) if a.action=='run' else audit(a.root)
    if a.report:save(a.report,result)
    print({'status':result['status'],'tests':result['tests'],'root':str(a.root)})
    raise SystemExit(0 if result['status']=='PASS' else 1)
