"""Frozen finite source-template differential archive and independent recheck."""
import argparse
from copy import deepcopy
from pathlib import Path
import math
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT, load, save, sha
from source_enclosure.format import identity
from scripts.run_hz_lifted_source import FILES as OLD_FILES, SOURCES, check_cost

DESIGN = 'docs/hz_expert_template_design_20261001.md'
DESIGN_SHA = '619a878b1f943c429676533a4c7e016a09ceff40334d838c868ce9787f9ec9f4'
FREEZE = '1aed9be952d99eec903437d4e49ab9ccfe4a4733'
FILES = sorted(set(OLD_FILES)|{DESIGN,'scoped_source/hz_templates.py',
    'scoped_source/check_hz_templates.py','scoped_source/test_hz_templates.py','scripts/run_hz_templates.py'})
TESTS = sorted('test_'+s for s in ('complete_sources','numeric_differential','propagation_counts',
    'private_identity','common_entry','template_inventory','template_selection','guard','private_pollution',
    'gate_property','missing_pair','partial','stale','operator_bridges','guard_dependent_range',
    'deadline_mutation','checker_independence','cost_inventory'))


def run(root):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=FREEZE or sha(ROOT/DESIGN)!=DESIGN_SHA: raise ValueError('separate frozen execution required')
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic();bindings={n:sha(ROOT/n) for n in FILES}
    for n in FILES:
        dest=root/'implementation'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    save(root/'implementation.json',bindings)
    from scoped_source import test_hz_templates as tests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.TemplateTests)
    names=['scoped_source.test_hz_templates.TemplateTests.'+n for n in TESTS]
    if [t.id() for t in suite]!=names: raise ValueError('frozen test roster')
    with (root/'tests.log').open('x') as log:
        class Result(unittest.TextTestResult):
            outcomes=[]
            def startTest(self,t):self.state='PASS';super().startTest(t)
            def addError(self,t,e):self.state='ERROR';super().addError(t,e)
            def addFailure(self,t,e):self.state='FAIL';super().addFailure(t,e)
            def addSkip(self,t,e):self.state='SKIP';super().addSkip(t,e)
            def addExpectedFailure(self,t,e):self.state='EXPECTED_FAILURE';super().addExpectedFailure(t,e)
            def addUnexpectedSuccess(self,t):self.state='UNEXPECTED_SUCCESS';super().addUnexpectedSuccess(t)
            def stopTest(self,t):self.outcomes.append({'test':t.id(),'status':self.state});super().stopTest(t)
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result,failfast=True).run(suite)
    save(root/'observations.json',tests.OBS)
    success=result.wasSuccessful() and result.outcomes==[{'test':n,'status':'PASS'} for n in names]
    end=time.monotonic()
    summary={'status':'PASS' if success else 'FAIL','head':head,'design_sha256':DESIGN_SHA,
        'tests':result.testsRun,'outcomes':result.outcomes,
        'exceptions':{k:len(getattr(result,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')},
        'start':start,'end':end,'seconds':end-start,'implementation_sha256':sha(root/'implementation.json'),
        'observations_sha256':sha(root/'observations.json'),'tests_log_sha256':sha(root/'tests.log'),
        'real_requests':0,'native_solves':0,'cuda_calls':0,'hard_supervision':False,'performance_claim':False}
    save(root/'summary.json',summary);return summary


def audit(root):
    from scoped_source.endpoint_source_controls import cases
    from scoped_source.test_hz_templates import numeric_comparison,dispatch_roster,check_view_bridge,negative_queries,recheck_negative
    from scoped_source.check_hz_templates import check
    from scoped_source.check_hz_lifted_source import check as old_check
    s=load(root/'summary.json');binding=load(root/'implementation.json',s['implementation_sha256'])
    obs=load(root/'observations.json',s['observations_sha256'])
    names=['scoped_source.test_hz_templates.TemplateTests.'+n for n in TESTS]
    if (s['status']!='PASS' or s['head']!=FREEZE or s['design_sha256']!=DESIGN_SHA
            or binding[DESIGN]!=DESIGN_SHA or set(binding)!=set(FILES)
            or s['tests']!=18 or s['outcomes']!=[{'test':n,'status':'PASS'} for n in names]
            or s['exceptions']!={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}
            or s['tests_log_sha256']!=sha(root/'tests.log')
            or any(type(s[k]) is not int or s[k]!=0 for k in ('real_requests','native_solves','cuda_calls'))
            or s['hard_supervision'] is not False or s['performance_claim'] is not False
            or any(type(s[k]) not in (int,float) or not math.isfinite(s[k]) for k in ('start','end','seconds'))
            or not s['end']>=s['start'] or s['seconds']!=s['end']-s['start']):
        raise ValueError('frozen suite completion/scope')
    for n,digest in binding.items():
        if sha(ROOT/n)!=digest or sha(root/'implementation'/n)!=digest:raise ValueError('executed implementation identity')
    if (set(obs)!={'cases','bridges','partial','negatives','differentials','auxiliary'} or set(obs['cases'])!=set(SOURCES)
            or set(obs['differentials'])!=set(SOURCES) or set(obs['bridges'])!={'affine','relu','guard'}):
        raise ValueError('complete saved inventory')
    expected={n:d for n,d,_ in cases()};results={};total=0;last=s['start']
    for i,name in enumerate(SOURCES):
        for arm in (('per_pair','template') if i%2==0 else ('template','per_pair')):
            item=obs['cases'][name][arm]
            if not last<=item['start']<=item['end']<=s['end']:raise ValueError('frozen arm order/overlap')
            last=item['end']
    for name,(anchor,duties,endpoints) in SOURCES.items():
        arms=obs['cases'][name];e=expected[name]['request']['experts']
        if set(arms)!={'per_pair','template'}:raise ValueError('two fresh arms required')
        rows={}
        for arm,item in arms.items():
            doc=item['source']
            if doc!=expected[name] or identity(doc)!=anchor:raise ValueError('frozen declaration changed')
            checker=check if arm=='template' else old_check
            value=checker(doc,item['package'],expected_source_sha256=anchor,deadline=time.monotonic()+300)
            if (value!=item['checked'] or value['required']!=duties or value['checked_endpoints']!=endpoints
                    or value['missing_endpoints']!=0 or name=='unsafe_tied' and value['positive']!=0):
                raise ValueError('complete fresh source/endpoint check')
            check_cost(item);total+=item['cost']['total']
            if item['dispatch']!=dispatch_roster(e,arm):raise ValueError('observed propagation dispatch inventory')
            rows[arm]={'checked':value,'cost_seconds':item['cost'],'serialized_bytes':item['serialized_bytes'],
                       'expert_propagation_calls':len(item['dispatch'])-1}
        differential=numeric_comparison(arms['per_pair']['package'],arms['template']['package'])
        if differential!=obs['differentials'][name]:raise ValueError('full numerical pair differential')
        results[name]={'arms':rows,'pair_differentials':differential}
    aux=obs['auxiliary']
    if set(aux)!={'expired_producer','mutated_source'}:raise ValueError('auxiliary call inventory')
    for name in ('expired_producer','mutated_source'):
        item=aux[name];expired=name=='expired_producer';c=item['cost']
        if (item['source_sha256']!=SOURCES['weighted_sign'][0]
                or set(c)!=({'source_creation','total'} if expired else {'source_creation','construction','proposals','total'})
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in c.values())
                or not last<=item['start']<=item['end']<=s['end'] or c['total']!=item['end']-item['start']
                or sum(v for k,v in c.items() if k!='total')>c['total']
                or item['deadline']!=item['start']+(-1 if expired else 300)
                or not expired and item['end']>=item['deadline']
                or item['injection_reached'] is not (not expired) or item['actual_candidate_calls']!=0
                or item['dispatch']!=([] if expired else dispatch_roster(3,'template'))
                or item['error_type']!=('TimeoutError' if expired else 'ValueError')
                or item['error']!=('source construction/check deadline' if expired else 'source changed during generation')):
            raise ValueError('auxiliary failure reach/cost inventory')
        changed=deepcopy(expected['weighted_sign'])
        if not expired:changed['request']['radius']='1/2'
        if item['observed_source_sha256']!=identity(changed):raise ValueError('auxiliary mutation identity')
        total+=c['total'];last=item['end']
    if total>s['seconds']:raise ValueError('suite cost undercount')
    parent=obs['cases']['tied_partial_reuse']['template'];p=deepcopy(parent['package'])
    p['proof']['pairs'][-1]['candidates']=None;partial=obs['partial']
    if partial['source']!=parent['source'] or partial['package']!=p:raise ValueError('fixed partial omission')
    checked=check(partial['source'],p,expected_source_sha256=identity(partial['source']),deadline=time.monotonic()+300)
    if checked!=partial['checked'] or checked['status']!='UNKNOWN_MISSING_EVIDENCE':raise ValueError('partial acceptance')
    for name,item in obs['bridges'].items():
        if check_view_bridge(name,item,time.monotonic()+300)!=item['checked']:raise ValueError('operator view check')
    negatives=negative_queries(obs)
    if set(obs['negatives'])!=set(negatives):raise ValueError('fixed negative input roster')
    for key,spec in negatives.items():
        if recheck_negative(spec)!=obs['negatives'][key]:raise ValueError('negative input/actual refusal changed')
    return {'status':'PASS','issues':0,'summary_sha256':sha(root/'summary.json'),'freeze_head':FREEZE,
            'design_sha256':DESIGN_SHA,'tests':18,'source_packages_checked':9,'cases':results,'partial':checked,
            'operator_bridges_checked':3,'negative_inputs_rechecked':len(negatives),'suite_seconds':s['seconds'],
            'auxiliary_failures':aux,
            'real_requests':0,'native_solves':0,'cuda_calls':0,'hard_supervision':False,'performance_claim':False,
            'all_six_goal_gates_remain_open':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','audit']);p.add_argument('root',type=Path)
    p.add_argument('--report',type=Path);a=p.parse_args()
    if not a.root.is_absolute() or not a.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('project result directory required')
    result=run(a.root) if a.action=='run' else audit(a.root)
    if a.report:save(a.report,result)
    print({'status':result['status'],'tests':result['tests'],'root':str(a.root)})
    raise SystemExit(0 if result['status']=='PASS' else 1)
