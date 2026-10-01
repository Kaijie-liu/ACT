"""Finite row-lift/source control archive; never a real-request admission."""
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
from scripts.run_hz_source_controls import FILES as SOURCE_FILES

DESIGN='docs/hz_lifted_source_design_20261001.md'
DESIGN_SHA='07a2c3dbd48aea46e75fd34b0a3e2d84ff09e3a33216edabd8c22b91533b5573'
FREEZE='11a1bd9972a1f96bc347ca43b4b2721cb7234e82'
REFERENCE_CONFIG='configs/hz_source_representation_20261001.json'
REFERENCE_CONFIG_SHA='507578e57caea2b2b92f332ce00da1c94d046572b36f5e30399d1f0365475738'
SOURCES={
 'weighted_sign':('b50e6ac79b92ac1d25e6a2ad8d83d1958353e90eeba869663e3d9d7c9eff0de2',3,6),
 'tied_partial_reuse':('de8ddb321d0d7016ea9d61e8898b9c72a0fb66833aa6cdb28adfb598bf221a5f',18,18),
 'unsafe_tied':('36898a6957fe529ad5b43306c721f9d51b077f05ee5f71c8f9fd6de49e238c1e',6,6),
 'unresolved_sign':('b01dded09de8c72f4350e50ee9ed4a924f6cbfe2eb34821915e4a3e58465a1de',6,12)}
TESTS=sorted('test_'+s for s in ('row_kernel_differential','shared_global_identity','row_binding_mutations',
 'row_capacity_refusal','complete_four_sources','source_reference_mutations','guard_ownership','layer_inventory',
 'gate_property','missing_pair','partial_evidence','stale_evidence','rounding_affine','rounding_relu','rounding_guard',
 'expiry_mutation','checker_independence','cost_inventory'))
FILES=sorted(set(SOURCE_FILES)|{DESIGN,'scoped_source/check_hz_binary64.py','scoped_source/hz_binary64.py',
 'scoped_source/test_hz_binary64.py','scoped_source/check_hz_row_enclosure.py','scoped_source/hz_row_enclosure.py',
 'scoped_source/hz_lifted_source.py','scoped_source/check_hz_lifted_source.py',
 'scoped_source/test_hz_lifted_source.py','scripts/run_hz_lifted_source.py','scoped_proof/io.py',
 'act/back_end/solver/lp_certificate.py','scoped_source/rowwise_native.py',REFERENCE_CONFIG})


def check_cost(item):
    c=item['cost']
    if (set(c)!={'source_creation','construction','proposals','serialization','checking','total'}
            or any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in c.values())
            or item['deadline']!=item['start']+300 or not item['start']<=item['end']<item['deadline']
            or c['total']!=item['end']-item['start'] or sum(v for k,v in c.items() if k!='total')>c['total']):
        raise ValueError('complete cooperative cost/deadline inventory')
    import json
    if len(json.dumps(item['package'],sort_keys=True,allow_nan=False).encode())!=item['serialized_bytes']:
        raise ValueError('serialization byte cost')


def run(root):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=FREEZE or sha(ROOT/DESIGN)!=DESIGN_SHA or sha(ROOT/REFERENCE_CONFIG)!=REFERENCE_CONFIG_SHA:
        raise ValueError('separate frozen execution required')
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic();bindings={n:sha(ROOT/n) for n in FILES}
    for n in FILES:
        p=root/'implementation'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    save(root/'implementation.json',bindings)
    from scoped_source import test_hz_lifted_source as tests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.LiftedSourceTests)
    names=['scoped_source.test_hz_lifted_source.LiftedSourceTests.'+n for n in TESTS]
    if [t.id() for t in suite]!=names:raise ValueError('fixed control inventory')
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
    ok=result.wasSuccessful() and result.outcomes==[{'test':n,'status':'PASS'} for n in names]
    summary={'status':'PASS' if ok else 'FAIL','tests':result.testsRun,'outcomes':result.outcomes,
             'exceptions':{k:len(getattr(result,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')},
             'head':head,'design_sha256':DESIGN_SHA,'seconds':time.monotonic()-start,
             'implementation_sha256':sha(root/'implementation.json'),'observations_sha256':sha(root/'observations.json'),
             'tests_log_sha256':sha(root/'tests.log'),'real_requests':0,'native_solves':0,'cuda_calls':0,
             'hard_supervision':False,'performance_claim':False}
    save(root/'summary.json',summary);return summary


def audit(root):
    from scoped_source.endpoint_source_controls import cases
    from scoped_source.test_hz_binary64 import patterns
    from scoped_source.test_hz_lifted_source import bridge_sources,check_bridge,sparse_reference,fixed_refusals,recheck_refusal
    from scoped_source.check_hz_binary64 import check as kernel_check
    from scoped_source.check_hz_row_enclosure import check as row_check
    from scoped_source.check_hz_lifted_source import check
    s=load(root/'summary.json');bindings=load(root/'implementation.json',s['implementation_sha256'])
    obs=load(root/'observations.json',s['observations_sha256'])
    names=['scoped_source.test_hz_lifted_source.LiftedSourceTests.'+n for n in TESTS]
    if (s['status']!='PASS' or s['tests']!=len(names) or s['outcomes']!=[{'test':n,'status':'PASS'} for n in names]
            or s['exceptions']!={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}
            or s['head']!=FREEZE or s['design_sha256']!=DESIGN_SHA or bindings[DESIGN]!=DESIGN_SHA
            or set(bindings)!=set(FILES) or bindings[REFERENCE_CONFIG]!=REFERENCE_CONFIG_SHA
            or sha(root/'tests.log')!=s['tests_log_sha256']
            or any(type(s[k]) is not int or s[k]!=0 for k in ('real_requests','native_solves','cuda_calls'))
            or s['hard_supervision'] is not False or s['performance_claim'] is not False
            or type(s['seconds']) not in (float,int) or not math.isfinite(s['seconds']) or s['seconds']<0):
        raise ValueError('strict finite execution inventory')
    for n,digest in bindings.items():
        if sha(ROOT/n)!=digest or sha(root/'implementation'/n)!=digest:raise ValueError('source/checker execution identity')
    if set(obs)!={'sources','rows','bridges','sparse_global','partial','refusals'} or set(obs['sources'])!=set(SOURCES):
        raise ValueError('complete saved case inventory')
    expected={n:d for n,d,_ in cases()};results={}
    for name,(source_hash,duties,endpoints) in SOURCES.items():
        r=obs['sources'][name];doc=expected[name]
        if r['source']!=doc or identity(doc)!=source_hash:raise ValueError('fixed declaration identity')
        checked=check(doc,r['package'],expected_source_sha256=source_hash,deadline=time.monotonic()+300)
        if (checked!=r['checked'] or checked['required']!=duties or checked['checked_endpoints']!=endpoints
                or checked['missing_endpoints']!=0 or name=='unsafe_tied' and checked['positive']!=0):
            raise ValueError('complete fresh source/endpoint record')
        check_cost(r)
        results[name]={**checked,'cost_seconds':r['cost'],'serialized_bytes':r['serialized_bytes']}
    if sum(r['cost']['total'] for r in obs['sources'].values())>s['seconds']:raise ValueError('suite cost undercount')
    parent=obs['sources']['tied_partial_reuse'];p=deepcopy(parent['package']);p['proof']['pairs'][-1]['candidates']=None
    item=obs['partial']
    if item['source']!=parent['source'] or item['package']!=p:raise ValueError('registered partial omission')
    partial=check(item['source'],p,expected_source_sha256=identity(item['source']),deadline=time.monotonic()+300)
    if partial!=item['checked'] or partial['status']!='UNKNOWN_MISSING_EVIDENCE':raise ValueError('partial evidence status')
    references=patterns()
    if set(obs['rows'])!=set(references) or set(obs['bridges'])!={'affine','relu','guard'}:raise ValueError('row/bridge inventory')
    for name,(ref,_) in references.items():
        item=obs['rows'][name]
        if item['reference']!=ref:raise ValueError('row fixed reference')
        result=row_check(ref,item['proof'],expected_reference_sha256=identity(ref),expected_owner='pair/0-1',deadline=time.monotonic()+300)
        kernel_check(ref,item['kernel_proof'],expected_reference_sha256=identity(ref),expected_owner='pair/0-1',deadline=time.monotonic()+300)
        if result!=item['checked'] or item['proof']['target']!=item['kernel_proof']['target']:raise ValueError('row/kernel exact differential')
    global_row=obs['sparse_global'];ref=sparse_reference()
    if global_row['reference']!=ref:raise ValueError('sparse global source')
    if row_check(ref,global_row['proof'],expected_reference_sha256=identity(ref),expected_owner='global',deadline=time.monotonic()+300)!=global_row['checked']:
        raise ValueError('global row assembly check')
    bridge_results={}
    for name,item in obs['bridges'].items():
        if any(item.get(k)!=v for k,v in bridge_sources(name).items()):raise ValueError('fixed operator bridge source')
        result=check_bridge(name,item,time.monotonic()+300)
        if result!=item['checked']:raise ValueError('exact operator reference/enclosure bridge')
        bridge_results[name]=result
    negatives=fixed_refusals(obs)
    if set(obs['refusals'])!=set(negatives):raise ValueError('specific negative input roster')
    for key,spec in negatives.items():
        if recheck_refusal(spec)!=obs['refusals'][key]:raise ValueError('fixed negative input or refusal reason changed')
    return {'status':'PASS','issues':0,'summary_sha256':sha(root/'summary.json'),'freeze_head':FREEZE,
            'design_sha256':DESIGN_SHA,'tests':len(TESTS),'source_packages_checked':5,'cases':results,
            'partial':partial,'kernel_differentials':8,'sparse_global':global_row['checked'],'bridges':bridge_results,
            'negative_inputs_rechecked':len(negatives),'suite_seconds':s['seconds'],'real_requests':0,'native_solves':0,'cuda_calls':0,
            'hard_supervision':False,'performance_claim':False,'all_six_goal_gates_remain_open':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','audit']);p.add_argument('root',type=Path)
    p.add_argument('--report',type=Path);a=p.parse_args()
    if not a.root.is_absolute() or not a.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):raise ValueError('project result directory')
    result=run(a.root) if a.action=='run' else audit(a.root)
    if a.report:save(a.report,result)
    print({'status':result['status'],'tests':result['tests'],'root':str(a.root)})
    raise SystemExit(0 if result['status']=='PASS' else 1)
