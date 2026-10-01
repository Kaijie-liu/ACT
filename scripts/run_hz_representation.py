"""Frozen finite representation comparison and independent mathematical recheck."""
import argparse
from fractions import Fraction as F
import math
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT,load,save,sha

CONFIG='configs/hz_source_representation_20261001.json'
CONFIG_SHA='507578e57caea2b2b92f332ce00da1c94d046572b36f5e30399d1f0365475738'


def protocol(): return load(ROOT/CONFIG,CONFIG_SHA)


def files():
    from scripts.run_hz_source_controls import FILES
    return list(dict.fromkeys(FILES+[CONFIG,'docs/hz_source_representation_design_20261001.md',
        'scoped_source/hz_representation.py','scoped_source/check_hz_representation.py',
        'scoped_source/test_hz_representation.py','scripts/run_hz_representation.py',
        'act/back_end/solver/rational_mccormick.py','act/back_end/solver/check_rational_mccormick.py',
        'act/back_end/solver/sparse_lp_certificate.py','act/back_end/solver/lp_certificate.py',
        'scoped_source/rowwise_native.py','scoped_proof/io.py']))


def names():
    return sorted('scoped_source.test_hz_representation.RepresentationTests.test_'+n for n in protocol()['controls'])


def check_cost(item):
    keys={'source_creation','propagation_and_joint','representation','proposals','serialization','checking','total'}
    c=item['cost_seconds']
    if (set(c)!=keys or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in c.values())
            or item.get('error') or item['end']>=item['deadline'] or item['deadline']!=item['start']+300
            or c['total']!=item['end']-item['start'] or sum(v for k,v in c.items() if k!='total')>c['total']):
        raise ValueError('complete finite cost/deadline inventory')


def compare(obs):
    from source_enclosure.format import identity
    cfg=protocol()
    if set(obs)!={n+'/'+m for n in cfg['cases'] for m in cfg['arms']}:
        raise ValueError('eight-arm evidence inventory')
    out={}
    for name,spec in cfg['cases'].items():
        a,b=[obs[name+'/'+m] for m in cfg['arms']]
        if (identity(a['source'])!=spec['source_sha256'] or a['source']!=b['source']
                or a['package']['lowering']!=b['package']['lowering']
                or a['package']['mode']!='endpoints' or b['package']['mode']!='mccormick'):
            raise ValueError('same source/domain/gate/property comparison binding')
        for mode,item in zip(cfg['arms'],(a,b)):
            check_cost(item); r=item['checked']
            expected=spec['endpoints'] if mode=='endpoints' else spec['duties']
            if (r['mode']!=mode or r['required']!=spec['duties'] or r['checked_targets']!=expected
                    or r['missing_targets']!=0 or item['case']!=name or item['mode']!=mode):
                raise ValueError('all finite obligations retained')
        def positives(item): return {(tuple(r['pair']),r['property']) for r in item['checked']['results'] if r['positive']}
        pa,pb=positives(a),positives(b)
        pack=lambda rows:[{'pair':list(pair),'property':prop} for pair,prop in sorted(rows)]
        out[name]={'source_sha256':spec['source_sha256'],'same_checked_lowering_sha256':identity(a['package']['lowering']),
                   'endpoints_positive':len(pa),'mccormick_positive':len(pb),'required':spec['duties'],
                   'gained_properties':pack(pa-pb),'lost_properties':pack(pb-pa),
                   'endpoint_request_positive':len(pa)==spec['duties'],'mc_request_positive':len(pb)==spec['duties'],
                   'cost_seconds':{m:item['cost_seconds'] for m,item in zip(cfg['arms'],(a,b))}}
    return out


def diagnostic(obs,deadline):
    from scoped_source.hz_representation import registered_point
    from scoped_source.check_hz_representation import check,check_point,THRESHOLD
    from source_enclosure.format import identity
    item=obs['weighted_sign/mccormick']; package=item['package']
    # Source and every MC row are checked before this proposed point is tested.
    check(item['source'],package,expected_source_sha256=identity(item['source']),expected_mode='mccormick',deadline=deadline)
    candidate=registered_point(package); duty=package['duties'][0]
    if (duty['pair'],duty['property'])!=([0,1],'class1'): raise ValueError('registered diagnostic obligation')
    result={**candidate,'pair':duty['pair'],'property':duty['property'],'lp_sha256':identity(duty['construction']['lp']),
            'network_counterexample':False,'lp_optimum_claimed':False}
    try: value=check_point(duty['construction']['lp'],candidate['point'],deadline)
    except ValueError as exc:
        return dict(result,status='REGISTERED_POINT_NOT_FEASIBLE',error=str(exc),representation_separation=False)
    endpoint=next(r for r in obs['weighted_sign/endpoints']['checked']['results'] if r['pair']==[0,1] and r['property']=='class1')
    lower=F(endpoint['lower_bound'])
    return dict(result,status='EXACT_FEASIBLE',checked_objective=str(value),endpoint_lower_bound=str(lower),
                representation_separation=lower>THRESHOLD and value<=THRESHOLD)


def run(root):
    root.mkdir(parents=True,exist_ok=False); cfg=protocol(); start=time.monotonic()
    bindings={n:sha(ROOT/n) for n in files()}
    for n in bindings:
        target=root/'implementation'/n; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/n,target)
    save(root/'implementation.json',bindings)
    before=time.monotonic()
    from scoped_source import test_hz_representation as tests
    imports=time.monotonic()-before
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.RepresentationTests)
    if [t.id() for t in suite]!=names(): raise ValueError('fixed test inventory')
    with (root/'tests.log').open('x') as stream:
        class Result(unittest.TextTestResult):
            outcomes=[]
            def startTest(self,t): self.state='PASS'; super().startTest(t)
            def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
            def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
            def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
            def addExpectedFailure(self,t,e): self.state='EXPECTED_FAILURE'; super().addExpectedFailure(t,e)
            def addUnexpectedSuccess(self,t): self.state='UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
            def stopTest(self,t): self.outcomes.append({'test':t.id(),'status':self.state}); super().stopTest(t)
        r=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result,failfast=True).run(suite)
    save(root/'observations.json',tests.OBS)
    ok=r.wasSuccessful() and r.outcomes==[{'test':n,'status':'PASS'} for n in names()]
    summary={'status':'PASS' if ok else 'FAIL','tests':r.testsRun,'outcomes':r.outcomes,
             'exceptional_tests':{k:len(getattr(r,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')},
             'protocol_sha256':CONFIG_SHA,'implementation_sha256':sha(root/'implementation.json'),
             'observations_sha256':sha(root/'observations.json'),'tests_log_sha256':sha(root/'tests.log'),
             'seconds':time.monotonic()-start,'imports_seconds':imports,
             'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'native_solves':0,'cuda_calls':0,'real_requests':0,'hard_budget_supervision':False,'performance_claim':False}
    save(root/'summary.json',summary); return summary


def audit(root):
    from scoped_source.check_hz_representation import check
    from source_enclosure.format import identity
    s=load(root/'summary.json'); bindings=load(root/'implementation.json',s['implementation_sha256'])
    obs=load(root/'observations.json',s['observations_sha256'])
    if (s['status']!='PASS' or s['tests']!=len(names()) or s['outcomes']!=[{'test':n,'status':'PASS'} for n in names()]
            or s['exceptional_tests']!={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}
            or s['protocol_sha256']!=CONFIG_SHA or s['tests_log_sha256']!=sha(root/'tests.log') or set(bindings)!=set(files())
            or any(type(s[k]) is not int or s[k]!=0 for k in ('native_solves','cuda_calls','real_requests'))
            or s['hard_budget_supervision'] is not False or s['performance_claim'] is not False):
        raise ValueError('archive scope/test/identity inventory')
    for n,h in bindings.items():
        if sha(ROOT/n)!=h or sha(root/'implementation'/n)!=h: raise ValueError('implementation changed: '+n)
    for item in obs.values():
        result=check(item['source'],item['package'],expected_source_sha256=identity(item['source']),
                     expected_mode=item['mode'],deadline=time.monotonic()+300)
        if result!=item['checked']: raise ValueError('saved result changed')
    return {'status':'PASS','root':str(root),'summary_sha256':sha(root/'summary.json'),'tests':len(names()),
            'cases':compare(obs),'diagnostic':diagnostic(obs,time.monotonic()+300),'packages_rechecked':len(obs),
            'endpoint_targets':42,'mc_targets':33,'native_solves':0,'cuda_calls':0,'real_requests':0,
            'hard_budget_supervision':False,'performance_claim':False,'all_six_goal_gates_remain_open':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['run','audit']); parser.add_argument('root',type=Path)
    parser.add_argument('--report',type=Path); a=parser.parse_args()
    if not a.root.is_absolute() or not a.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new project result directory required')
    result=run(a.root) if a.action=='run' else audit(a.root)
    if a.report: save(a.report,result)
    print({'status':result['status'],'tests':result['tests'],'root':str(a.root)})
