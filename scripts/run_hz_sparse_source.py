"""Frozen finite sparse-source connection and checking-only archive audit.

CPU, cooperative budgets; not physical GPU, hard supervision, or a speed study.
The audit calls independent checkers, never a source/candidate producer or solver.
ACT transitive imports still require its environment; no stdlib portability claim.
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT, load, save, sha
from source_enclosure.format import identity
from scripts.run_hz_blocks import FILES as OLD_FILES
from scripts.run_hz_lifted_source import check_cost as old_cost
from scripts.run_hz_sparse_rows import expected_counts

DESIGN = 'docs/hz_sparse_source_design_20261012.md'
DESIGN_SHA = '82c3f17ee8a2ecf67c948cbf00e3fe0a086d729b00453452104147a94526549b'
FREEZE = '6e46b210e41dae8491e8fffe28b595f088e74331'
FILES = sorted(set(OLD_FILES) | {DESIGN, 'scoped_source/hz_sparse_source.py',
    'scoped_source/check_hz_sparse_source.py', 'scoped_source/hz_sparse_source_controls.py',
    'scoped_source/test_hz_sparse_source.py', 'scripts/run_hz_sparse_source.py',
    'scoped_source/hz_sparse_rows.py', 'scoped_source/check_hz_sparse_rows.py',
    'scoped_source/hz_sparse_row_fixtures.py', 'scripts/run_hz_sparse_rows.py'})
FIXED = {'scoped_source/hz_sparse_rows.py':'9f20754e0a103261c0292d90c735d02d9aa2e959cf47560c2c9b48d59ace8c99',
         'scoped_source/check_hz_sparse_rows.py':'4ed68696c73d1eb1c410ccc82091a9db3918a997c74c459ac0d1aa17475ab9f0',
         'configs/hz_source_representation_20261001.json':'507578e57caea2b2b92f332ce00da1c94d046572b36f5e30399d1f0365475738'}
EXCEPTIONS = ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')


def times(record, budget):
    values = [record[k] for k in ('start','end','deadline')]
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
            or not values[0] <= values[1] < values[2] or values[2] != values[0]+budget):
        raise ValueError('finite ordered cost/deadline inventory')
    return values[0],values[1]


def check_cost(record):
    times(record,300); old_cost(record)
    if type(record['serialized_bytes']) is not int: raise ValueError('integer serialized size')


def inventory(obs):
    from scoped_source import hz_sparse_source_controls as control
    if (set(obs) != {'cases','bridges','negatives','faults','capacity','partial','differentials','dense','independence'}
            or set(obs['cases']) != set(control.SOURCES) or set(obs['bridges']) != set(control.BRIDGES)
            or set(obs['faults']) != set(control.FAULTS) or set(obs['partial']) != set(control.ARMS)
            or set(obs['capacity']) != {'old_dense','global_factor','row_output'}
            or set(obs['differentials']) != set(control.SOURCES)):
        raise ValueError('complete frozen sparse-source inventory required')
    for entries in (obs['cases'],obs['bridges']):
        if any(set(arms) != set(control.ARMS) for arms in entries.values()):
            raise ValueError('both complete arms required')
    expected = control.negative_queries(obs)
    if set(expected) != set(obs['negatives']): raise ValueError('complete concrete mutation inventory')
    for name,spec in expected.items():
        if spec != obs['negatives'][name]['input']: raise ValueError('specific negative input: '+name)
    for name,r in obs['faults'].items():
        if (r['error_type'],r['error']) != control.ERRORS[name]: raise ValueError('specific fault reason: '+name)


def run(root):
    if root.name not in tuple(f'hz_sparse_source_20261012_r{i}' for i in (1,2,3)):
        raise ValueError('one of three finite repair-attempt directories required')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head != FREEZE or sha(ROOT/DESIGN) != DESIGN_SHA: raise ValueError('frozen execution identity')
    for name,digest in FIXED.items():
        if sha(ROOT/name) != digest: raise ValueError('frozen primitive/declaration changed')
    root.mkdir(parents=True,exist_ok=False); start = time.monotonic()
    bindings = {n:sha(ROOT/n) for n in FILES}
    for name in FILES:
        target = root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save(root/'implementation.json',bindings)
    importing = time.monotonic()
    from scoped_source import hz_sparse_source_controls as control
    from scoped_source import test_hz_sparse_source as tests
    imported = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.SparseSourceTests)
    names = ['scoped_source.test_hz_sparse_source.SparseSourceTests.'+n for n in control.TESTS]
    if [t.id() for t in suite] != names: raise ValueError('frozen twelve-test roster')
    with (root/'tests.log').open('x') as out:
        class Result(unittest.TextTestResult):
            def __init__(self,*a,**k): super().__init__(*a,**k); self.outcomes=[]
            def startTest(self,t): self.state='PASS'; self.start=time.monotonic(); super().startTest(t)
            def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
            def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
            def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
            def addExpectedFailure(self,t,e): self.state='EXPECTED_FAILURE'; super().addExpectedFailure(t,e)
            def addUnexpectedSuccess(self,t): self.state='UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
            def stopTest(self,t):
                self.outcomes.append({'test':t.id(),'status':self.state,'start':self.start,'end':time.monotonic()})
                super().stopTest(t)
        result = unittest.TextTestRunner(stream=out,verbosity=2,resultclass=Result,failfast=True).run(suite)
    ended = time.monotonic(); save(root/'observations.json',tests.OBS); published = time.monotonic()
    success = (result.wasSuccessful() and [(r['test'],r['status']) for r in result.outcomes] == [(n,'PASS') for n in names]
               and published-start < 1800)
    summary = {'status':'PASS' if success else 'FAIL', 'head':head,'design_sha256':DESIGN_SHA,
        'tests':result.testsRun,'outcomes':result.outcomes,
        'exceptions':{k:len(getattr(result,k)) for k in EXCEPTIONS},
        'implementation_sha256':sha(root/'implementation.json'),'observations_sha256':sha(root/'observations.json'),
        'tests_log_sha256':sha(root/'tests.log'),'times':[start,importing,imported,ended,published],
        'import_seconds':imported-importing,'suite_seconds':ended-imported,
        'archive_seconds':published-ended,'total_seconds':published-start,
        'real_requests':0,'native_solves':0,'cuda_calls':0,'hard_supervision':False,'performance_claim':False}
    save(root/'summary.json',summary); return summary


def refusal(name, record):
    """Reconstruct admission failure, without invoking the refused producer."""
    from scoped_source import check_hz_row_enclosure as global_check, check_hz_binary64 as local
    from scoped_source import check_hz_sparse_rows as sparse
    from scoped_source import hz_sparse_source_controls as control
    ref = control.dense_reference() if name=='old_dense' else control.capacity(name)
    if record['reference'] != ref: raise ValueError('fixed capacity source')
    try:
        if name=='old_dense': local.parse(global_check.projected(ref,'output',0)[0])
        elif name=='global_factor': global_check.parse(ref)
        else:
            w = sparse.Work(time.monotonic()+30)
            sparse.parse_owned(sparse.capture(ref,w,'capacity')[0],w)
    except ValueError as exc:
        if (record['error_type'],record['error']) != (type(exc).__name__,str(exc)):
            raise ValueError('specific capacity refusal changed')
    else: raise ValueError('admission refusal absent')


def faults(obs):
    from scoped_source import hz_sparse_source_controls as control
    parent = obs['cases']['weighted_sign']['sparse']
    phases = ('before_producer','before_checker','after_live_snapshot','after_construction','after_endpoints')
    for name,phase in zip(control.FAULTS,phases):
        r = obs['faults'][name]; doc = deepcopy(parent['source']); p = deepcopy(parent['package'])
        if name=='source_before_proposals': doc['request']['margin']='1'
        if name=='checker_late_package': p['source_sha256']='0'*64
        expected_clock = r['deadline']+1 if name in control.FAULTS[:3] else r['start']
        if (r['reached'] != [phase] or r['source_before'] != identity(parent['source'])
                or r['package_before'] != identity(parent['package']) or r['source_after'] != identity(doc)
                or r['package_after'] != identity(p) or type(r['proposal_calls']) is not int
                or r['proposal_calls'] != 0 or r['simulated_clock'] != expected_clock):
            raise ValueError('specific fault reach/identity/candidate inventory')


def audit(root):
    start = time.monotonic()
    from scoped_source import hz_sparse_source_controls as control
    from scoped_source import check_hz_sparse_source as new_check, check_hz_block_source as old_check
    s = load(root/'summary.json')
    binding = load(root/'implementation.json',s['implementation_sha256'])
    obs = load(root/'observations.json',s['observations_sha256'])
    names = ['scoped_source.test_hz_sparse_source.SparseSourceTests.'+n for n in control.TESTS]
    if (s['status'] != 'PASS' or s['tests'] != 12 or s['head'] != FREEZE or s['design_sha256'] != DESIGN_SHA
            or [(r['test'],r['status']) for r in s['outcomes']] != [(n,'PASS') for n in names]
            or s['exceptions'] != {k:0 for k in EXCEPTIONS} or set(binding) != set(FILES)
            or binding[DESIGN] != DESIGN_SHA or sha(root/'tests.log') != s['tests_log_sha256']
            or any(type(s[k]) is not int or s[k] != 0 for k in ('real_requests','native_solves','cuda_calls'))
            or s['hard_supervision'] is not False or s['performance_claim'] is not False):
        raise ValueError('frozen completion/scope')
    for name,digest in binding.items():
        if sha(ROOT/name) != digest or sha(root/'implementation'/name) != digest:
            raise ValueError('executed implementation binding: '+name)
    for name,digest in FIXED.items():
        if binding[name] != digest: raise ValueError('unchanged primitive/declaration')
    t = s['times']
    if (len(t) != 5 or any(type(v) not in (int,float) or not math.isfinite(v) for v in t)
            or any(a>b for a,b in zip(t,t[1:])) or t[-1]-t[0] >= 1800
            or [s[k] for k in ('import_seconds','suite_seconds','archive_seconds','total_seconds')]
               != [t[2]-t[1],t[3]-t[2],t[4]-t[3],t[4]-t[0]]):
        raise ValueError('suite cost envelope')
    inventory(obs); intervals=[]; results={}; last=t[2]
    for i,name in enumerate(control.SOURCES):
        anchor,duties,endpoints = control.SOURCES[name]; arms=obs['cases'][name]
        for mode in control.ARMS[::1 if i%2==0 else -1]:
            r=arms[mode]; a,b=times(r,300); intervals.append((a,b)); check_cost(r)
            if not last<=a<=b<=t[3]: raise ValueError('fixed arm order/complete charging')
            last=b
            if identity(r['source']) != anchor: raise ValueError('fixed declaration identity')
            checker = old_check if mode=='old' else new_check
            got=checker.check(r['source'],r['package'],expected_source_sha256=anchor,deadline=time.monotonic()+300)
            if (got != r['checked'] or (got['required'],got['checked_endpoints'],got['missing_endpoints']) != (duties,endpoints,0)
                    or name=='unsafe_tied' and got['positive'] != 0):
                raise ValueError('complete independent source/output recheck')
            if r['dispatch'] != ['router']+[f'expert{j}' for j in range(r['source']['request']['experts'])]:
                raise ValueError('once per expert dispatch')
            counts=r['boundaries']
            if (set(counts) != {'joint','remap','export','direct_export','old_projection'}
                    or any(type(v) is not int or v<0 for v in counts.values())
                    or any(v!=0 for k,v in counts.items() if k!='old_projection')
                    or (counts['old_projection']<=0 if mode=='old' else counts['old_projection']!=0)):
                raise ValueError('registered construction path instrumentation')
            if mode=='sparse':
                lifts=control.lift_records(r['package']); counts=[expected_counts(l['reference'],l['enclosure'],256) for l in lifts]
                expected={k:sum(c[k] for c in counts) for k in counts[0]}
                rows=sum(sum(len(l['reference']['state']['hz'][k]) for k in ('c','b','ub')) for l in lifts)
                if (got['sparse_core_work'] != expected or got['checked_sparse_lifts'] != len(lifts)
                        or got['checked_row_lifts'] != rows): raise ValueError('independent core visit/row counts')
        diff=control.differential(arms['old'],arms['sparse'])
        if diff != obs['differentials'][name]: raise ValueError('complete differential receipt')
        results[name]={'required':duties,'endpoints':endpoints,'differential':diff,
            'arms':{mode:{'status':r['checked']['status'],'positive':r['checked']['positive'],
                         'result_sha256':identity(r['checked']),'cost_seconds':r['cost'],
                         'serialized_bytes':r['serialized_bytes'],'boundaries':r['boundaries']}
                    for mode,r in arms.items()},'sparse_core_work':arms['sparse']['checked']['sparse_core_work']}
    for name in control.BRIDGES:
        arms=obs['bridges'][name]
        for mode in control.ARMS:
            r=arms[mode]; a,b=times(r,30); intervals.append((a,b))
            if not last<=a<=b<=t[3]: raise ValueError('operator bridge execution order')
            last=b
            if control.check_bridge(name,r['item'],mode,time.monotonic()+30) != r['checked']:
                raise ValueError('independent operator bridge')
        x,y=arms['old']['item']['lift'],arms['sparse']['item']['lift']
        if (x['reference']!=y['reference'] or x['snapshot']!=y['snapshot'] or x['enclosure']['target']!=y['enclosure']['target']
                or name!='guard' and arms['sparse']['checked']['added_continuous']<=0
                or name=='guard' and y['reference']['state']['hz']['ub']==y['enclosure']['target']['state']['hz']['ub']):
            raise ValueError('rounding bridge target/nonzero compensation')
    test_last=last
    for row in s['outcomes']:
        a,b=row['start'],row['end']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)) or not test_last<=a<=b<=t[3]:
            raise ValueError('ordered test cost')
        test_last=b
    windows={r['test'].rsplit('.',1)[1]:r for r in s['outcomes']}
    def auxiliary(r,test):
        a,b=times(r,30); w=windows[test]
        if not w['start']<=a<=b<=w['end']: raise ValueError('auxiliary cost window')
        intervals.append((a,b))
    for name,r in obs['negatives'].items():
        auxiliary(r,'test_05_sparse_mutations' if name.startswith('sparse_') else 'test_04_source_mutations')
        if control.recheck_negative(r['input'],time.monotonic()+30) != r['checked']: raise ValueError('specific negative recheck')
    for name,r in obs['capacity'].items(): auxiliary(r,'test_07_dense_capacity'); refusal(name,r)
    dense=obs['dense']; auxiliary(dense,'test_07_dense_capacity')
    if dense['reference'] != control.dense_reference() or dense['lift']['reference'] != dense['reference']:
        raise ValueError('fixed dense reference')
    got=new_check.lift(dense['lift'],'dense',time.monotonic()+30)[1]
    if got != dense['checked'] or got['added_continuous'] != 1: raise ValueError('dense row enclosure')
    for mode,r in obs['partial'].items():
        auxiliary(r,'test_08_partial_stale'); parent=obs['cases']['tied_partial_reuse'][mode]
        p=deepcopy(parent['package']); p['proof']['pairs'][-1]['candidates']=None
        if r['package'] != p: raise ValueError('specific missing candidates')
        got=(old_check if mode=='old' else new_check).check(parent['source'],p,
            expected_source_sha256=identity(parent['source']),deadline=time.monotonic()+300)
        if got!=r['checked'] or (got['status'],got['checked_endpoints'],got['missing_endpoints'])!=('UNKNOWN_MISSING_EVIDENCE',15,3):
            raise ValueError('partial must remain UNKNOWN')
    faults(obs)
    for name,r in obs['faults'].items():
        auxiliary(r,'test_09_deadlines' if name in control.FAULTS[:3] else 'test_10_identity_faults')
    auxiliary(obs['independence'],'test_11_independence')
    if obs['independence']['checked'] != obs['cases']['weighted_sign']['sparse']['checked']:
        raise ValueError('independence result binding')
    intervals.sort()
    if any(b>c for (_,b),(c,_) in zip(intervals,intervals[1:])) or sum(b-a for a,b in intervals)>s['suite_seconds']:
        raise ValueError('overlapping or uncharged control work')
    return {'status':'PASS','issues':0,'tests':12,'freeze_head':FREEZE,'design_sha256':DESIGN_SHA,
        'summary_sha256':sha(root/'summary.json'),'observations_sha256':s['observations_sha256'],
        'cases':results,'operator_bridges':{n:{a:r['checked'] for a,r in arms.items()} for n,arms in obs['bridges'].items()},
        'source_packages_rechecked':8,'source_endpoint_queries_per_arm':42,
        'negative_inputs_rechecked':len(obs['negatives']),'capacity_refusals_rechecked':3,
        'dense_row_accepted':True,'partial_endpoints_checked':15,'partial_endpoints_missing':3,
        'fault_receipts':obs['faults'],'suite_seconds':s['suite_seconds'],
        'import_seconds':s['import_seconds'],'archive_seconds':s['archive_seconds'],'total_seconds':s['total_seconds'],
        'independent_audit_seconds':time.monotonic()-start,'real_requests':0,'native_solves':0,'cuda_calls':0,
        'hard_supervision':False,'portable_stdlib_claim':False,'performance_claim':False,
        'all_six_goal_gates_remain_open':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run','audit']); p.add_argument('root',type=Path)
    p.add_argument('--report',type=Path); args=p.parse_args()
    if not args.root.is_absolute() or not args.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('project result directory required')
    result=run(args.root) if args.action=='run' else audit(args.root)
    if args.report: save(args.report,result)
    print({'status':result['status'],'tests':result['tests'],'root':str(args.root)})
    raise SystemExit(0 if result['status']=='PASS' else 1)
