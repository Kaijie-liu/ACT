"""Fixed CPU endpoint controls and solver-free recheck of saved obligations."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='configs/hz_endpoint_controls_20261001.json'
PROTOCOL_SHA='5bd0e784c123cab4a89354e4208eaaf01a10c19580790f9d8c55cddfa5d61201'
FILES=[PROTOCOL,'docs/hz_endpoint_design_20261001.md',
       'act/back_end/moe/hz_endpoints.py','act/back_end/moe/check_hz_endpoints.py',
       'act/back_end/moe/test_hz_endpoints.py','scripts/run_hz_endpoint_controls.py',
       'act/back_end/moe/weighted_top2.py','act/back_end/moe/batched_support.py',
       'act/back_end/moe/check_batched_support.py','act/back_end/solver/solver_hz.py',
       'act/back_end/solver/hz_lp_export.py','act/back_end/solver/check_hz_lp_export.py',
       'act/back_end/solver/lp_certificate.py','scoped_source/rowwise_bound.py',
       'scoped_source/rowwise_native.py','scoped_source/endpoint_controls.py',
       'scoped_source/endpoint_check.py','upstream_source/checker.py','source_enclosure/format.py']
OBS={'separation_projected','separation_analytic','route_witnesses','mc_negative_point',
     'shared_relation','independent_relation','private_factors','multiclass','equal_gate',
     'unit_gate','orientation_a','orientation_b','nonpositive','partial'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())
def save(path,value):
    with path.open('x') as f: json.dump(value,f,sort_keys=True,separators=(',',':'),allow_nan=False); f.write('\n')


def names():
    if sha(ROOT/PROTOCOL)!=PROTOCOL_SHA: raise ValueError('frozen endpoint protocol changed')
    return sorted('act.back_end.moe.test_hz_endpoints.HZEndpointTests.test_'+n for n in load(ROOT/PROTOCOL)['controls'])


def run(root):
    required=names(); root.mkdir(parents=True,exist_ok=False); bindings={}
    for name in FILES:
        dest=root/'implementation'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest); bindings[name]=sha(dest)
    save(root/'implementation.json',bindings)
    from act.back_end.moe.test_hz_endpoints import HZEndpointTests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(HZEndpointTests)
    if [t.id() for t in suite]!=required: raise ValueError('fixed control inventory')
    begin=time.monotonic()
    with (root/'tests.log').open('x') as log:
        class Result(unittest.TextTestResult):
            outcomes=[]
            def startTest(self,t): self.start=time.monotonic(); self.state='PASS'; super().startTest(t)
            def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
            def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
            def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
            def addExpectedFailure(self,t,e): self.state='EXPECTED_FAILURE'; super().addExpectedFailure(t,e)
            def addUnexpectedSuccess(self,t): self.state='UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
            def stopTest(self,t):
                self.outcomes.append({'test':t.id(),'status':self.state,'seconds':time.monotonic()-self.start})
                super().stopTest(t)
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result).run(suite)
    save(root/'observations.json',HZEndpointTests.observations)
    summary={'status':'PASS' if result.wasSuccessful() and all(r['status']=='PASS' for r in result.outcomes) else 'FAIL',
             'tests':result.testsRun,'outcomes':result.outcomes,'protocol_sha256':PROTOCOL_SHA,
             'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'implementation_sha256':sha(root/'implementation.json'),'observations_sha256':sha(root/'observations.json'),
             'tests_log_sha256':sha(root/'tests.log'),'seconds':time.monotonic()-begin,
             'native_solves':0,'gpu_calls':0,'real_requests':0,'source_complete_proofs':0,'hard_budget_supervision':False}
    save(root/'summary.json',summary); return summary


def verify_mc(item,deadline):
    """Check the MC rows against the same original HZ and evaluate its point exactly."""
    from act.back_end.moe.check_hz_endpoints import validate_request, _source
    from scoped_source.rowwise_bound import identity, clock, rows, rational
    from scoped_source.endpoint_controls import exact_point
    request=item['request']; validate_request(request,expected_request_sha256=identity(request),deadline=deadline)
    p=request['pairs'][0]; batch=p['batch']; source,nc,nb=_source(batch['source'],clock(deadline)); n=nc+nb
    duty=item['duty']; lp=item['lp']; tick=clock(deadline)
    if p['pair']!=[0,1] or duty['gate']!=p['gate']['bounds']: raise ValueError('MC pair/gate binding')
    for side,row in [('a',0),('b',request['classes'])]:
        coefficients=[F(0)]*n
        for key,shift in [('Gc',0),('Gb',nc)]:
            for j,v in source[key][row].items(): coefficients[j+shift]=v
        if duty[side]!={'c':list(map(str,coefficients)),'offset':str(source['c'][row])}: raise ValueError('MC form changed')
    if (request['properties']!=[{'id':'margin','q':['1','-1'],'offset':'0'}]
            or any(source[k][i] for k in ('Gc','Gb') for i in (1,3))
            or source['c'][1]!=0 or source['c'][3]!=0): raise ValueError('MC property control identity')
    base=dict(batch['base'],c=['0']*n,offset='0')
    if duty['base']!=base: raise ValueError('MC base differs from endpoint')
    d=[F(a)-F(b) for a,b in zip(duty['a']['c'],duty['b']['c'])]; d0=F(duty['a']['offset'])-F(duty['b']['offset'])
    dl,du=F(-3,2),F(3,2); lo,hi=map(F,duty['gate'])
    if d0-sum(map(abs,d))<dl or d0+sum(map(abs,d))>du: raise ValueError('difference range not justified by box')
    a=[dict(r) for r in rows(base['A'],(len(base['b']),n),tick)]
    eq=[dict(r) for r in rows(base['E'],(len(base['h']),n),tick)]
    rhs=list(map(rational,base['b']))
    for cd,ct,cw,bound in [(lo,dl,-1,lo*dl-lo*d0),(hi,du,-1,hi*du-hi*d0),
                          (-hi,-dl,1,-hi*dl+hi*d0),(-lo,-du,1,-lo*du+lo*d0)]:
        a.append({j:v for j,v in enumerate([cd*v for v in d]+[ct,F(cw)]) if v}); rhs.append(bound)
    clean=lambda r:{j:v for j,v in r.items() if v}
    if ([clean(dict(r)) for r in rows(lp['A'],(len(rhs),n+2),tick)]!=[clean(r) for r in a]
            or [clean(dict(r)) for r in rows(lp['E'],(len(eq),n+2),tick)]!=[clean(r) for r in eq]
            or list(map(rational,lp['b']))!=rhs or list(map(rational,lp['h']))!=list(map(rational,base['h']))):
        raise ValueError('MC planes/base rows changed')
    corners=[g*v for g in (lo,hi) for v in (dl,du)]
    if (list(map(rational,lp['lower']))!=list(map(rational,base['lower']))+[lo,min(corners)]
            or list(map(rational,lp['upper']))!=list(map(rational,base['upper']))+[hi,max(corners)]
            or list(map(rational,lp['c']))!=list(map(F,duty['b']['c']))+[F(0),F(1)]
            or rational(lp['offset'])!=F(duty['b']['offset'])): raise ValueError('MC box/objective changed')
    value=exact_point(lp,item['point'])
    if value!=F(item['value']) or value!=F(-1,32): raise ValueError('MC point/control value')
    return {'same_base_gate_property':True,'exact_feasible_value':str(value),'original_network_counterexample':False}


def check_control_links(obs):
    """Separate individually valid proofs are not automatically a paired comparison."""
    from scoped_source.rowwise_bound import identity
    reference=identity(obs['separation_analytic']['request'])
    if (identity(obs['separation_projected']['request'])!=reference
            or identity(obs['mc_negative_point']['request'])!=reference):
        raise ValueError('endpoint/MC comparison request differs')
    a,b=(obs[k]['request'] for k in ('shared_relation','independent_relation'))
    for key in ('experts','classes','context','properties'):
        if a[key]!=b[key]: raise ValueError('relation comparison task differs')
    if len(a['pairs'])!=len(b['pairs']): raise ValueError('relation pair coverage differs')
    for left,right in zip(a['pairs'],b['pairs']):
        if (any(left[k]!=right[k] for k in ('pair','sources','gate'))
                or left['relation_mode']!='shared_input' or right['relation_mode']!='independent_inputs'):
            raise ValueError('relation comparison original sources/gate differ')


def check_frozen_fixtures(obs):
    """Named mechanism controls must contain the registered coefficients."""
    from act.back_end.moe.check_hz_endpoints import _source
    from scoped_source.rowwise_bound import clock
    from itertools import combinations
    tick=clock(time.monotonic()+300)
    cfg=load(ROOT/PROTOCOL)['separation_fixture']
    def matrix(dense): return [{j:F(v) for j,v in enumerate(row) if F(v)} for row in dense]
    def expected(c,gc,auc,ub):
        return {'c':list(map(F,c)),'b':[],'ub':list(map(F,ub)),
                'Gc':matrix(gc),'Gb':[{} for _ in c],'Ac':[],'Ab':[],
                'Auc':matrix(auc),'Aub':[{} for _ in ub]}
    request=obs['separation_analytic']['request']
    if (request['experts']!=3 or request['classes']!=2
            or request['properties']!=[{'id':'margin','q':['1','-1'],'offset':'0'}]):
        raise ValueError('frozen separation dimensions/property')
    scores=[[F(v) for v in row] for row in cfg['scores']]
    for record,pair,gate in zip(request['pairs'],combinations(range(3),2),cfg['gate_intervals_by_pair']):
        if record['pair']!=list(pair) or record['gate']['bounds']!=gate or record['relation_mode']!='shared_input':
            raise ValueError('frozen separation pair/gate')
        outsider=next(i for i in range(3) if i not in pair)
        guards=[[scores[outsider][0]-scores[i][0]] for i in pair]
        rhs=[scores[i][1]-scores[outsider][1] for i in pair]
        parsed,nc,nb=_source(record['sources']['entry'],tick)
        if (nc,nb)!=(1,0) or parsed!=expected([0],[[1]],guards,rhs): raise ValueError('frozen entry coefficients')
        for side,expert in zip(('a','b'),pair):
            parsed,nc,nb=_source(record['sources'][side],tick)
            relu=F(cfg['relu_coefficient']); constant=F(cfg['margin_constant'])+relu
            coeff=F(cfg['linear_coefficients'][expert])
            rows=[r+[0,0] for r in guards]+[[1,F(-1,2),0],[-1,0,F(-1,2)],[-1,1,0],[1,0,1]]
            target=expected([constant,0],[[coeff,relu/2,relu/2],[0,0,0]],rows,rhs+[F(1,2),F(1,2),0,0])
            if (nc,nb)!=(3,0) or parsed!=target: raise ValueError('frozen private expert coefficients')
    request=obs['shared_relation']['request']; record=request['pairs'][0]
    if (request['experts']!=2 or request['classes']!=2 or len(request['pairs'])!=1
            or request['properties']!=[{'id':'margin','q':['1','-1'],'offset':'0'}]
            or record['pair']!=[0,1] or record['gate']['bounds']!=['1/2','1/2']):
        raise ValueError('frozen relation task')
    for side,c,g in [('entry',[0],[[1]]),('a',[F(1,4),0],[[1],[0]]),('b',[F(1,4),0],[[-1],[0]])]:
        parsed,nc,nb=_source(record['sources'][side],tick)
        if (nc,nb)!=(1,0) or parsed!=expected(c,g,[],[]): raise ValueError('frozen relation source')


def check_cost(name,package):
    if name in ('separation_analytic','partial'):
        if 'cost_seconds' in package: raise ValueError('analytic/partial package has no timed execution claim')
        return
    cost=package.get('cost_seconds')
    if (type(cost) is not dict or set(cost)!={'prepare','propose','check','total'}
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<300 for v in cost.values())
            or cost['total']<=0 or abs(cost['total']-sum(cost[k] for k in ('prepare','propose','check')))>1e-8):
        raise ValueError('mandatory cost partition')


def check_reference_results(checked):
    for name,result in checked.items():
        if name!='partial' and result['missing_endpoints']!=0:
            raise ValueError('complete control unexpectedly lost evidence')
    expected={
        'separation_analytic':('CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE',3,3,6,'3/32'),
        'shared_relation':('CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE',1,1,1,'1/4'),
        'independent_relation':('UNKNOWN_NONPOSITIVE',1,0,1,'-3/4')}
    for name,signature in expected.items():
        r=checked[name]
        if tuple(r[k] for k in ('status','required','positive','checked_endpoints','minimum_bound'))!=signature:
            raise ValueError('complete registered reference changed')
    if (checked['partial']['status']!='UNKNOWN_MISSING_EVIDENCE'
            or checked['partial']['missing_endpoints']!=2 or checked['partial']['checked_endpoints']!=4):
        raise ValueError('registered partial control changed')


def audit(root):
    summary=load(root/'summary.json'); bindings=load(root/'implementation.json'); obs=load(root/'observations.json')
    if (set(bindings)!=set(FILES) or summary['implementation_sha256']!=sha(root/'implementation.json')
            or summary['observations_sha256']!=sha(root/'observations.json')
            or summary['tests_log_sha256']!=sha(root/'tests.log') or summary['protocol_sha256']!=PROTOCOL_SHA
            or summary['tests']!=18 or [r['test'] for r in summary['outcomes']]!=names()
            or summary['status']!='PASS' or any(r['status']!='PASS' for r in summary['outcomes'])
            or set(obs)!=OBS): raise ValueError('complete frozen control inventory')
    for name,value in bindings.items():
        if sha(ROOT/name)!=value or sha(root/'implementation'/name)!=value: raise ValueError('source identity: '+name)
    if any(summary[k]!=0 for k in ('native_solves','gpu_calls','real_requests','source_complete_proofs')) or summary['hard_budget_supervision']:
        raise ValueError('scope changed')
    check_control_links(obs)
    check_frozen_fixtures(obs)
    from act.back_end.moe.check_hz_endpoints import check_request
    from scoped_source.rowwise_bound import identity
    checked={}; end=time.monotonic()+300
    for name,p in obs.items():
        if name in ('route_witnesses','mc_negative_point'): continue
        accepted=check_request(p['request'],p['proof'],expected_request_sha256=identity(p['request']),deadline=end)
        if accepted!=p['checked']: raise ValueError('saved bound mismatch: '+name)
        check_cost(name,p)
        checked[name]={'status':accepted['status'],'required':accepted['required'],'positive':accepted['positive'],
                       'checked_endpoints':accepted['checked_endpoints'],'missing_endpoints':accepted['missing_endpoints'],
                       'minimum_bound':min((F(r['lower_bound']) for r in accepted['results'] if r['lower_bound'] is not None),default=None)}
        if checked[name]['minimum_bound'] is not None: checked[name]['minimum_bound']=str(checked[name]['minimum_bound'])
    expected=[{'x':'-1','legal_pairs':[[1,2]]},{'x':'-1/2','legal_pairs':[[0,1],[1,2]]},{'x':'1','legal_pairs':[[0,1]]}]
    if obs['route_witnesses']!=expected: raise ValueError('control route/tie witnesses')
    check_reference_results(checked)
    return {'status':'PASS','root':str(root),'summary_sha256':sha(root/'summary.json'),'tests':18,
            'packages_rechecked':len(checked),'checked':checked,'mc':verify_mc(obs['mc_negative_point'],end),
            'source_complete':False,'gpu_calls':0,'native_solves':0,'real_requests':0,'hard_budget_supervision':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['run','audit']); p.add_argument('root',type=Path); p.add_argument('--report',type=Path)
    args=p.parse_args(); root=args.root.resolve()
    if not root.is_relative_to(ROOT.parent/'baseline_runs'): raise ValueError('project archive only')
    result=run(root) if args.action=='run' else audit(root)
    if args.report: save(args.report,result)
    print(result)
