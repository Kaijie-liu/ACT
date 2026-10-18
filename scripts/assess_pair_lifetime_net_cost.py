"""Static integer/source accounting only; no production imports or controls."""
import ast
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
DOC='docs/hz_pair_lifetime_net_cost_20261018.md'
REPORT='docs/hz_pair_lifetime_net_cost_20261018.json'
OLD='docs/hz_pair_scope_lifetime_20261018.json'
COUNTS='docs/hz_support_amortization_20261018.json'
FAILED='docs/hz_pair_scope_lifetime_results_20261018.json'
FREEZE='docs/hz_pair_scope_lifetime_freeze_20261018.json'
SHAPE='docs/hz_native_full_shape_decision_20261018.json'


def require(ok,why):
    if not ok: raise ValueError(why)


def read(path): return json.loads((ROOT/path).read_bytes())
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def analyze():
    # Bind the implemented path, without pretending the failed controls passed.
    freeze=read(FREEZE)
    for path,h in freeze['files'].items(): require(sha(path)==h,'frozen implementation drift '+path)
    old=read(OLD); counts=read(COUNTS); shape=read(SHAPE)['shape']; failed=read(FAILED)
    require(sha(COUNTS)==old['anchors'][COUNTS] and sha(SHAPE)==counts['anchors'][SHAPE], 'prior integer premises')
    require(failed['status']=='FAILED_BATCH_RECORDED' and failed['tests']['failed']==1, 'R1 remains failed')
    require(shape['experts']==8 and shape['classes']==10 and shape['pairs']==28 and
            shape['property_duties']==252 and shape['parameter_values']==6961368, 'registered topology')
    tree=ast.parse((ROOT/'scoped_source/pair_workspace.py').read_bytes())
    names={n.name for n in tree.body if isinstance(n,ast.ClassDef)}
    require({'PairLease','PairWorkspace','Scope'}<=names,'implemented lease classes')
    # Syntax presence is an identity anchor, not an automatic semantic proof.
    examples=[]
    for endpoints,tiles in ((9,2),(18,3)):
        objective=endpoints; dual=(endpoints+1)//2; pulls=objective+dual
        common=4+18*tiles+5*pulls
        examples.append(dict(endpoints=endpoints,tiles=tiles,objective_pulls=objective,
                             dual_pulls=dual,j_scoped=common,j_global=common+4))
    require([(x['j_scoped'],x['j_global']) for x in examples]==[(110,114),(193,197)], 'full plan charges')
    vertices=[]
    for c in counts['vertices']:
        ur,vr={'positive':(0,0),'negative':(0,128),'unstable':(128,0)}[c['router']]
        nu,nv=c['unstable_experts'],c['negative_experts']
        experts=[(384,0)]*nu+[(0,384)]*nv+[(0,0)]*(8-nu-nv)
        n=6696+4*ur+vr+sum(4*u+v for u,v in experts)
        f=3072+3*(ur+sum(u for u,v in experts))
        require((n,f)==(c['nodes'],c['mandatory_factor_lower']),'same count domain')
        present=missing=0
        for a,b in combinations(range(8),2):
            ua,va=experts[a];ub,vb=experts[b]
            s=4008+4*ur+vr+4*(ua+ub)+va+vb
            t=3072+3*(ur+ua+ub); h=2*(n+f)+s+t+53
            h1=max(0,2*n-5*s)-1; h2=max(0,3*n-6*s)-1
            present+=max(10*h1+4*h2-h+4,18*h1+9*h2-h+4)
            missing+=max(9*h1+4*max(0,h2)+max(0,h1)-h+4,18*h1+9*max(0,h2)-h+4)
        vertices.append(dict(router=c['router'],unstable_experts=nu,negative_experts=nv,
                             all_present_upper=present,completed_missing_upper=missing))
    require(len(vertices)==135,'count vertex catalog')
    maxima={key:max(vertices,key=lambda x:x[key]) for key in ('all_present_upper','completed_missing_upper')}
    require(all(x[key]==2839088 for key,x in maxima.items()), 'closed net upper')
    require(old['maxima']['all_present_component_upper']['all_present_component_upper']+4*28==2839088,
            'old component upper plus explicit lifetime difference')
    unstable=next(x for x in vertices if x['router']=='unstable' and x['unstable_experts']==8)
    positive=next(x for x in vertices if x['router']=='positive' and x['unstable_experts']==x['negative_experts']==0)
    require(unstable['all_present_upper']==1602608 and positive['all_present_upper']==-747012 and
            positive['completed_missing_upper']==-746872,'corner figures')
    observed=[]
    for r in failed['comparisons']:
        duties,endpoints,tiles=freeze['full_rosters'][r['source']]
        pairs=28 if r['source']=='eight_private_six_outputs' else 3
        pulls=endpoints+endpoints//2  # these retained fixtures have even tile sizes
        common=4*pairs+18*tiles+5*pulls
        g,s=r['lifetime_by_arm'][2:]
        total=lambda x:sum(x[k] for k in ('global_seal_work','lease_work','plan_work'))
        require(total(g)==common+4*pairs and total(s)==common,'retained observed J charges')
        observed.append(dict(source=r['source'],profile=r['profile'],j_scoped=total(s),j_global=total(g),
                             difference=total(g)-total(s)))
    paths=(DOC,OLD,COUNTS,FAILED,FREEZE,SHAPE,'scripts/assess_pair_lifetime_net_cost.py',
           'scoped_source/pair_workspace.py','scoped_source/scaled_endpoints.py',
           'scoped_source/bounded_runtime.py','scoped_source/hz_request_store.py')
    return dict(schema='PAIR_LIFETIME_IMPLEMENTED_NET_WORK_STATIC_V1',base_head='d181f6b37a0ff2ab3ead32d16de74df2460a9909',
        anchors={p:sha(p) for p in paths},formula=dict(common='4R+18M+5C',global_extra='4R',
            difference='J_global-J_scoped=4R',delta='sum_call_gamma-H+4 per pair',
            full_completion_only=True,source='same checked Shared/tuple forest and candidate/missing plan'),
        full_plan_examples=examples,observed_synthetic_record_checks=observed,vertices_checked=len(vertices),
        maxima=maxima,all_unstable=unstable,all_positive=positive,
        limits=dict(interpretation='optimistic net Work upper, NOT measured gain or absolute resource bound',
            gate_pulls='no workspace, unchanged and excluded from C',wire='different counting, excluded',
            elapsed_and_heap='not represented by Work difference'),
        decision=dict(r1='FAIL_UNCHANGED',runtime_edits=0,new_controls=0,new_sources=0,new_candidates=0,
            native_calls=0,models=0,gpu=0,quota_changes=False,acceptance_changes=False,
            source_floor=2*shape['parameter_values'],work_cap=8000000,full_shape_admitted=False,
            next='Only separately authorized factor-fault repair; no performance rerun or wrapper expansion',goals='G1-G6 OPEN'))


if __name__=='__main__':
    result=analyze()
    if sys.argv[1:]==['--check']:
        require(result==read(REPORT),'static report drift')
        print('Static source/record/integer accounting agrees. R1 remains FAIL; no verifier executed.')
    else:
        require(not sys.argv[1:],'use no arguments or --check')
        print(json.dumps(result,sort_keys=True,indent=2))
