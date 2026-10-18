"""Source-bound integer accounting only; no production imports or new runs.

The convex vertex reduction is explained in the companion document. This
script checks that arithmetic and frozen records agree; it is not a network
verifier, an execution admission, or a timing benchmark.
"""
import ast
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE='37d3251ff0836027315ed63b1fed683098cdb7f1'
DOC='docs/hz_support_amortization_20261018.md'
REPORT='docs/hz_support_amortization_20261018.json'
FREEZE='docs/hz_support_traversal_freeze_20261018_r2.json'
RESULT='docs/hz_support_traversal_controls_results_20261018_r2.json'
SHAPE='docs/hz_native_full_shape_decision_20261018.json'


def require(ok,why):
    if not ok: raise ValueError(why)


def raw(path): return (ROOT/path).read_bytes()
def sha(path): return hashlib.sha256(raw(path)).hexdigest()
def load(path): return json.loads(raw(path))


def gains(n,s):
    # max(0,G)-1 overestimates the actual R2 G>1 selector at G=1.
    return max(0,2*n-5*s)-1,max(0,3*n-6*s)-1


def pair_upper(n,f,rr,ur,left,right):
    ua,va=left; ub,vb=right
    s=4008+rr+4*(ua+ub)+va+vb
    sf=3072+3*(ur+ua+ub)  # mandatory factors; compensation only adds cost
    setup=2*(n+f)+s+sf+53
    h1,h2=gains(n,s)
    single=10*h1+4*h2-2*setup
    double=18*h1+9*h2-3*setup
    single_missing=9*h1+4*max(0,h2)+max(0,h1)-2*setup
    double_missing=18*h1+9*max(0,h2)-3*setup
    tile8=8*h1+4*h2-setup
    tile2=2*h1+h2-setup
    tile1=2*h1-setup
    ideal_gate=max(max(0,tile8)+max(0,tile1),2*max(0,tile8)+max(0,tile2))
    return dict(all_present=max(single,double),missing=max(single_missing,double_missing),
                ideal_free_tile_gate=ideal_gate)


def analyze():
    freeze=load(FREEZE)
    for path,value in freeze['files'].items():
        require(sha(path)==value,'R2 bound source drift '+path)
    shape=load(SHAPE)['shape']
    require(shape['expert_widths']==[3072,256,128,10] and
            shape['router_widths']==[3072,128,8] and shape['experts']==8 and
            shape['pairs']==28 and shape['property_duties']==252,'registered topology')
    require(shape['affine_outputs']+shape['inputs']+shape['guards']==6696,'base node count')
    require(3072+128+8+2*(256+128+10)+12==4008,'pair base node count')
    require(28+6+6+6+7==53 and shape['views']==6+8*6+28,'ancestry and setup constants')
    # Syntax inspection only, never importing these classes.
    tree=ast.parse(raw('scoped_source/pair_workspace.py'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PairWorkspace')
    select=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='select')
    require(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                n.func.attr=='prepare_traversal' for n in ast.walk(select)), 'R2 eager setup identity')
    r=load(RESULT); paired=r['paired']; observed=[]
    require(len(paired)==16 and r['tests_completed']==10,'retained R2 denominator')
    for a,b in zip(paired[::2],paired[1::2],strict=True):
        require(not a['enabled'] and b['enabled'] and
                (a['source'],a['profile'])==(b['source'],b['profile']),'R2 arms')
        require(a['receipts_sha256']==b['receipts_sha256'] and
                a['full_result_sha256']==b['full_result_sha256'],'R2 exact result identity')
        s=b['traversal']; delta=a['total_work']-b['total_work']
        require(delta==s['planned_iteration_saving']-s['setup_work']-s['selector_work'],'observed complete costs')
        observed.append(dict(source=a['source'],profile=a['profile'],net_work_saving=delta))

    corners=[]
    states=(('positive',0,0),('negative',0,128),('unstable',128,0))
    for router,ur,vr in states:
        rr=4*ur+vr
        for nu in range(9):
            for nv in range(9-nu):
                experts=[(384,0)]*nu+[(0,384)]*nv+[(0,0)]*(8-nu-nv)
                re=sum(4*u+v for u,v in experts); ue=sum(u for u,v in experts)
                n=6696+rr+re; f=3072+3*(ur+ue)
                pairs=[pair_upper(n,f,rr,ur,experts[a],experts[b]) for a,b in combinations(range(8),2)]
                corners.append(dict(router=router,unstable_experts=nu,negative_experts=nv,
                    nodes=n,mandatory_factor_lower=f,
                    all_present_upper=sum(p['all_present'] for p in pairs),
                    completed_missing_upper=sum(p['missing'] for p in pairs),
                    ideal_free_tile_gate_upper=sum(p['ideal_free_tile_gate'] for p in pairs)))
    require(len(corners)==135,'complete exchange-symmetric vertex catalog')
    best={key:max(corners,key=lambda p:p[key]) for key in
          ('all_present_upper','completed_missing_upper','ideal_free_tile_gate_upper')}
    require(best['all_present_upper']['all_present_upper']==-1134330,'full R2 upper')
    require(best['completed_missing_upper']['completed_missing_upper']==-1134315,'missing R2 upper')
    require(best['ideal_free_tile_gate_upper']['ideal_free_tile_gate_upper']==958454,'hypothetical free gate upper')
    unstable=next(c for c in corners if c['router']=='unstable' and c['unstable_experts']==8)
    require(unstable['all_present_upper']==-2616096,'all unstable upper')

    paths=(DOC,FREEZE,RESULT,SHAPE,'scripts/assess_support_amortization.py',
           'scoped_source/pair_workspace.py','scoped_source/scaled_endpoints.py',
           'scoped_source/forest_endpoints.py','scoped_source/bounded_runtime.py',
           'scoped_source/check_source_dag.py','scoped_source/check_shared_source.py',
           'scoped_source/check_affine_references.py','scoped_source/check_scaled_references.py',
           'scoped_source/wire_forest.py')
    return dict(schema='SUPPORT_AMORTIZATION_STATIC_DECISION_V1',base_head=BASE,
        anchors={p:sha(p) for p in paths},observed_r2=observed,vertices=corners,best_upper=best,
        all_unstable=unstable,
        conditions=dict(representation='registered Shared / tuple-node cost model, not Wire',
            count_domain='u>=0,v>=0,u+v<=units, rho=4u+v; continuous triangle outer domain',
            factors='mandatory 3u plus input; extra compensation ignored only to upper-bound savings',
            cost='coupled same-source Work delta; not elapsed time or quota',
            completion='each compared consumer completes; no guarantee for interrupted prefixes',
            endpoint_choice='each pair may independently take its more favorable 9 or 18 endpoint count',
            reduction='convex upper on product triangles; equal-size experts permit exchange symmetry'),
        representation_correction=dict(wire_iteration_saving='(2*b+1)*(N-S)-4*S',
            wire_setup='ordinary setup + N',wire_executed=False),
        decision=dict(current_r2='STOP_FULL_REGISTERED_SHAPE_PROMOTION',
            hypothetical_gate='UNIMPLEMENTED_REQUIRES_EXPLICIT_BRANCH_CONTRACT',
            all_checks_preserved='no runtime edits; cannot skip R2 canonical checks under an unchanged fault contract',
            source_floor=2*shape['parameter_values'],full_shape_admitted=False,
            new_controls=0,new_sources=0,new_candidates=0,native_calls=0,models=0,gpu=0,
            quota_changes=False,acceptance_changes=False,goals='G1-G6 OPEN'))


if __name__=='__main__':
    result=analyze()
    if sys.argv[1:]==['--check']:
        require(result==load(REPORT),'static report drift')
        print('PASS: source binding, retained-record arithmetic and 135 static count vertices; no verifier execution')
    else:
        require(not sys.argv[1:],'use no arguments or --check')
        print(json.dumps(result,sort_keys=True,indent=2))
