"""Independent exact accounting/roster check, NOT an LP feasibility checker."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

COMMON=[1,2,3,4,5,6,8,9]
GROUPS=('prefix_selected','property_selected','other')
KEYS=('checked_lower_bound','relaxed_objective','product_gap','final_affine_residual',
      'weighted_last_relu_signed_gap','product_and_last_relu_replaced','point_minus_bound',
      'dual_constant','residual_box_correction')


def scope(d):
    for k in ('new_solver_calls','network_forward_calls','new_source_propagations','new_lower_bounds_proved','new_complete_safe'):
        if d[k]!=0:raise ValueError('analysis scope')
    if d['exact_primal_feasibility_checked'] or d['production_verdict_changed']:
        raise ValueError('proof upgrade')


def check_point(r):
    if r['point_status']=='MISSING_UNVERIFIED_POINT':return None
    if r['point_status']!='UNVERIFIED_POINT_DIAGNOSTIC_ONLY':raise ValueError('point status')
    t={k:F(v) for k,v in r['exact_terms'].items()}
    J,P,A,R,T=[t[k] for k in ('relaxed_objective','product_gap','final_affine_residual',
                              'weighted_last_relu_signed_gap','product_and_last_relu_replaced')]
    if (r['decomposition_residual']!='0' or J+P-A-R!=T or
            J+P!=t['product_only_replaced'] or J-A-R!=t['last_relu_only_replaced']):
        raise ValueError('point accounting')
    if (F(r['checked_lower_bound'])!=t['checked_lower_bound'] or
            t['checked_lower_bound']!=t['dual_constant']+t['residual_box_correction'] or
            J-t['checked_lower_bound']!=t['point_minus_bound']):raise ValueError('dual accounting')
    if set(r['exact_relu_groups'])!=set(GROUPS) or sum(map(F,r['exact_relu_groups'].values()))!=R:
        raise ValueError('group accounting')
    if t['gate_value']*t['difference_value']-t['relaxed_product']!=P:
        raise ValueError('product accounting')
    if r['gate_range']!=['0','1'] or F(r['difference_range'][0])>F(r['difference_range'][1]):
        raise ValueError('range identity')
    # Values at a possibly infeasible point remain diagnostics even if tiny.
    return t


def summarize(path):
    d=json.loads(path.read_bytes());scope(d)
    if (d['schema']!='PAIRED_PROPERTY_SAVED_DIAGNOSIS_V1' or d['diagnostic_properties']!=COMMON or
            d['required_output_properties']!=9 or d['excluded_from_point_diagnosis']!=[7] or
            d['exact_LP_optimality_checked']):raise ValueError('fixed diagnosis')
    groups=d['row_groups']
    if groups!={'prefix_selected':[[1,0],[1,1],[2,0],[2,1]],
                'property_selected':[[1,10],[1,30],[2,17],[2,31]]}:
        raise ValueError('frozen row groups')
    if [a['arm'] for a in d['arms']]!=['prefix','property']:raise ValueError('arm roster')
    arms=[];tables=[]
    for arm in d['arms']:
        scope(arm)
        if (arm['diagnostic_properties']!=COMMON or arm['required_output_properties']!=9 or
                [r['competitor'] for r in arm['properties']]!=COMMON or
                [r['competitor'] for r in arm['endpoint_ledger']]!=list(range(1,10)) or
                any(r['paired_diagnosis']!=(r['competitor'] in COMMON) for r in arm['endpoint_ledger'])):
            raise ValueError('property roster')
        nodes=arm['last_relu_rows'];expected={(e,j) for e in (1,2) for j in range(64)}
        if len(nodes)!=128 or {(r['expert'],r['row']) for r in nodes}!=expected:raise ValueError('hidden roster')
        for r in nodes:
            lo,hi=map(F,r['range']);branch='active' if lo>=0 else 'inactive' if hi<=0 else 'unstable'
            group=next((n for n,v in groups.items() if [r['expert'],r['row']] in v),'other')
            if lo>hi or r['branch']!=branch or r['group']!=group:raise ValueError('hidden range/group')
        unstable={(r['expert'],r['row']) for r in nodes if r['branch']=='unstable'}
        if (len(unstable)!=arm['last_relu_unstable'] or
                sum(c['counts'].get('unstable',0) for c in arm['relu_census'])!=arm['total_relu_binaries']):
            raise ValueError('ReLU census')
        points=[];table={}
        for r in arm['properties']:
            t=check_point(r);table[r['competitor']]=t
            if t is not None:
                if (len(r['observations'])!=len(unstable) or
                        {(x['expert'],x['row']) for x in r['observations']}!=unstable):raise ValueError('observed row roster')
                points.append({k:v for k,v in r.items() if k not in ('observations','display_terms')})
                points[-1]['display_terms']={k:float(v) for k,v in t.items()}
            else:points.append(r)
        arms.append({k:arm[k] for k in ('arm','package_manifest_sha256','source_sha256','request','pair',
                                      'relu_census','total_relu_binaries','last_relu_unstable','endpoint_ledger','seconds')})
        arms[-1].update(properties=points,group_counts=dict(Counter(r['group'] for r in nodes)),
                        selected_rows=[r for r in nodes if r['group']!='other'],
                        points_present=sum(v is not None for v in table.values()))
        tables.append(table)
    if arms[0]['request']!=arms[1]['request'] or [r['competitor'] for r in d['paired']]!=COMMON:
        raise ValueError('paired identity')
    pairs=[]
    for r in d['paired']:
        k=r['competitor'];a,b=[t[k] for t in tables]
        if a is None or b is None:
            if r.get('status')!='MISSING_SAVED_POINT_NO_PAIRED_ACCOUNTING':raise ValueError('missing point upgraded')
            pairs.append(r);continue
        delta={v:b[v]-a[v] for v in KEYS}
        if ({v:F(x) for v,x in r['exact_delta'].items()}!=delta or r['decomposition_residual']!='0' or
                delta['checked_lower_bound']!=delta['product_and_last_relu_replaced']-delta['product_gap']+
                delta['final_affine_residual']+delta['weighted_last_relu_signed_gap']-delta['point_minus_bound']):
            raise ValueError('paired accounting')
        rows=[next(x for x in arm['properties'] if x['competitor']==k) for arm in d['arms']]
        gd={g:F(rows[1]['exact_relu_groups'][g])-F(rows[0]['exact_relu_groups'][g]) for g in GROUPS}
        if {g:F(x) for g,x in r['exact_relu_group_delta'].items()}!=gd:raise ValueError('paired group')
        pairs.append({**r,'display_delta':{v:float(x) for v,x in delta.items()},
                      'display_relu_group_delta':{v:float(x) for v,x in gd.items()}})
    return {'schema':'PAIRED_PROPERTY_DIAGNOSIS_SUMMARY_V1','status':'PASS_ACCOUNTING_ONLY','issues':[],
        'raw_analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_analysis_bytes':path.stat().st_size,
        'analysis_script_sha256':d['script_sha256'],'summary_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'review_sha256':d['review_sha256'],'diagnostic_properties':COMMON,'excluded_from_point_diagnosis':[7],
        'required_output_properties':9,'arms':arms,'paired':pairs,
        'new_solver_calls':0,'new_source_propagations':0,'new_lower_bounds_proved':0,'new_complete_safe':0,
        'exact_primal_feasibility_checked':False,'exact_LP_optimality_checked':False,
        'production_verdict_changed':False,'analysis_seconds':d['seconds'],
        'scope':'Independent arithmetic/identity checking of saved diagnosis, NOT primal, optimum, causal or safety proof.'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('analysis',type=Path)
    p.add_argument('--output',type=Path);a=p.parse_args();s=summarize(a.analysis)
    if a.output:
        with a.output.open('x') as stream:json.dump(s,stream,indent=2);stream.write('\n')
        print(json.dumps({'status':s['status'],'pairs':len(s['paired']),'output':str(a.output)}))
    else:print(json.dumps(s,indent=2))
