"""Compact, reproducible saved-analysis accounting; not independent LP proof."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys


def summarize(path):
    d=json.loads(path.read_bytes())
    if d['schema']!='LAST_RELU_SAVED_LOCALIZATION_V1' or d['required_properties']!=9:
        raise ValueError('registered analysis')
    for key in ('new_solver_calls','network_forward_calls','new_source_propagations','new_lower_bounds_proved','new_complete_safe'):
        if d[key]!=0:raise ValueError('analysis claim scope')
    if d['exact_primal_feasibility_checked'] or d['production_verdict_changed']:raise ValueError('claim upgrade')
    if [r['competitor'] for r in d['properties']]!=list(range(1,10)):raise ValueError('complete property inventory')
    points=[]
    for r in d['properties']:
        row={'competitor':r['competitor'],'point_status':r['point_status'],'checked_lower_bound':r['checked_lower_bound']}
        if r['point_status']=='MISSING_UNVERIFIED_POINT':points.append(row);continue
        if r['point_status']!='UNVERIFIED_POINT_DIAGNOSTIC_ONLY':raise ValueError('unknown diagnostic point status')
        t={k:F(v) for k,v in r['exact_terms'].items()}
        only_relu=t['relaxed_objective']-t['final_affine_residual']-t['weighted_last_relu_signed_gap']
        if (r['decomposition_residual']!='0' or
                t['relaxed_objective']+t['product_gap']!=t['product_only_replaced'] or
                only_relu+t['product_gap']!=t['product_and_last_relu_replaced']):raise ValueError('point accounting')
        exact={**r['exact_terms'],'last_relu_only_replaced':str(only_relu)}
        row.update(exact_terms=exact,display_terms={k:float(F(v)) for k,v in exact.items()},candidate_sha256=r['candidate_sha256'])
        points.append(row)
    nodes=d['last_relu_rows'];unstable={(r['expert'],r['row']) for r in nodes if r['branch']=='unstable'}
    ranked=d['ranked_unstable_rows']
    if len(nodes)!=128 or len({(r['expert'],r['row']) for r in nodes})!=128 or len(unstable)!=d['last_relu_unstable']:
        raise ValueError('hidden row inventory')
    if len(ranked)!=len(unstable) or {(r['expert'],r['row']) for r in ranked}!=unstable:raise ValueError('ranking dropped rows')
    if sum(c['counts'].get('unstable',0) for c in d['relu_census'])!=d['total_relu_binaries']:
        raise ValueError('binary census')
    ordered=sorted(ranked,key=lambda r:(-F(r['max_observed_harm']),r['expert'],r['row']))
    if ordered!=ranked:raise ValueError('ranking order')
    rank_display=[{'expert':r['expert'],'row':r['row'],'worst_competitor':r['worst_competitor'],
        'harm_float':float(F(r['max_observed_harm']))} for r in ranked]
    return {'schema':'LAST_RELU_LOCALIZATION_SUMMARY_V1','raw_analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'raw_analysis_bytes':path.stat().st_size,'analysis_script_sha256':d['script_sha256'],
        'review_sha256':d['review_sha256'],'package_manifest_sha256':d['package_manifest_sha256'],'source_sha256':d['source_sha256'],
        'request_index':d['request']['dataset_index'],'pair':d['pair'],'relu_census':d['relu_census'],
        'required_properties':9,'points_present':sum('exact_terms' in r for r in points),
        'properties':points,'last_relu_unstable':len(unstable),'ranked_unstable_rows':rank_display,
        'new_solver_calls':0,'new_source_propagations':0,'new_complete_safe':0,
        'exact_primal_feasibility_checked':False,'production_verdict_changed':False,
        'analysis_seconds':d['seconds'],'scope':d['scope']}


if __name__=='__main__':print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
