"""Derived archive from reviewed records only; no new model or solver work."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys


def activation(interval):
    lo,hi=map(F,interval)
    if lo>hi:raise ValueError('reversed range')
    return 'inactive' if hi<=0 else 'active' if lo>=0 else 'unstable'


def analyze(path):
    path=path/'review.json' if path.is_dir() else path
    reviewed=json.loads(path.read_bytes())
    if reviewed['status']!='PASS' or reviewed['issues']:raise ValueError('review not accepted')
    if [r['arm'] for r in reviewed['arms']]!=['prefix','property']:raise ValueError('two-arm denominator/order')
    result={'schema':'PROPERTY_RANGE_SAVED_ANALYSIS_V1','review_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'execution_head':reviewed['execution_head'],'prequery_comparison':reviewed['prequery_comparison'],
        'arms':[],'new_solver_calls':0,'source_propagations':0,'network_forward_calls':0,
        'exact_LP_optimality_or_primal_feasibility_checked':False,'production_verdict_changed':False,
        'scope':'Single observed conv98, fixed two-arm order; no population, speedup, LP impossibility or model-unsafe claim.'}
    maps=[]
    for arm in reviewed['arms']:
        out={'arm':arm['arm'],'status':arm.get('outcome',arm['status']),
             'seconds':arm.get('total_seconds'),'complete_mathematical_check':arm.get('complete',False),
             'required_obligations':9,'terminal_sha256':arm.get('terminal_sha256')}
        maps.append({})
        if not out['complete_mathematical_check']:
            out.update(positive=None,nonpositive=None,missing=None,unchecked_obligations=9,
                       reason='No complete source/obligation mathematical check; missing is not zero-cost success.')
            result['arms'].append(out);continue
        r,g=arm['result'],arm['generation']
        if r['required_obligations']!=9 or [x['competitor'] for x in r['rows']]!=list(range(1,10)):
            raise ValueError('all nine obligations required')
        values={x['competitor']:F(x['checked_lower_bound']) for x in r['rows'] if 'checked_lower_bound' in x}
        if (sum(v>0 for v in values.values())!=r['positive_bounds'] or
            sum(v<=0 for v in values.values())!=r['nonpositive_bounds'] or 9-len(values)!=r['missing_bounds']):
            raise ValueError('positive/nonpositive/missing accounting')
        if r['complete_declared_real_output_proof']!=(r['positive_bounds']==9):raise ValueError('request aggregation')
        selections=[];expected=[];rosters={}
        for i,snap in sorted(arm['selection_snapshots'].items(),key=lambda x:int(x[0])):
            record=snap['record'];rosters[int(i)]=record
            expected.extend((int(i),j) for j in record['selected'])
            selected=[record['rows'][j] for j in record['selected']]
            selections.append({'expert':int(i),'source_sha256':record['source_sha256'],
                'classifier_sha256':record['classifier_sha256'],'selected':record['selected'],
                'rows':[{**x,'score_float':float(F(x['score']))} for x in selected]})
        if [(x['expert'],x['row']) for x in g['range_rows']]!=expected:raise ValueError('selected range roster')
        ranges=[]
        for row in g['range_rows']:
            lo,hi=map(F,row['generator_box']);new=row['checked_range'];record=rosters[row['expert']]
            if row['generator_box']!=record['rows'][row['row']]['range']:raise ValueError('pre-query range mismatch')
            item={'expert':row['expert'],'row':row['row'],'source_sha256':row['source_sha256'],
                  'fact_sha256':row['fact_sha256'],'outcome':row['outcome'],'calls':row['calls'],
                  'generator_box':row['generator_box'],'generator_box_float':[float(lo),float(hi)],
                  'checked_range':new,'checked_range_float':None,'strictly_tighter':False,
                  'before':activation(row['generator_box']),'after':None,'width_ratio':None}
            if row['fact_sha256'] is not None:
                a,b=map(F,new)
                if not lo<=a<=b<=hi:raise ValueError('range widened')
                item.update(checked_range_float=[float(a),float(b)],strictly_tighter=a>lo or b<hi,
                            after=activation(new),width_ratio=float((b-a)/(hi-lo)) if hi!=lo else None)
            elif new is not None:raise ValueError('unproved range without a fact')
            ranges.append(item)
        out.update(positive=r['positive_bounds'],nonpositive=r['nonpositive_bounds'],missing=r['missing_bounds'],
            unchecked_obligations=0,selection=selections,range_rows=ranges,
            range_calls=arm['range_calls_entered'],output_calls=arm['output_calls'],accepted_ranges=arm['accepted_range_facts'],
            phase_seconds={s['phase']:s['seconds'] for s in arm['stages']},
            selection_seconds=arm['selection_seconds'],range_subcosts={k:sum(x.get(k,0.) for x in g['range_rows']) for k in
                ('assembly_seconds','conversion_seconds','native_seconds','check_seconds','seconds')},
            prefix_seconds=g['prefix_generation_seconds'],
            layer_propagation_seconds=sum(x['propagation_seconds'] for x in g['steps']),
            layer_delta_serialization_seconds=sum(x['delta_serialization_seconds'] for x in g['steps']),
            joint_and_output_seconds=g['joint_and_output_seconds'],joint=r['source_checks']['joint'],
            bundle_bytes=arm['sealed']['bundle_bytes'],manifest_sha256=arm['sealed']['manifest_sha256'],
            independent_review_seconds=arm['review_seconds'],
            bound_rows=[{'competitor':x['competitor'],'status':x['status'],
                'checked_lower_bound':x.get('checked_lower_bound'),
                'checked_lower_bound_float':float(values[x['competitor']]) if x['competitor'] in values else None}
                for x in r['rows']])
        if out['accepted_ranges']!=sum(x['fact_sha256'] is not None for x in ranges):raise ValueError('fact count')
        if sum(x['calls'] for x in ranges)!=out['range_calls'] or out['range_calls']>8 or out['output_calls']>9:
            raise ValueError('call count')
        maps[-1]=values;result['arms'].append(out)
    old,new=maps
    result['common_bound_changes']=[{'competitor':k,'property_minus_prefix':str(new[k]-old[k]),
        'property_minus_prefix_float':float(new[k]-old[k])} for k in sorted(old.keys()&new.keys())]
    result['newly_available_competitors']=sorted(new.keys()-old.keys())
    result['lost_available_competitors']=sorted(old.keys()-new.keys())
    a,b=result['arms']
    result['descriptive_seconds_difference']=b['seconds']-a['seconds'] if a['seconds'] is not None and b['seconds'] is not None else None
    result['complete_positive_requests']=sum(a.get('positive')==9 for a in result['arms'])
    return result


if __name__=='__main__':print(json.dumps(analyze(Path(sys.argv[1]).resolve()),indent=2))
