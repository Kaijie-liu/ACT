"""Saved-evidence analysis only: no model, matrix construction or solver query."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys


def analyze(root):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    path=root if root.is_file() else root/'review.json'
    review=json.loads(path.read_bytes())
    if review['status']!='PASS' or review['issues']:raise ValueError('review not accepted')
    if [a['arm'] for a in review['arms']]!=['range_off','range_on']:raise ValueError('arm coverage')
    result={'schema':'RANGED_SOURCE_SAVED_ANALYSIS_V1','review_sha256':sha(path),
        'execution_head':review['execution_head'],'arms':[],'new_solver_calls':0,
        'exact_primal_or_LP_optimality_checked':False,'production_verdict_changed':False,
        'scope':'One fixed stored-source input, fixed order, no speedup/population or intrinsic-gap claim.'}
    bound_maps=[]
    for a in review['arms']:
        if not a['complete']:raise ValueError('cannot summarize absent complete mathematical check')
        r,g=a['result'],a['generation']
        if r['required_obligations']!=9 or [x['competitor'] for x in r['rows']]!=list(range(1,10)):
            raise ValueError('nine output obligations')
        bounds={x['competitor']:F(x['checked_lower_bound']) for x in r['rows'] if 'checked_lower_bound' in x}
        bound_maps.append(bounds)
        if (sum(v>0 for v in bounds.values())!=r['positive_bounds'] or
                sum(v<=0 for v in bounds.values())!=r['nonpositive_bounds'] or
                9-len(bounds)!=r['missing_bounds']):raise ValueError('bound count')
        rowcosts={k:sum(x.get(k,0.) for x in g['range_rows']) for k in
            ('assembly_seconds','conversion_seconds','native_seconds','check_seconds','seconds')}
        ranges=[]
        for x in g['range_rows']:
            if x['fact_sha256'] is None:continue
            old=list(map(F,x['generator_box']));new=list(map(F,x['checked_range']))
            if not old[0]<=new[0]<=new[1]<=old[1]:raise ValueError('range widened')
            ranges.append({'expert':x['expert'],'row':x['row'],'source_sha256':x['source_sha256'],
                'fact_sha256':x['fact_sha256'],'generator_box':x['generator_box'],'checked_range':x['checked_range'],
                'generator_box_float':list(map(float,old)),'checked_range_float':list(map(float,new)),
                'width_ratio':float((new[1]-new[0])/(old[1]-old[0])) if old[1]!=old[0] else None,
                'previously_unstable':old[0]<0<old[1],'now_inactive':new[1]<=0})
        result['arms'].append({'arm':a['arm'],'status':a['outcome'],'seconds':a['total_seconds'],
            'phase_seconds':{s['phase']:s['seconds'] for s in a['stages']},
            'positive':r['positive_bounds'],'nonpositive':r['nonpositive_bounds'],'missing':r['missing_bounds'],
            'joint':r['source_checks']['joint'],'range_calls':a['range_calls_entered'],
            'accepted_ranges':a['accepted_range_facts'],'range_rows':ranges,'range_subcosts':rowcosts,
            'prefix_seconds':g['prefix_generation_seconds'],
            'layer_propagation_seconds':sum(x['propagation_seconds'] for x in g['steps']),
            'layer_delta_serialization_seconds':sum(x['delta_serialization_seconds'] for x in g['steps']),
            'joint_and_output_seconds':g['joint_and_output_seconds'],
            'bound_rows':[{'competitor':x['competitor'],'status':x['status'],
                'checked_lower_bound':x.get('checked_lower_bound'),
                'checked_lower_bound_float':float(bounds[x['competitor']]) if x['competitor'] in bounds else None}
                for x in r['rows']],
            'bundle_bytes':a['sealed']['bundle_bytes'],'manifest_sha256':a['sealed']['manifest_sha256'],
            'terminal_sha256':a['terminal_sha256']})
    old,new=bound_maps
    result['common_bound_changes']=[{'competitor':k,'new_minus_old':str(new[k]-old[k]),
        'new_minus_old_float':float(new[k]-old[k])} for k in sorted(old.keys()&new.keys())]
    result['newly_available_competitors']=sorted(new.keys()-old.keys())
    result['lost_available_competitors']=sorted(old.keys()-new.keys())
    result['descriptive_seconds_difference']=result['arms'][1]['seconds']-result['arms'][0]['seconds']
    result['complete_positive_requests']=sum(a['positive']==9 for a in result['arms'])
    return result


if __name__=='__main__':print(json.dumps(analyze(Path(sys.argv[1]).resolve()),indent=2))
