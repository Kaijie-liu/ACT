"""Saved-record archival review only; no model, solver or new bound checking."""
from collections import Counter
import copy
import statistics
import time
from portable_proof.runtime import compact,digest
from single_check_portable.execution import ROOT,read,save_new
from source_cache_ablation.study import verify,summarize,OUTPUT,FREEZE,REVIEW

DEST=ROOT/'docs/source_cache_ablation_v1_execution_results.json'


def aggregates(rows,indices):
    if len(rows)!=2*len(indices) or len({r['job_id'] for r in rows})!=len(rows):raise ValueError('row roster')
    if {r['dataset_index'] for r in rows}!=set(indices):raise ValueError('changed inputs')
    pairs=[];arms={}
    for index in indices:
        pair={r['arm']:r for r in rows if r['dataset_index']==index}
        if set(pair)!={'matrix_only','both'}:raise ValueError('missing arm')
        a,b=pair['both'],pair['matrix_only']
        for r in (a,b):
            if r['wall_seconds']<0 or r['status']=='TIMEOUT' and r['complete_independent_check']:
                raise ValueError('time/acceptance')
            c=r['checked_result']
            if c and c['required_obligations']!=sum(c[k] for k in ('positive_obligations','nonpositive_obligations','missing_obligations')):
                raise ValueError('obligation accounting')
            if r['status']=='CHECKED_CONDITIONAL' and (not c or c['positive_obligations']!=c['required_obligations']):
                raise ValueError('partial positive promoted')
        pairs.append({'dataset_index':index,'matrix_only_seconds':b['wall_seconds'],'both_seconds':a['wall_seconds'],
                      'both_minus_matrix_seconds':a['wall_seconds']-b['wall_seconds'],
                      'complete_check_difference':int(a['complete_independent_check'])-int(b['complete_independent_check']),
                      'positive_request_difference':int(a['status']=='CHECKED_CONDITIONAL')-int(b['status']=='CHECKED_CONDITIONAL')})
    for arm in ('matrix_only','both'):
        subset=[r for r in rows if r['arm']==arm];timers={}
        for r in subset:
            for name,v in r['metrics']['timings'].items():
                t=timers.setdefault(name,{'calls':0,'exclusive_seconds':0.})
                t['calls']+=v['calls'];t['exclusive_seconds']+=v['exclusive_seconds']
        arms[arm]={'denominator':len(indices),'status_counts':dict(Counter(r['status'] for r in subset)),
            'complete_checks':sum(r['complete_independent_check'] for r in subset),
            'conditional_positive_requests':sum(r['status']=='CHECKED_CONDITIONAL' for r in subset),
            'mean_whole_seconds':statistics.mean(r['wall_seconds'] for r in subset),
            'total_whole_seconds':sum(r['wall_seconds'] for r in subset),
            'checked_obligations':{k:sum(r['checked_result'][k] for r in subset if r['checked_result'])
                for k in ('required_obligations','positive_obligations','missing_obligations','nonpositive_obligations')},
            'entered_queries':sum(r['query_count'] for r in subset),
            'exclusive_upstream_timers':timers,
            'phase_seconds':{name:sum(r['phases'][name]['seconds'] for r in subset)
                for name in ('capture','propose','package','check')
                if all(r['phases'][name]['seconds'] is not None for r in subset)}}
    return {'by_arm':arms,'paired':pairs,
            'paired_cost_median_seconds':statistics.median(r['both_minus_matrix_seconds'] for r in pairs),
            'total_request_seconds':sum(r['wall_seconds'] for r in rows),
            'both_vs_matrix_total_reduction_fraction':1-arms['both']['total_whole_seconds']/arms['matrix_only']['total_whole_seconds']}


def source_compare(manifests,roots):
    a,b=manifests;ar,br=roots
    def routes(m):
        r=copy.deepcopy(m['routes'])
        for branch in r.get('branches',[]):branch.pop('elapsed',None)
        return r
    rows=[]
    if a['contexts'].keys()!=b['contexts'].keys():raise ValueError('source context inventory differs')
    for key in a['contexts']:
        item={'pair':a['contexts'][key]['pair']}
        for kind in ('joint_source','router_source'):
            refs=[m['contexts'][key][kind] for m in (a,b)];values=[]
            for root,ref in zip((ar,br),refs):
                p=root/'source'/ref['file']
                if not p.resolve().is_relative_to((root/'source').resolve()) or digest(p.read_bytes())!=ref['sha256']:
                    raise ValueError('source identity mismatch')
                values.append(read(p))
            item[kind]={'both_sha256':refs[0]['sha256'],'matrix_only_sha256':refs[1]['sha256'],
                        'equal':values[0]==values[1],
                        'differing_top_level_keys':sorted(k for k in values[0].keys()|values[1].keys() if values[0].get(k)!=values[1].get(k))}
        rows.append(item)
    return {'request_equal':a['request']==b['request'],
            'routes_equal_omitting_only_branch_elapsed':routes(a)==routes(b),
            'common_fact_reference_equal':a['common_facts']==b['common_facts'],'pairs':rows}


def review():
    start=time.monotonic();v=verify();rebuilt=summarize();saved=read(OUTPUT/'summary.json')
    if {k:x for k,x in saved.items() if k!='final_archival_audit_seconds'}!=rebuilt:raise ValueError('summary reconstruction differs')
    if saved['status']!='PASS' or len(saved['rows'])!=8:raise ValueError('incomplete/error run requires a separate archival record')
    rows=[];inventory={};manifests={}
    for original in saved['rows']:
        root=OUTPUT/original['job_id'];m=read(root/'source/manifest.json');manifests[original['job_id']]=m
        cost=original['costs'];metrics=cost['proposal_metrics'];queries=cost['proposal_queries']
        if metrics is None or any(q['status']!='PROPOSED' for q in queries):raise ValueError('current receipt expects completed proposal records')
        if metrics['timings']['dual_evaluate']['calls']!=3*len(queries):raise ValueError('exact evaluation count differs')
        row={k:original[k] for k in ('job_id','dataset_index','arm','status','complete_independent_check')}
        row.update(wall_seconds=cost['whole_request_seconds'],phases=cost['phases'],metrics=metrics,
                   residual_clock_seconds=cost['residual_clock_seconds'],query_count=len(queries),
                   route_pairs=m['routes']['feasible'],weighted_status=dict(Counter(r.get('weighted_status','REUSED_OR_PENDING') for r in m['obligations'])))
        result=read(root/'tail/check.log')['result'] if row['complete_independent_check'] else None
        row['checked_result']={k:x for k,x in result.items() if k!='obligations'} if result else None
        row['full_result_sha256']=digest(compact(result)) if result else None
        row['checked_obligations']=result['obligations'] if result else None
        if (metrics['matrix_cache']['enabled'] is not True or
            metrics['source_cache']['enabled'] is not (row['arm']=='both')):raise ValueError('incorrect arm cache configuration')
        for p in [*root.glob('*.json'),*root.glob('source/*.json'),*root.glob('tail/*.json'),
                  *root.glob('tail/*.log'),root/'tail/portable/bundle.json',
                  OUTPUT/(original['job_id']+'_row.json'),OUTPUT/(original['job_id']+'_resource.json')]:
            if p.exists():inventory[str(p.relative_to(ROOT))]=digest(p.read_bytes())
        rows.append(row)
    sources=[]
    for i,sample in enumerate(v['samples']):
        names=[f'rank{i}_{arm}' for arm in ('both','matrix_only')]
        sources.append({'dataset_index':sample['dataset_index'],
                        **source_compare([manifests[n] for n in names],[OUTPUT/n for n in names])})
    for p in (FREEZE,REVIEW,OUTPUT/'launch.json',OUTPUT/'summary.json'):
        inventory[str(p.relative_to(ROOT))]=digest(p.read_bytes())
    from pathlib import Path
    return {'schema':'SOURCE_CACHE_ATTRIBUTION_ARCHIVE_V1','status':'PASS','issues':[],
            'archival_sources':{str(p.relative_to(ROOT)):digest(p.read_bytes())
                                for p in Path(__file__).parent.glob('*.py')},
            'execution_head':read(OUTPUT/'launch.json')['head'],'freeze_sha256':digest(FREEZE.read_bytes()),
            'rows':rows,**aggregates(rows,[s['dataset_index'] for s in v['samples']]),
            'source_comparisons':sources,'artifact_sha256':inventory,
            'independent_archival_seconds':time.monotonic()-start,
            'automatic_final_audit_seconds':saved['final_archival_audit_seconds'],
            'new_solver_or_model_or_bound_check_calls':0,'original_records_unchanged':True,
            'scope':'saved-record structural review, not independent reproof of network-to-HZ, guard or route exclusion',
            'checked_evidence_comparisons':rebuilt['evidence_comparisons'],
            'frozen_arm_configuration_verified':True,
            'timing_caveat':'nested exclusive categories are observations; source-cache contrast with matrix cache ON; small observed cohort, no population speed guarantee'}


if __name__=='__main__':
    import json
    if DEST.exists():raise FileExistsError('no archive overwrite')
    result=review();save_new(DEST,result)
    print(json.dumps({k:result[k] for k in ('status','by_arm','paired','source_comparisons','independent_archival_seconds')},indent=2))
