"""Read-only independent-process archival review of the frozen eight requests.

No model, solver, bound proposal, or post-deadline proof completion. Raw records
remain untouched. Execution audits are structural, not reproof of lowering.
"""
from collections import Counter
from pathlib import Path
import statistics
import time

from portable_proof.runtime import digest, compact
from single_check_portable.execution import read, save_new, ROOT
from upstream_portable.study import verify, summarize, OUTPUT, FREEZE, REVIEW

DEST=ROOT/'docs/upstream_portable_v1_execution_results.json'


def aggregates(rows):
    if len(rows)!=8 or len({r['job_id'] for r in rows})!=8:raise ValueError('missing/duplicate row')
    if {r['dataset_index'] for r in rows}!={207,209,211,214}:raise ValueError('changed inputs')
    paired=[]
    for index in (207,209,211,214):
        pair={r['arm']:r for r in rows if r['dataset_index']==index}
        if set(pair)!={'single_check','double_check'}:raise ValueError('missing arm')
        a,b=pair['single_check'],pair['double_check']
        if min(a['wall_seconds'],b['wall_seconds'])<0:raise ValueError('invalid time')
        for r in (a,b):
            if r['status']=='TIMEOUT' and r['complete_independent_check']:raise ValueError('timeout promoted')
        paired.append({'dataset_index':index,
            'complete_checks':[b['complete_independent_check'],a['complete_independent_check']],
            'wall_difference_single_minus_double':a['wall_seconds']-b['wall_seconds'],
            'positive_difference':int(a['status']=='CHECKED_CONDITIONAL')-int(b['status']=='CHECKED_CONDITIONAL')})
    by_arm={}
    for arm in ('double_check','single_check'):
        subset=[r for r in rows if r['arm']==arm]
        by_arm[arm]={'denominator':4,'status_counts':dict(Counter(r['status'] for r in subset)),
            'complete_checks':sum(r['complete_independent_check'] for r in subset),
            'conditional_positive':sum(r['status']=='CHECKED_CONDITIONAL' for r in subset),
            'mean_whole_seconds':statistics.mean(r['wall_seconds'] for r in subset)}
    return {'by_arm':by_arm,'paired':paired,
            'paired_time_difference_median':statistics.median(p['wall_difference_single_minus_double'] for p in paired),
            'whole_request_seconds_sum':sum(r['wall_seconds'] for r in rows)}


def review():
    start=time.monotonic();frozen=verify();rebuilt=summarize();saved=read(OUTPUT/'summary.json')
    if {k:v for k,v in saved.items() if k!='final_archival_audit_seconds'}!=rebuilt:
        raise ValueError('fresh summary differs')
    if saved['status']!='PASS' or len(saved['rows'])!=8:raise ValueError('execution did not finish cleanly')
    rows=[];inventory={};results={};contexts={}
    for original in saved['rows']:
        job=original['job_id'];root=OUTPUT/job;manifest=read(root/'source/manifest.json')
        phases=original['costs']['phases'];tail_names=['package','check'] if original['arm']=='single_check' else ['precheck','package','check']
        last=read(root/'tail_entered.json')['seconds'];censored=None
        for name in tail_names:
            path=root/'tail'/(name+'_stage.json')
            if path.exists():last=read(path)['end_seconds']
            else:
                if original['status']=='TIMEOUT':censored=name
                break
        row={k:original[k] for k in ('job_id','dataset_index','arm','status','complete_independent_check')}
        row.update(wall_seconds=original['costs']['whole_request_seconds'],phases=phases,
            censored_tail_stage=censored,
            post_last_complete_window_seconds=max(0,original['driver_exit_seconds']-last) if censored else None,
            censored_scope='window includes handoff/cleanup; not exact interrupted-stage runtime',
            route_pairs=manifest['routes']['feasible'],required_obligations=len(manifest['obligations']),
            generated_support_status=dict(Counter(v['status'] for v in manifest['supports'].values())),
            generated_weighted_status=dict(Counter(v.get('weighted_status','REUSED_OR_PENDING') for v in manifest['obligations'])),
            handoff=read(root/'source/handoff.json'),
            recorded_query_seconds=sum(v['seconds'] for v in original['costs']['proposal_queries'] if 'seconds' in v),
            recorded_queries=len(original['costs']['proposal_queries']),
            incomplete_query_records=sum('seconds' not in v for v in original['costs']['proposal_queries']))
        contexts[job]={'request':manifest['request'],'routes':manifest['routes'],
            'common_facts':manifest['common_facts'],'contexts':manifest['contexts']}
        if row['complete_independent_check']:
            result=read(root/'tail/check.log')['result'];results[job]=result
            row['checked_result']={k:result[k] for k in result if k!='obligations'}
            row['full_checked_result_sha256']=digest(compact(result))
        else:row['checked_result']=None
        for p in [*root.glob('*.json'),root/'source/request.json',root/'source/manifest.json',
                  root/'source/generation.json',root/'source/handoff.json',root/'source/query_log.json',
                  *root.glob('tail/*.json'),*root.glob('tail/*.log'),root/'tail/portable/bundle.json',
                  OUTPUT/(job+'_row.json'),OUTPUT/(job+'_resource.json')]:
            if p.exists():inventory[str(p.relative_to(ROOT))]=digest(p.read_bytes())
        rows.append(row)
    for p in (FREEZE,REVIEW,OUTPUT/'launch.json',OUTPUT/'summary.json'):
        inventory[str(p.relative_to(ROOT))]=digest(p.read_bytes())
    pairs=[]
    for i in range(4):
        a=f'rank{i}_single_check';b=f'rank{i}_double_check'
        pairs.append({'rank':i,'upstream_request_routes_and_source_references_equal':contexts[a]==contexts[b],
                      'both_checked_results_equal':results[a]==results[b] if a in results and b in results else None})
    return {'schema':'UPSTREAM_PORTABLE_ARCHIVE_V1','status':'PASS','issues':[],
        'execution_head':read(OUTPUT/'launch.json')['head'],'freeze_sha256':digest(FREEZE.read_bytes()),
        'original_results_unchanged':True,'real_requests':8,'rows':rows,**aggregates(rows),
        'upstream_pair_comparisons':pairs,'artifact_sha256':inventory,
        'independent_archival_review_seconds':time.monotonic()-start,
        'automatic_final_audit_seconds':saved['final_archival_audit_seconds'],
        'new_proposals_or_checks_during_archival':0,
        'trust_boundary':'supplied rational proof checker; network-to-HZ, guard lowering and route exclusions trusted; no deployed-float claim'}


if __name__=='__main__':
    import json
    if DEST.exists():raise FileExistsError('immutable archive; never overwrite')
    result=review();save_new(DEST,result)
    print(json.dumps({k:result[k] for k in ('status','by_arm','paired','upstream_pair_comparisons','independent_archival_review_seconds')},indent=2))
