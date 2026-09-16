"""Same-input, separately frozen source-cache attribution. Never resumes its parent."""
import argparse
import copy
import fcntl
from pathlib import Path
import time
import statistics

from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from source_cache_ablation.flow import ARMS, supervise, audit

FREEZE=ROOT/'docs/source_cache_ablation_v1_freeze.json'
REVIEW=ROOT/'docs/source_cache_ablation_v1_selection_review.json'
PARENT_FREEZE=ROOT/'docs/reuse_supervised_v1_freeze.json'
PARENT_ARCHIVE=ROOT/'docs/reuse_supervised_v1_execution_results.json'
OUTPUT=ROOT/'data/moe/results/source_cache_ablation_comparison_20260916_v1'
POLICY={'samples':4,'requests':8,'arms':list(ARMS),'epsilon':2/255,
        'total_seconds':300,'work_seconds':298,'proposal_cap_seconds':60,'tail_reserve_seconds':80,
        'tail_cache_enabled':True,'threads':1,'workers':1,'resume':False,'retry':False,
        'scope':'same four observed inputs; separate source-cache attribution follow-up, NOT new-input confirmation',
        'primary':'complete independent check plus paired full cost; separate missing/nonpositive/route-incomplete/timeout/error',
        'secondary':'conditional positive and complete charged cost; no promised SAFE gain',
        'arm_difference':'matrix cache ON in both; source cache OFF versus ON; all checks/order identical',
        'representation_changed':False,'query_order_changed':False,'default_changed':False,
        'decision_rule':'descriptive paired completion, conditional-positive and full cost; no default switch on a timer category alone; missing/differing evidence must be disclosed',
        'resource_wait_limit_seconds':86400,'resource_poll_seconds':30,
        'archival_audit':'separate reported cost; cannot rescue late/incomplete runs'}


def hashes():
    parent=read(PARENT_FREEZE)
    names=list(parent['sources'])+['docs/reuse_supervised_v1_freeze.json',
        'docs/reuse_supervised_v1_selection_review.json','docs/reuse_supervised_v1_execution_results.json',
        'docs/source_cache_ablation_v1.md']
    names += [str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py')]
    return {n:digest((ROOT/n).read_bytes()) for n in sorted(set(names))}


def verify_old():
    from reuse_supervised.study import verify as parent_verify
    parent_verify()


def jobs(samples):
    return [{'rank':i,'dataset_index':s['dataset_index'],'arm':ARMS[(i+j)%2],
             'job_id':f'rank{i}_{ARMS[(i+j)%2]}','position':j}
            for i,s in enumerate(samples) for j in range(2)]


def freeze(controls):
    if any(p.exists() for p in (FREEZE,REVIEW,OUTPUT)):raise FileExistsError('no overwrite or repeat')
    verify_old();control=read(controls)
    if control['status']!='PASS' or control['sources']!=hashes():raise ValueError('control gate')
    parent=read(PARENT_FREEZE);archive=read(PARENT_ARCHIVE)
    if archive['status']!='PASS' or archive['freeze_sha256']!=digest(PARENT_FREEZE.read_bytes()):
        raise ValueError('parent archive binding')
    if [s['dataset_index'] for s in parent['samples']]!=[220,222,230,232]:
        raise ValueError('no expansion or selection permitted')
    # Identical materialized tensors and subject, not a fresh clean/route scan.
    value=copy.deepcopy(parent)
    value.update(schema='SOURCE_CACHE_ATTRIBUTION_STUDY_V1',status='FROZEN_NOT_EXECUTED',
        policy=POLICY,sources=hashes(),controls={'path':str(controls),'sha256':digest(controls.read_bytes())},
        parent_freeze_sha256=digest(PARENT_FREEZE.read_bytes()),
        parent_archive_sha256=digest(PARENT_ARCHIVE.read_bytes()),
        jobs=jobs(parent['samples']),output=str(OUTPUT),execution_started=False,
        new_selection=False,new_verification_calls=0)
    save_new(FREEZE,value)
    return {'status':value['status'],'indices':[s['dataset_index'] for s in value['samples']],
            'requests':8,'verification_calls':0,'new_selection':False}


def verify():
    verify_old();v=read(FREEZE)
    parent=read(PARENT_FREEZE)
    if v['parent_freeze_sha256']!=digest(PARENT_FREEZE.read_bytes()) or v['parent_archive_sha256']!=digest(PARENT_ARCHIVE.read_bytes()):raise ValueError('parent binding')
    for key in ('samples','materialized_inputs','subject','dimensions','parent_identity','dataset','selection_config','excluded_indices','exclusion_inventory'):
        if v[key]!=parent[key]:raise ValueError('same-input identity changed: '+key)
    if v['policy']!=POLICY or v['sources']!=hashes() or v['jobs']!=jobs(v['samples']) or len(v['samples'])!=4:
        raise ValueError('study freeze drift')
    if v['execution_started'] or v['output']!=str(OUTPUT):raise ValueError('execution identity')
    c=v['controls'];receipt=read(Path(c['path']))
    if digest(Path(c['path']).read_bytes())!=c['sha256'] or receipt['status']!='PASS' or receipt['sources']!=v['sources']:
        raise ValueError('control receipt drift')
    for record in [v['exclusion_inventory'],v['dataset'],
                   {'path':v['subject']['checkpoint'],'sha256':v['subject']['checkpoint_sha256']},
                   *v['materialized_inputs'].values()]:
        if digest(Path(record['path']).read_bytes())!=record['sha256']:raise ValueError('input/source file drift')
    return v


def reconstruct():
    v=verify();parent=read(PARENT_FREEZE)
    if v['parent_archive_sha256']!=digest(PARENT_ARCHIVE.read_bytes()):raise ValueError('archive changed')
    if v['parent_freeze_sha256']!=digest(PARENT_FREEZE.read_bytes()):raise ValueError('parent changed')
    for key in ('samples','materialized_inputs','subject','dimensions','parent_identity',
                'dataset','selection_config','excluded_indices','exclusion_inventory'):
        if v[key]!=parent[key]:raise ValueError('same-input follow-up drift: '+key)
    if v['new_selection'] is not False or v['new_verification_calls']!=0:
        raise ValueError('not a same-input freeze')
    result={'status':'PASS','issues':[],'freeze_sha256':digest(FREEZE.read_bytes()),
            'samples':4,'requests':8,'indices':[s['dataset_index'] for s in v['samples']],
            'scope':'separate-process exact parent-selection and tensor-hash reconstruction; no model or solver calls'}
    save_new(REVIEW,result);return result


def request_for(v,job,head):
    from moe_evidence.schema import classification_properties
    if job not in v['jobs']:raise ValueError('unregistered job')
    sample=v['samples'][job['rank']];dims=v['dimensions'];subject=v['subject']
    r={'schema':'WEIGHTED_TOP2_REQUEST_V1','top_k':2,**dims,'mode':'eval','tie_policy':'ANY_LEGAL_TOPK',
       'gate':'selected_softmax','epsilon':2/255,'model_state':subject['model_state'],
       'clean_prediction':sample['label'],**{k:sample[k] for k in ('center','lower','upper')},
       'properties':classification_properties(dims['classes'],sample['label'])}
    return {'method':'evidence','protocol':'SOURCE_CACHE_ATTRIBUTION_STUDY_V1','epsilon':2/255,'subject':subject,
            'sample':sample,'tensors':v['materialized_inputs'][str(sample['dataset_index'])],
            'config':v['parent_identity']['method_configs']['monolithic'],'head':head,'evidence_request':r}


def loop(roster,run_one,publish):
    stopped=False;rows=[]
    for job in roster:
        result={'status':'NOT_RUN_AFTER_ERROR','complete_independent_check':False} if stopped else run_one(job)
        row={**job,**result};publish(row);rows.append(row)
        if row['status']=='ERROR':stopped=True
    return rows


def paired_summary(samples, rows):
    paired=[]
    for i,sample in enumerate(samples):
        pair={r['arm']:r for r in rows if r['rank']==i}
        if set(pair)!=set(ARMS):raise ValueError('missing paired terminal')
        a,b=pair['both'],pair['matrix_only']
        measured='costs' in a and 'costs' in b
        paired.append({'rank':i,'dataset_index':sample['dataset_index'],
            'complete_check_difference':int(a['complete_independent_check'])-int(b['complete_independent_check']),
            'conditional_positive_difference':int(a['status']=='CHECKED_CONDITIONAL')-int(b['status']=='CHECKED_CONDITIONAL'),
            'whole_cost_difference_seconds':a['costs']['whole_request_seconds']-b['costs']['whole_request_seconds'] if measured else None,
            'cost_pair_observed':measured})
    return paired


def launch():
    """Explicit opt-in only; freeze and independent reconstruction must already pass."""
    from scripts.optional_evidence_dev_contract import git
    from evidence_cohort.run import wait_resource
    v=verify();review=read(REVIEW)
    if review['status']!='PASS' or review['freeze_sha256']!=digest(FREEZE.read_bytes()):raise ValueError('selection gate')
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):
        raise ValueError('clean feature branch required')
    head=git('rev-parse','HEAD')
    if head!=git('rev-parse','@{u}'):raise ValueError('commit/push before execution')
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT/'writer.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save_new(OUTPUT/'launch.json',{'head':head,'freeze_sha256':digest(FREEZE.read_bytes()),
                                     'review_sha256':digest(REVIEW.read_bytes())})
        def run_one(job):
            try:
                events=[];wait=wait_resource(events.append)
                save_new(OUTPUT/(job['job_id']+'_resource.json'),{'events':events,'wait':wait})
                # Start before request construction/serialization, imports and process startup.
                started=time.monotonic();req=request_for(v,job,head)
                value=supervise(req,job['arm'],OUTPUT/job['job_id'],started=started)
                begin=time.monotonic();checked=audit(OUTPUT/job['job_id'])
                if (checked['status'],checked['complete_independent_check'])!=(value['status'],value['complete_independent_check']):
                    raise ValueError('post-execution audit mismatch')
                return {**checked,'archival_audit_seconds':time.monotonic()-begin}
            except Exception as exc:
                return {'status':'ERROR','error':repr(exc),'complete_independent_check':False}
        loop(v['jobs'],run_one,lambda row:save_new(OUTPUT/(row['job_id']+'_row.json'),row))
    begin=time.monotonic();result=summarize();result['final_archival_audit_seconds']=time.monotonic()-begin
    save_new(OUTPUT/'summary.json',result)
    return result


def summarize():
    from collections import Counter
    v=verify();launch=read(OUTPUT/'launch.json')
    if launch['freeze_sha256']!=digest(FREEZE.read_bytes()) or launch['review_sha256']!=digest(REVIEW.read_bytes()):
        raise ValueError('launch binding')
    rows=[];stopped=False
    for job in v['jobs']:
        row=read(OUTPUT/(job['job_id']+'_row.json'))
        if any(row[k]!=val for k,val in job.items()):raise ValueError('roster drift')
        if row['status']=='NOT_RUN_AFTER_ERROR':
            if not stopped:raise ValueError('silent skipped request')
        elif stopped:raise ValueError('continued after error')
        elif (OUTPUT/job['job_id']/'outer.json').exists():
            checked=audit(OUTPUT/job['job_id'])
            if row['status']!='ERROR' and any(row[k]!=checked[k] for k in checked):raise ValueError('terminal mismatch')
            p=read(OUTPUT/job['job_id']/'plan.json')
            if p['request']!=request_for(v,job,launch['head']) or p['arm']!=job['arm']:raise ValueError('wrong request')
        elif row['status']!='ERROR':raise ValueError('no terminal')
        stopped |= row['status']=='ERROR'
        if (OUTPUT/job['job_id']/'publication.json').exists():
            from source_cache_ablation.flow import costs
            row={**row,'costs':costs(OUTPUT/job['job_id'])}
        rows.append(row)
    paired=paired_summary(v['samples'],rows)
    # Compare available evidence, not just endpoint status. Missing is NOT equal.
    evidence=[]
    from reuse_archive.review import source_compare
    for rank,sample in enumerate(v['samples']):
        roots=[OUTPUT/f'rank{rank}_{arm}' for arm in ('both','matrix_only')]
        item={'dataset_index':sample['dataset_index'],'source_comparison':None,
              'checked_rows_equal':None,'checked_rows_compared':0}
        if all((r/'source/manifest.json').exists() for r in roots):
            mm=[read(r/'source/manifest.json') for r in roots]
            item['source_comparison']=source_compare(mm,roots)
        terminal={r['arm']:r for r in rows if r['rank']==rank}
        if all(terminal[a]['complete_independent_check'] for a in ARMS):
            checks=[read(r/'tail/check.log')['result'] for r in roots]
            lookup=[{(tuple(x['pair']),x['property_index']):x for x in c['obligations']} for c in checks]
            common=lookup[0].keys()&lookup[1].keys()
            item.update(checked_rows_compared=len(common),
                checked_rows_equal=all(lookup[0][k]==lookup[1][k] for k in common),
                check_results_equal=checks[0]==checks[1],
                checked_result_sha256=[digest((r/'tail/check.log').read_bytes()) for r in roots])
        evidence.append(item)
    measured=[r['whole_cost_difference_seconds'] for r in paired if r['cost_pair_observed']]
    cost_summary={}
    for arm in ARMS:
        times=[r['costs']['whole_request_seconds'] for r in rows if r['arm']==arm and 'costs' in r]
        cost_summary[arm]={'observed_requests':len(times),'missing_cost_requests':4-len(times),
                           'mean_seconds':statistics.mean(times) if times else None,'sum_seconds':sum(times) if times else None}
    result={'status':'AUDITED_WITH_ERRORS' if stopped else 'PASS','requests':8,'rows':rows,'paired':paired,
            'counts':{a:dict(Counter(r['status'] for r in rows if r['arm']==a)) for a in ARMS},
            'cost_summary':cost_summary,'paired_cost_observed':len(measured),
            'paired_cost_median_seconds':statistics.median(measured) if measured else None,
            'evidence_comparisons':evidence,
            'scope':'same observed inputs, descriptive source-cache attribution; structural audit not independent upstream reproof'}
    return result


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','reconstruct','launch','audit'))
    p.add_argument('--controls',type=Path);a=p.parse_args()
    value=freeze(a.controls) if a.mode=='freeze' else reconstruct() if a.mode=='reconstruct' else launch() if a.mode=='launch' else summarize()
    print(json.dumps(value,sort_keys=True,indent=2))
