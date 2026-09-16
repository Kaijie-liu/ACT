"""Freeze-only by default. Small new-input engineering comparison, not old-cohort resume."""
import argparse
import copy
import fcntl
from pathlib import Path
import time
import statistics

from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from reuse_supervised.flow import ARMS, supervise, audit

FREEZE=ROOT/'docs/reuse_supervised_v1_freeze.json'
REVIEW=ROOT/'docs/reuse_supervised_v1_selection_review.json'
RAW=ROOT/'data/moe/results/reuse_supervised_selection_20260916_v1'
OUTPUT=ROOT/'data/moe/results/reuse_supervised_comparison_20260916_v1'
POLICY={'samples':4,'requests':8,'arms':list(ARMS),'epsilon':2/255,
        'total_seconds':300,'work_seconds':298,'proposal_cap_seconds':60,'tail_reserve_seconds':80,
        'tail_cache_enabled':True,'threads':1,'workers':1,'resume':False,'retry':False,
        'scope':'new clean-only inputs, engineering development; not a confirmatory superiority test',
        'primary':'complete independent check plus paired full cost; separate missing/nonpositive/route-incomplete/timeout/error',
        'secondary':'conditional positive and complete charged cost; no promised SAFE gain',
        'arm_difference':'upstream source and exact CSR caches both OFF versus both ON; all checks/order identical',
        'resource_wait_limit_seconds':86400,'resource_poll_seconds':30,
        'archival_audit':'separate reported cost; cannot rescue late/incomplete runs'}


def hashes():
    old=read(ROOT/'docs/upstream_reuse_controls_attempt001.json')
    names=list(old['sources'])+['docs/upstream_reuse_controls_attempt001.json','docs/reuse_supervised_v1.md']
    names += [str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py')]
    return {n:digest((ROOT/n).read_bytes()) for n in sorted(set(names))}


def verify_old():
    from upstream_portable.study import verify as verify_study
    from upstream_reuse.controls import hashes as reuse_hashes
    verify_study()
    if read(ROOT/'docs/upstream_reuse_controls_attempt001.json')['sources']!=reuse_hashes():
        raise ValueError('sealed reuse source drift')


def jobs(samples):
    return [{'rank':i,'dataset_index':s['dataset_index'],'arm':ARMS[(i+j)%2],
             'job_id':f'rank{i}_{ARMS[(i+j)%2]}','position':j}
            for i,s in enumerate(samples) for j in range(2)]


def freeze(controls):
    if any(p.exists() for p in (FREEZE,REVIEW,RAW,OUTPUT)):raise FileExistsError('no overwrite/reselection')
    verify_old();control=read(controls)
    if control['status']!='PASS' or control['sources']!=hashes():raise ValueError('control gate')
    from evidence_cohort.contract import selection
    from scripts.freeze_general_evidence import provenance
    from act.pipeline.moe.freeze_conv_three_arm import exclusions,verify_records,clean_selection
    from evidence_cohort.run import resource,resource_ok
    import torch
    resource_state=resource()
    if not resource_ok(resource_state):raise RuntimeError('resources unavailable; no selection started')
    parent=selection();cfg,ident=provenance();cfg=copy.deepcopy(cfg)
    from upstream_portable.study import verify as prior_verify
    prior=prior_verify()
    # Fixed before querying any new verification endpoint: ordered next prefix, not result driven.
    cfg['selection'].update(sample_count=4,smoke_count=0,
        start_index=max(s['dataset_index'] for s in prior['samples'])+1)
    records=exclusions();excluded=verify_records(records)
    excluded |= set(parent['excluded_indices']) | {s['dataset_index'] for s in parent['samples']}
    excluded |= set(prior['excluded_indices']) | {s['dataset_index'] for s in prior['samples']}
    clean,tensors=clean_selection(cfg,excluded)
    RAW.mkdir(exist_ok=False);save_new(RAW/'exclusions.json',records)
    materialized={}
    for i,values in tensors.items():
        p=RAW/f'{i}.pt';torch.save(values,p);materialized[str(i)]={'path':str(p),'sha256':digest(p.read_bytes())}
    value={'schema':'UPSTREAM_REUSE_SUPERVISED_STUDY_V1','status':'FROZEN_NOT_EXECUTED',
        'policy':POLICY,'sources':hashes(),'controls':{'path':str(controls),'sha256':digest(controls.read_bytes())},
        'prior_freeze_sha256':digest((ROOT/'docs/upstream_portable_v1_freeze.json').read_bytes()),
        'parent_selection_sha256':digest((ROOT/'docs/general_evidence_v1_selection.json').read_bytes()),
        'selection_config':cfg,'parent_identity':ident,'dimensions':parent['dimensions'],**clean,
        'materialized_inputs':materialized,'excluded_indices':sorted(excluded),
        'exclusion_inventory':{'path':str(RAW/'exclusions.json'),'sha256':digest((RAW/'exclusions.json').read_bytes())},
        'dataset':parent['dataset'],'jobs':jobs(clean['samples']),'resource':resource_state,
        'output':str(OUTPUT),'execution_started':False}
    save_new(FREEZE,value)
    return {'status':value['status'],'indices':[s['dataset_index'] for s in clean['samples']],
            'requests':len(value['jobs']),'verification_calls':0}


def verify():
    verify_old();v=read(FREEZE)
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
    from act.pipeline.moe.freeze_conv_three_arm import verify_records,clean_selection
    from evidence_cohort.contract import selection
    import torch
    v=verify();parent=selection()
    from upstream_portable.study import verify as prior_verify
    prior=prior_verify()
    if v['prior_freeze_sha256']!=digest((ROOT/'docs/upstream_portable_v1_freeze.json').read_bytes()):raise ValueError('prior freeze changed')
    if digest((ROOT/'docs/general_evidence_v1_selection.json').read_bytes())!=v['parent_selection_sha256']:
        raise ValueError('parent selection drift')
    excluded=verify_records(read(Path(v['exclusion_inventory']['path'])))
    excluded |= set(parent['excluded_indices']) | {s['dataset_index'] for s in parent['samples']}
    excluded |= set(prior['excluded_indices']) | {s['dataset_index'] for s in prior['samples']}
    if sorted(excluded)!=v['excluded_indices']:raise ValueError('exclusion union drift')
    cfg=v['selection_config']
    if (cfg['selection']['sample_count'],cfg['selection']['smoke_count'],cfg['selection']['start_index'])!=(
            4,0,max(s['dataset_index'] for s in prior['samples'])+1):raise ValueError('selection rule drift')
    rebuilt,tensors=clean_selection(cfg,excluded)
    if any(v[k]!=obj for k,obj in rebuilt.items()):raise ValueError('clean-only reconstruction')
    if set(tensors)&excluded or set(tensors)!=set(int(k) for k in v['materialized_inputs']):raise ValueError('overlap')
    for i,expected in tensors.items():
        actual=torch.load(v['materialized_inputs'][str(i)]['path'],map_location='cpu',weights_only=True)
        if actual.keys()!=expected.keys() or any(not torch.equal(actual[k],expected[k]) for k in expected):
            raise ValueError('represented input changed')
    result={'status':'PASS','issues':[],'freeze_sha256':digest(FREEZE.read_bytes()),'samples':4,'requests':8,
            'indices':sorted(tensors),'scope':'fresh clean-only reconstruction, zero verification calls'}
    save_new(REVIEW,result);return result


def request_for(v,job,head):
    from moe_evidence.schema import classification_properties
    if job not in v['jobs']:raise ValueError('unregistered job')
    sample=v['samples'][job['rank']];dims=v['dimensions'];subject=v['subject']
    r={'schema':'WEIGHTED_TOP2_REQUEST_V1','top_k':2,**dims,'mode':'eval','tie_policy':'ANY_LEGAL_TOPK',
       'gate':'selected_softmax','epsilon':2/255,'model_state':subject['model_state'],
       'clean_prediction':sample['label'],**{k:sample[k] for k in ('center','lower','upper')},
       'properties':classification_properties(dims['classes'],sample['label'])}
    return {'method':'evidence','protocol':'UPSTREAM_REUSE_SUPERVISED_STUDY_V1','epsilon':2/255,'subject':subject,
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
        a,b=pair['reuse_on'],pair['reuse_off']
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
            from reuse_supervised.flow import costs
            row={**row,'costs':costs(OUTPUT/job['job_id'])}
        rows.append(row)
    paired=paired_summary(v['samples'],rows)
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
            'scope':'structural execution review, not independent reproof of upstream lowering'}
    return result


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','reconstruct','launch','audit'))
    p.add_argument('--controls',type=Path);a=p.parse_args()
    value=freeze(a.controls) if a.mode=='freeze' else reconstruct() if a.mode=='reconstruct' else launch() if a.mode=='launch' else summarize()
    print(json.dumps(value,sort_keys=True,indent=2))
