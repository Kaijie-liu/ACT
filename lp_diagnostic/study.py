"""Freeze four ordered saved LPs; launch is a separate explicit action."""
import argparse
import fcntl
from pathlib import Path
import time
from collections import Counter
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity,rational
from lp_diagnostic.flow import supervise,audit,costs

FREEZE=ROOT/'docs/lp_diagnostic_v1_freeze.json'
REVIEW=ROOT/'docs/lp_diagnostic_v1_selection_review.json'
OUTPUT=ROOT/'data/moe/results/lp_diagnostic_20260919_v1'
ANALYSIS=ROOT/'docs/nonpositive_v1_analysis.json'
ARCHIVE=ROOT/'docs/reuse_supervised_v1_execution_results.json'
POLICY={'requests':4,'indices':[220,222,230,232],
    'selection':'first nonpositive property in original pair/property order per observed input',
    'total_seconds':300,'work_seconds':298,'proposal_deadline_seconds':218,'native_cap_seconds':60,
    'tail_reserve_seconds':80,'workers':1,'threads':1,'retry':False,'resume':False,
    'upper_feasibility_tolerance':0,'repair':False,'representation_changed':False,
    'scope':'single supplied-LP diagnostics; not complete MoE requests, speed comparison or confirmation cohort',
    'decision':'checked feasible nonpositive objective limits this LP only; absent/nonexact primal remains unresolved',
    'historical_network_propagation':'not rerun; excluded from diagnostic clock; no full-network timing claim'}


def verify_old():
    from source_cache_ablation.study import verify
    from lp_sandwich.controls import sources
    verify();c=read(ROOT/'docs/lp_sandwich_controls_attempt002.json')
    if c['status']!='PASS' or c['sources']!=sources():raise ValueError('LP component drift')


def hashes():
    parent=read(ROOT/'docs/source_cache_ablation_v1_freeze.json')['sources']
    names=set(parent)|set(read(ROOT/'docs/lp_sandwich_controls_attempt002.json')['sources'])
    names.update(['docs/lp_sandwich_controls_attempt002.json','docs/lp_diagnostic_v1.md',
                  str(ANALYSIS.relative_to(ROOT)),str(ARCHIVE.relative_to(ROOT))])
    names.update(str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py'))
    return {n:digest((ROOT/n).read_bytes()) for n in sorted(names)}


def select():
    a=read(ANALYSIS);ar=read(ARCHIVE)
    if a['archive_sha256']!=digest(ARCHIVE.read_bytes()) or a['nonpositive']!=30:raise ValueError('parent analysis binding')
    result=[];raw=ROOT/'data/moe/results/reuse_supervised_comparison_20260916_v1'
    for rank,index in enumerate(POLICY['indices']):
        rows=sorted([r for r in a['obligations'] if r['dataset_index']==index and not r['positive']],
                    key=lambda r:(r['pair'],r['property_index']))
        row=rows[0];key=(row['pair'],row['property_index']);base=raw/f'rank{rank}_reuse_on/source'
        manifest_path=base/'manifest.json'
        if digest(manifest_path.read_bytes())!=ar['artifact_sha256'][str(manifest_path.relative_to(ROOT))]:raise ValueError('parent manifest drift')
        m=read(manifest_path)
        obs=[v for v in m['obligations'] if (v['pair'],v['property_index'])==key]
        if len(obs)!=1:raise ValueError('parent obligation coverage')
        ob=obs[0];p=(base/ob['weighted']['file']).resolve()
        if not p.is_relative_to(base.resolve()):raise ValueError('export path')
        sha=digest(p.read_bytes())
        if sha!=row['weighted_export_sha256'] or sha!=ob['weighted']['sha256'] or sha!=ar['artifact_sha256'][str(p.relative_to(ROOT))]:
            raise ValueError('export binding')
        ex=read(p)
        if identity(ex['source'])!=ex['source_sha256']:raise ValueError('source identity')
        s={'schema':'LP_OBLIGATION_IDENTITY_V1','request_id':m['request_id'],
           'source_sha256':ex['source_sha256'],'export_sha256':sha,'pair':row['pair'],
           'property_index':row['property_index'],'property':ob['property'],
           'lp_sha256':identity(ex['lp']),'acceptance_threshold':m['positive_threshold']}
        if m['request_id']!=row['request_id'] or rational(row['lower']['exact'])>rational(m['positive_threshold']):
            raise ValueError('parent request/negative selection')
        result.append({'job_id':f'input{index}_p{row["property_index"]}','dataset_index':index,
            'export':{'path':str(p),'sha256':sha},'statement':s,'statement_sha256':identity(s),
            'prior_checked_lower':row['lower']['exact'],
            'prior_classification':row['attribution']})
    return result


def freeze(controls):
    if any(p.exists() for p in (FREEZE,REVIEW,OUTPUT)):raise FileExistsError('no overwrite')
    verify_old();c=read(controls)
    if c['status']!='PASS' or c['sources']!=hashes():raise ValueError('controls not current')
    from evidence_cohort.run import resource,resource_ok
    r=resource()
    if not resource_ok(r):raise RuntimeError('resource gate before read-only selection')
    jobs=select();v={'schema':'FROZEN_LP_DIAGNOSTIC_V1','status':'FROZEN_NOT_EXECUTED','policy':POLICY,
        'sources':hashes(),'controls':{'path':str(controls),'sha256':digest(controls.read_bytes())},
        'jobs':jobs,'output':str(OUTPUT),'resource':r,'real_solver_calls':0}
    save_new(FREEZE,v);return {'status':v['status'],'jobs':[r['job_id'] for r in jobs],'real_solver_calls':0}


def verify():
    verify_old();v=read(FREEZE)
    if v['sources']!=hashes() or v['policy']!=POLICY or v['output']!=str(OUTPUT):raise ValueError('freeze drift')
    c=v['controls'];receipt=read(Path(c['path']))
    if digest(Path(c['path']).read_bytes())!=c['sha256'] or receipt['status']!='PASS' or receipt['sources']!=v['sources']:
        raise ValueError('control binding')
    if len(v['jobs'])!=4 or len({j['job_id'] for j in v['jobs']})!=4:raise ValueError('roster')
    for j in v['jobs']:
        if digest(Path(j['export']['path']).read_bytes())!=j['export']['sha256'] or identity(j['statement'])!=j['statement_sha256']:
            raise ValueError('LP/statement drift')
    return v


def reconstruct():
    v=verify()
    if v['jobs']!=select():raise ValueError('ordered selection reconstruction')
    r={'status':'PASS','issues':[],'freeze_sha256':digest(FREEZE.read_bytes()),
       'jobs':[j['job_id'] for j in v['jobs']],'real_solver_calls':0,'scope':'separate-process source/selection identity review only'}
    save_new(REVIEW,r);return r


def loop(jobs,run,publish):
    rows=[];stopped=False
    for job in jobs:
        if stopped:r={'status':'NOT_RUN_AFTER_ERROR','complete_independent_check':False}
        else:
            try:r=run(job)
            except Exception as e:r={'status':'ERROR','error':repr(e),'complete_independent_check':False}
        row={'job_id':job['job_id'],**r};publish(row);rows.append(row)
        stopped|=row['status']=='ERROR'
    return rows


def summarize():
    v=verify();launch=read(OUTPUT/'launch.json')
    if launch['freeze_sha256']!=digest(FREEZE.read_bytes()) or launch['review_sha256']!=digest(REVIEW.read_bytes()):raise ValueError('launch binding')
    rows=[];stopped=False;wait_sum=0;post_sum=0;whole_sum=0
    for job in v['jobs']:
        row=read(OUTPUT/(job['job_id']+'_row.json'));root=OUTPUT/job['job_id']
        if row['job_id']!=job['job_id']:raise ValueError('roster drift')
        if stopped:
            if row['status']!='NOT_RUN_AFTER_ERROR':raise ValueError('continued after error')
        elif row['status']=='NOT_RUN_AFTER_ERROR':raise ValueError('silent omitted job')
        elif (root/'outer.json').exists():
            a=audit(root)
            if row['status']!='ERROR' and any(row[k]!=a[k] for k in a):raise ValueError('terminal reconstruction')
            if row['status']=='ERROR' and a['status']!='ERROR' and not row.get('error'):raise ValueError('unexplained row error')
            if read(root/'plan.json')['spec']!=job:raise ValueError('request changed')
            row={**row,'costs':costs(root)}
            whole_sum+=row['costs']['whole_diagnostic_seconds']
            if row['status']=='CHECKED_LP_DIAGNOSTIC' and a['complete_independent_check']:
                row['diagnostic']=read(root/'check.log')
        elif row['status']!='ERROR':raise ValueError('missing terminal')
        resource_path=OUTPUT/(job['job_id']+'_resource.json')
        if row['status']!='NOT_RUN_AFTER_ERROR':
            resource=read(resource_path)
            if resource['observed_seconds']<0:raise ValueError('resource accounting')
            row['resource_accounting']=resource;wait_sum+=resource['observed_seconds']
        post=row.get('post_terminal_audit_seconds')
        if post is not None:
            if post<0:raise ValueError('post-audit cost')
            post_sum+=post
        stopped|=row['status']=='ERROR';rows.append(row)
    return {'status':'AUDITED_WITH_ERRORS' if stopped else 'PASS','denominator':4,'rows':rows,
        'status_counts':dict(Counter(r['status'] for r in rows)),
        'classification_counts':dict(Counter(r['diagnostic']['classification'] for r in rows if 'diagnostic' in r)),
        'cost_totals':{'diagnostic_publication_seconds':whole_sum,'resource_seconds':wait_sum,
            'preflight_seconds':launch['preflight_seconds'],'post_terminal_audit_seconds':post_sum,
            'post_terminal_audits_with_record':sum(r.get('post_terminal_audit_seconds') is not None for r in rows),
            'diagnostics_with_cost_record':sum('costs' in r for r in rows),
            'scope':'disjoint diagnostic publication clocks and separately labeled overhead; missing clocks not zero'},
        'scope':'single-LP diagnostic only, not four complete MoE proofs'}


def audit_saved():
    fresh=summarize();saved=read(OUTPUT/'summary.json')
    if saved.get('final_audit_seconds',-1)<0:raise ValueError('missing final audit time')
    if {k:v for k,v in saved.items() if k!='final_audit_seconds'}!=fresh:
        raise ValueError('saved aggregate/cost drift')
    return fresh


def launch():
    from scripts.optional_evidence_dev_contract import git
    from evidence_cohort.run import wait_resource
    t=time.monotonic();v=verify();r=read(REVIEW)
    if r['status']!='PASS' or r['freeze_sha256']!=digest(FREEZE.read_bytes()):raise ValueError('review gate')
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):raise ValueError('clean feature branch required')
    head=git('rev-parse','HEAD')
    if head!=git('ls-remote','origin','refs/heads/feat/moe-route-verification').split()[0]:raise ValueError('push before launch')
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT/'writer.lock').open('x') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save_new(OUTPUT/'launch.json',{'head':head,'freeze_sha256':digest(FREEZE.read_bytes()),
            'review_sha256':digest(REVIEW.read_bytes()),'preflight_seconds':time.monotonic()-t})
        def run(job):
            events=[];wait=None;resource_error=None;resource_started=time.monotonic()
            try:wait=wait_resource(events.append)
            except Exception as exc:resource_error=repr(exc);raise
            finally:
                save_new(OUTPUT/(job['job_id']+'_resource.json'),{'wait':wait,'events':events,
                    'observed_seconds':time.monotonic()-resource_started,'error':resource_error})
            started=time.monotonic();supervise(job,OUTPUT/job['job_id'],started=started)
            begin=time.monotonic();row=audit(OUTPUT/job['job_id'])
            return {**row,'post_terminal_audit_seconds':time.monotonic()-begin}
        loop(v['jobs'],run,lambda row:save_new(OUTPUT/(row['job_id']+'_row.json'),row))
    begin=time.monotonic();result=summarize();result['final_audit_seconds']=time.monotonic()-begin
    save_new(OUTPUT/'summary.json',result);return result


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('freeze','reconstruct','launch','audit'));p.add_argument('--controls',type=Path)
    a=p.parse_args();v=freeze(a.controls) if a.mode=='freeze' else reconstruct() if a.mode=='reconstruct' else launch() if a.mode=='launch' else audit_saved()
    print(json.dumps(v,indent=2,sort_keys=True))
