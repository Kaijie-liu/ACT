"""Freeze four ordered saved LPs; launch is a separate explicit action."""
import argparse
import fcntl
from pathlib import Path
import time
from collections import Counter
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity,rational
from sparse_supervised.flow import supervise,audit,costs
from sparse_basis.engine import POLICY as COMPONENT_POLICY

FREEZE=ROOT/'docs/sparse_supervised_real_v1_freeze.json'
REVIEW=ROOT/'docs/sparse_supervised_real_v1_selection_review.json'
CONTROL_REVIEW=ROOT/'docs/sparse_supervised_v1_review.json'
OUTPUT=ROOT/'data/moe/results/sparse_supervised_real_20260919_v1'
ANALYSIS=ROOT/'docs/nonpositive_v1_analysis.json'
ARCHIVE=ROOT/'docs/reuse_supervised_v1_execution_results.json'
POLICY={'requests':4,'indices':[220,222,230,232],
    'selection':'same four source LPs and original obligations as sealed lp_diagnostic_v1; no new property selection',
    'total_seconds':300,'work_seconds':298,'proposal_deadline_seconds':218,'native_cap_seconds':10,
    'workers':1,'threads':1,'retry':False,'resume':False,'basis_attempts':1,
    'upper_feasibility_tolerance':0,'representation_changed':False,
    'construction':'new sparse original-coordinate basis; no old point reuse, alternate bases or retries',
    'decision':'original-LP exact feasibility only; U<=0 constrains this LP relaxation, not network UNSAFE',
    'prior_lower':'reported frozen context only; no dual proposal, optimality or gap claim',
    'scope':'four supplied-LP development diagnostics, not complete MoE requests or performance comparison',
    'historical_network_propagation':'not rerun; excluded from clock; no full-network timing claim'}


def verify_old():
    from sparse_supervised.controls import old
    old()


def hashes():
    from sparse_supervised.flow import sources
    return sources()


def select():
    from lp_diagnostic.study import select as original_selection, verify as original_verify
    original = original_verify()
    jobs = original_selection()
    if jobs != original['jobs']:
        raise ValueError('sealed ordered LP selection drift')
    return jobs


def compatibility(jobs):
    from sparse_basis.engine import dimensions
    from sparse_supervised.inputs import load
    result=[]
    for j in jobs:
        v=load(j,lambda:None)
        n,ne,na=dimensions(v['lp'])
        result.append({'job_id':j['job_id'],'variables':n,'E_rows':ne,'A_rows':na,
                       'stored_nnz':sum(len(v['lp'][k]['data']) for k in ('E','A')),
                       'static_size_policy':'WITHIN_NEW_COMPONENT_CAPS',
                       'native_basis_mapping':'UNMEASURED','fill_bits_rank_runtime':'UNMEASURED'})
    return result


def runtime():
    import sys
    import highspy
    import highspy._core as core
    from sparse_basis.native import VERSION
    version=highspy.Highs().version()
    if version!=VERSION:raise ValueError('untested highspy version')
    return {'interpreter':sys.executable,'python':sys.version,
            'native_version':version,'native_binary_sha256':digest(Path(core.__file__).read_bytes())}


def freeze(controls):
    if any(p.exists() for p in (FREEZE,REVIEW,OUTPUT)):raise FileExistsError('no overwrite')
    verify_old();c=read(controls)
    if c['status']!='PASS' or c['sources']!=hashes():raise ValueError('controls not current')
    reviewed=read(CONTROL_REVIEW)
    if (reviewed['status']!='PASS' or reviewed['issues'] or reviewed['sources']!=hashes() or
            reviewed['controls_sha256']!=digest(controls.read_bytes())):
        raise ValueError('fresh control review gate')
    from evidence_cohort.run import resource,resource_ok
    r=resource()
    if not resource_ok(r):raise RuntimeError('resource gate before read-only selection')
    jobs=select();v={'schema':'FROZEN_SPARSE_BASIS_DIAGNOSTIC_V1','status':'FROZEN_NOT_EXECUTED','policy':POLICY,
        'sources':hashes(),'controls':{'path':str(controls),'sha256':digest(controls.read_bytes())},
        'control_review':{'path':str(CONTROL_REVIEW),'sha256':digest(CONTROL_REVIEW.read_bytes())},
        'jobs':jobs,'compatibility':compatibility(jobs),'runtime':runtime(),
        'component_policy':dict(COMPONENT_POLICY),
        'output':str(OUTPUT),'resource':r,'real_solver_calls':0}
    save_new(FREEZE,v);return {'status':v['status'],'jobs':[r['job_id'] for r in jobs],'real_solver_calls':0}


def verify():
    verify_old();v=read(FREEZE)
    if v['sources']!=hashes() or v['policy']!=POLICY or v['output']!=str(OUTPUT) or v['runtime']!=runtime() or v['component_policy']!=COMPONENT_POLICY:raise ValueError('freeze drift')
    c=v['controls'];receipt=read(Path(c['path']))
    if digest(Path(c['path']).read_bytes())!=c['sha256'] or receipt['status']!='PASS' or receipt['sources']!=v['sources']:
        raise ValueError('control binding')
    reviewed=read(CONTROL_REVIEW)
    if (v['control_review']!={'path':str(CONTROL_REVIEW),'sha256':digest(CONTROL_REVIEW.read_bytes())} or
            reviewed['sources']!=v['sources'] or reviewed['status']!='PASS' or reviewed['issues'] or
            reviewed['controls_sha256']!=c['sha256']):
        raise ValueError('control review binding')
    if len(v['jobs'])!=4 or len({j['job_id'] for j in v['jobs']})!=4:raise ValueError('roster')
    for j in v['jobs']:
        if digest(Path(j['export']['path']).read_bytes())!=j['export']['sha256'] or identity(j['statement'])!=j['statement_sha256']:
            raise ValueError('LP/statement drift')
    return v


def reconstruct():
    v=verify()
    if v['jobs']!=select() or v['compatibility']!=compatibility(v['jobs']):raise ValueError('ordered selection/size reconstruction')
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
            whole_sum+=row['costs']['whole_supplied_LP_seconds']
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
