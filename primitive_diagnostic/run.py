"""Ordered one-shot batch and independently reconstructed terminal accounting."""
from collections import Counter
import fcntl
import math
import re
from pathlib import Path
import sys
import time
from single_check_portable.execution import ROOT, ACT, read, save_new
from primitive_supervised.flow import supervise, audit, costs
from evidence_cohort.run import wait_resource
from primitive_diagnostic import contract as C


def roster(jobs):
    ids=[j['job_id'] for j in jobs]
    if len(ids)!=4 or len(set(ids))!=4 or any(not re.fullmatch(r'[a-zA-Z0-9_]+',s) for s in ids):
        raise ValueError('four distinct safe directory identities required')


def finite(value):
    if isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError('invalid cost')
    return value


def execute(jobs,root,launch_info,*,wait=wait_resource,invoke=supervise):
    """Injectable for analytic controls; CLI launch always uses frozen functions."""
    batch_begin=time.monotonic()
    root=Path(root).resolve();roster(jobs)
    if not root.is_relative_to(Path('/data1/Kane/MOE')):raise ValueError('outside workspace')
    root.mkdir(exist_ok=False)
    save_new(root/'launch.json',launch_info)
    stopped=False
    with (root/'writer.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in jobs:
            name=job['job_id']
            row={'job_id':name,'status':'NOT_RUN_AFTER_ERROR','complete_independent_check':False,
                 'error':None,'attempt_seconds':None,'post_terminal_audit_seconds':None}
            if not stopped:
                begin=time.monotonic();save_new(root/(name+'_entered.json'),{'started':begin,'job_id':name})
                events=[];wait_result=None;resource_error=None;resource_start=time.monotonic()
                try:
                    try:wait_result=wait(events.append)
                    except BaseException as exc:resource_error=repr(exc);raise
                    finally:
                        save_new(root/(name+'_resource.json'),{'events':events,'wait':wait_result,
                            'error':resource_error,'observed_seconds':time.monotonic()-resource_start})
                    start=time.monotonic()
                    save_new(root/(name+'_invocation.json'),{'original_started':start})
                    invoke(job,root/name,started=start)
                    post=time.monotonic()
                    try:
                        terminal=audit(root/name)
                        # Accounting must also be reconstructible before continuing.
                        costs(root/name)
                    finally:row['post_terminal_audit_seconds']=time.monotonic()-post
                    row.update(status=terminal['status'],complete_independent_check=terminal['complete_independent_check'])
                except BaseException as exc:
                    # Preserve partial directory; never retry or turn failure into a check.
                    row.update(status='ERROR',complete_independent_check=False,error=repr(exc))
                row['attempt_seconds']=time.monotonic()-begin
            save_new(root/(name+'_row.json'),row)
            stopped|=row['status']=='ERROR'
        save_new(root/'batch_terminal.json',{'status':'EXECUTION_ERROR' if stopped else 'EXECUTION_COMPLETED',
            'job_ids':[j['job_id'] for j in jobs],'seconds':time.monotonic()-batch_begin})
        begin=time.monotonic()
        try:
            result=summarize(jobs,root)
            save_new(root/'summary.json',result)
        except BaseException as exc:
            save_new(root/'final_audit_failure.json',{'error':repr(exc),'seconds':time.monotonic()-begin})
            raise
        save_new(root/'summary_publication.json',{'summary':C.ref(root/'summary.json'),
            'seconds':time.monotonic()-begin,'scope':'final audit plus summary serialization; outside request clocks'})
    return result


def summarize(jobs,root):
    """Strict roster/ledger replay; never proposes a new point or resumes a job."""
    roster(jobs);root=Path(root);rows=[];stopped=False
    expected={j['job_id']+'_row.json' for j in jobs}
    if {p.name for p in root.glob('*_row.json')}!=expected:raise ValueError('missing/extra terminal rows')
    for job in jobs:
        name=job['job_id'];row=read(root/(name+'_row.json'));folder=root/name
        if row['job_id']!=name or stopped!=(row['status']=='NOT_RUN_AFTER_ERROR'):
            raise ValueError('ordered error-stop roster')
        resource=None;cost=None;diag=None;terminal=None
        if stopped:
            if (folder.exists() or any((root/(name+s)).exists() for s in ('_entered.json','_resource.json','_invocation.json'))
                    or row['complete_independent_check'] or row['attempt_seconds'] is not None
                    or row['post_terminal_audit_seconds'] is not None or row['error'] is not None):
                raise ValueError('unstarted request has evidence or cost')
        else:
            entry=read(root/(name+'_entered.json'));resource=read(root/(name+'_resource.json'))
            if entry['job_id']!=name:raise ValueError('attempt binding')
            finite(entry['started'])
            attempt=finite(row['attempt_seconds']);resource_seconds=finite(resource['observed_seconds'])
            post=0 if row['post_terminal_audit_seconds'] is None else finite(row['post_terminal_audit_seconds'])
            if (resource['error'] is not None and row['status']!='ERROR') or resource_seconds+post>attempt+1e-7:
                raise ValueError('resource/post clock or failure')
            if (folder/'outer.json').exists():
                terminal=audit(folder);cost=costs(folder)
                p=read(folder/'plan.json')
                if p['spec']!=job or p['started']!=read(root/(name+'_invocation.json'))['original_started']:
                    raise ValueError('request/original-start identity')
                if not entry['started']<=p['started']<=entry['started']+attempt:
                    raise ValueError('request outside attempt clock')
                if (row['status']!=terminal['status'] or row['complete_independent_check']!=terminal['complete_independent_check']):
                    raise ValueError('outer/row terminal mismatch')
                if row['error'] is not None:raise ValueError('unresolved post-terminal failure')
                if cost['whole_supplied_LP_seconds']+resource_seconds+post>attempt+1e-7:
                    raise ValueError('double-counted or inflated attempt cost')
                if terminal['complete_independent_check']:diag=read(folder/'check.log')
            elif row['status']!='ERROR' or not row['error'] or row['complete_independent_check']:
                raise ValueError('missing outer terminal without recorded error')
            if row['complete_independent_check']!=(diag is not None):raise ValueError('spurious check')
        stopped|=row['status']=='ERROR'
        rows.append({**row,'diagnostic':diag,'costs':cost,'resource':resource,
                     'whole_request_status':None if terminal is None else terminal['status']})
    batch=read(root/'batch_terminal.json')
    if batch['job_ids']!=[j['job_id'] for j in jobs] or batch['status']!=('EXECUTION_ERROR' if stopped else 'EXECUTION_COMPLETED'):
        raise ValueError('batch terminal')
    attempt_sum=sum(r['attempt_seconds'] for r in rows if r['attempt_seconds'] is not None)
    if attempt_sum>finite(batch['seconds'])+1e-7:raise ValueError('batch cost accounting')
    from sparse_diagnostic_archive.review import aggregate
    stats=aggregate(rows)
    return {'status':'AUDITED_WITH_ERRORS' if stopped else 'PASS','denominator':4,'rows':rows,'aggregates':stats,
        'cost_totals':{'supplied_LP_seconds':stats['total_supplied_LP_seconds'],
            'resource_seconds':sum(r['resource']['observed_seconds'] for r in rows if r['resource'] is not None),
            'post_terminal_audit_seconds':sum(r['post_terminal_audit_seconds'] for r in rows if r['post_terminal_audit_seconds'] is not None),
            'attempt_seconds':attempt_sum,'batch_seconds':batch['seconds'],
            'batch_residual_seconds':batch['seconds']-attempt_sum,
            'preflight_seconds':finite(read(root/'launch.json')['preflight_seconds']),
            'scope':'nested request/attempt/batch clocks; do not add levels; final audit separately recorded'}}


def audit_saved(jobs,root):
    result=summarize(jobs,root);saved=read(root/'summary.json');p=read(root/'summary_publication.json')
    if result!=saved or p['summary']!=C.ref(root/'summary.json'):raise ValueError('summary/publication drift')
    finite(p['seconds'])
    return result


def launch_gate():
    from scripts.optional_evidence_dev_contract import git
    if Path(sys.executable).resolve()!=Path(ACT).resolve():raise ValueError('ACT interpreter required')
    v=C.verify();r=read(C.REVIEW)
    if r['status']!='PASS' or r['issues'] or r['freeze']!=C.ref(C.FREEZE) or r['sources']!=C.sources():
        raise ValueError('selection review gate')
    if C.OUTPUT.exists():raise FileExistsError('no resume/retry')
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):
        raise ValueError('clean feature checkout required')
    head=git('rev-parse','HEAD')
    if head!=git('ls-remote','origin','refs/heads/feat/moe-route-verification').split()[0]:
        raise ValueError('publish execution sources before launch')
    return v,head


def launch():
    begin=time.monotonic()
    # Same writer lock as other ACT request batches, held across final audit.
    with (ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        v,head=launch_gate()
        return execute(v['jobs'],C.OUTPUT,{'head':head,'freeze':C.ref(C.FREEZE),'review':C.ref(C.REVIEW),
            'preflight_seconds':time.monotonic()-begin,'schema':'PRIMITIVE_REAL_LAUNCH_V1'})


if __name__=='__main__':
    import argparse,json,os
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('freeze','review','audit','launch'))
    p.add_argument('--controls',type=Path);p.add_argument('--execute-frozen',action='store_true');a=p.parse_args()
    if a.mode=='launch' and not a.execute_frozen:p.error('launch requires explicit --execute-frozen')
    os.nice(10)
    value=C.freeze(a.controls) if a.mode=='freeze' else C.reconstruct() if a.mode=='review' else launch() if a.mode=='launch' else audit_saved(C.verify()['jobs'],C.OUTPUT)
    print(json.dumps(value,indent=2))
