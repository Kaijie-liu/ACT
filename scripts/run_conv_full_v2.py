"""One authorized90-request run; fail-stop, no resume or outcome-driven retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.conv_full_v2_contract import ACT, ROOT, DEFAULT, FREEZE, identity, full_selection, request_for
from scripts.conv_three_arm_contract import read
from scripts.conv_budget_smoke_v2 import artifacts
from scripts.run_conv_three_arm import execute, wait_resources
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256, _git_value


def terminal(root,job,started,code,expired,wait,error=None):
    d=root/job['job_id'];external=job['method']=='crown'
    row={**job,'budget_seconds':300,'wall_seconds':time.monotonic()-started,'outer_timeout':expired,
         'return_code':code,'status':'TIMEOUT' if expired else 'ERROR','package':None,
         'snapshot_sha256':None,'resource_wait':wait,'request_sha256':_sha256(d/'request.json'),
         'evidence_level':'CROWN_NUMERICAL_FILTER' if external else 'HZ_POLICY_ACCEPTED'}
    if error is not None:row['error']=error
    for name,key in [('common_facts.json','snapshot_sha256'),('routes.json','routes_sha256'),('external.json','external_sha256')]:
        if (d/name).exists():row[key]=_sha256(d/name)
    if not expired and code==0 and error is None:
        try:
            if external:
                status=read(d/'external.json')['status']
                if status not in ('POSITIVE','UNSAFE','UNKNOWN'):raise ValueError('bad external status')
                row['status']=status
            else:
                p=d/'package';status=read(p/'manifest.json')['status']
                if status not in ('SAFE','UNSAFE','UNKNOWN','TIMEOUT'):raise ValueError('bad ACT status')
                row.update(status=status,package=str(p),manifest_sha256=_sha256(p/'manifest.json'))
        except Exception as exc:row['error']=repr(exc)
    row['artifacts']=artifacts(d)
    atomic_json(d/'terminal.json',row)
    row['wall_seconds']=time.monotonic()-started
    if row['wall_seconds']>300:row.update(status='TIMEOUT',outer_timeout=True,package=None)
    atomic_json(d/'terminal.json',row)
    with (root/'rows.jsonl').open('a') as handle:
        handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
    return row


def final_audits(root,env):
    results=[]
    for filename in ('audit.final.json','audit.independent.json'):
        p=subprocess.run([ACT,'-m','scripts.audit_conv_full_v2','--root',str(root),'--output',str(root/filename)],
                         cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        with (root/(filename+'.log')).open('x') as log:log.write(p.stdout)
        result=read(root/filename) if (root/filename).exists() else {'status':'FAIL','issues':['auditor did not publish']}
        results.append(result)
        if p.returncode!=0:break
    equal=len(results)==2 and results[0]==results[1] and results[0]['status']=='PASS'
    summary={'audits_match':equal,'audit_status':results[-1]['status'],'run_terminal':read(root/'run_terminal.json'),
             'audit':results[-1],'additional_experiment_queued':False,
             'git_archival':'not automatic; independently inspect and commit compact results after execution'}
    atomic_json(root/'FULL_SUMMARY.json',summary)
    completed=equal and summary['run_terminal']['state']=='EXECUTION_COMPLETED'
    atomic_json(root/'supervisor.json',{'state':'COMPLETED_AUDITED' if completed else 'STOPPED_REVIEW_REQUIRED',
        'completed':len(summary['run_terminal']['completed_job_ids']),'planned':90,
        'audits_match':equal,'additional_experiment_queued':False,'updated_unix':time.time()})
    return completed


def run(root):
    if root.resolve()!=DEFAULT:raise ValueError('only the frozen output root is authorized')
    if (Path(sys.executable).resolve()!=Path(ACT).resolve() or _git_value('branch','--show-current')!='feat/moe-route-verification'
            or _git_value('status','--porcelain') or _git_value('rev-parse','HEAD')!=_git_value('rev-parse','@{upstream}')):
        raise ValueError('clean pushed feature branch and ACT env required')
    execution=identity();value=full_selection();freeze=read(FREEZE)
    if freeze['status']!='PASS' or freeze['execution']!=execution:raise ValueError('no current independent freeze gate')
    for name,sha in freeze['parent_artifacts'].items():
        if _sha256(ROOT/name)!=sha:raise ValueError('gate evidence changed')
    head=_git_value('rev-parse','HEAD');freeze_sha=_sha256(FREEZE)
    root.mkdir(exist_ok=False)
    atomic_json(root/'runtime.json',{'schema':'conv_three_arm_full_v2','smoke':False,'execution':execution,
        'selection':value,'jobs':value['full_jobs'],'git_head':head,'freeze_sha256':freeze_sha,
        'config':{'methods':value['identities']['method_configs']},'started_unix':time.time()})
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0'}
    rows=[];error=None;jobs=value['full_jobs']
    try:
        for job in jobs:
            if identity()!=execution or full_selection()!=value or _git_value('status','--porcelain'):
                raise ValueError('source/config/cohort drift during frozen execution')
            wait=wait_resources(root,job);d=root/job['job_id'];d.mkdir();started=time.monotonic()
            atomic_json(d/'request.json',request_for(value,job,head,execution))
            try:
                code,expired=execute([ACT,'-m','scripts.conv_full_v2_worker','--root',str(d),'--started',repr(started)],
                    d/'worker.log',started,env,heartbeat=lambda pid,elapsed:atomic_json(root/'supervisor.json',
                        {'state':'RUNNING','job_id':job['job_id'],'pid':pid,'elapsed_seconds':elapsed,
                         'completed':len(rows),'planned':90,'updated_unix':time.time()}))
                row=terminal(root,job,started,code,expired,wait)
            except BaseException as exc:
                rows.append(terminal(root,job,started,None,False,wait,repr(exc)));raise
            rows.append(row)
            print(f"{len(rows)}/90 {job['job_id']} index{job['dataset_index']} {row['status']} {row['wall_seconds']:.3f}s",flush=True)
            if row['status']=='ERROR':raise RuntimeError('ERROR retained; stopping with no replacement')
    except BaseException as exc:error=repr(exc)
    finally:
        atomic_json(root/'run_terminal.json',{'state':'EXECUTION_ERROR' if error else 'EXECUTION_COMPLETED',
            'error':error,'completed_job_ids':[r['job_id'] for r in rows],
            'unattempted':jobs[len(rows):],'full_started':True,'no_follow_on_run':True})
        atomic_json(root/'supervisor.json',{'state':'AUDITING','completed':len(rows),'planned':90,'updated_unix':time.time()})
    return 0 if final_audits(root,env) else 1


if __name__=='__main__':
    def stop(signum,frame):raise KeyboardInterrupt(f'supervisor signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args()
    with (ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sys.exit(run(args.root))
