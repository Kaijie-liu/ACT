"""Frozen four-request V2 smoke. No full, retry or resume entry point."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.conv_three_arm_contract import ACT, ROOT, selection, request_for, wrapper_hashes, read
from scripts.conv_budget_worker_v2 import execution_identity
from scripts.run_conv_three_arm import execute, wait_resources
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256, _git_value

DEFAULT = ROOT/'data/moe/results/conv_budget_smoke_20260915_v2'
FILES = ('scripts/conv_budget_smoke_v2.py', 'scripts/audit_conv_budget_smoke_v2.py',
         'scripts/test_conv_budget_smoke_v2.py', 'docs/conv_budget_smoke_v2.md')
ARMS = ('adaptive', 'monolithic')


def identity():
    return {'budget': execution_identity(), 'outer_sources': {p:_sha256(ROOT/p) for p in FILES},
            'old_wrappers': wrapper_hashes()}


def schedule(value):
    jobs = [j for j in value['smoke_jobs'] if j['method'] in ARMS]
    if len(jobs)!=4 or [(j['dataset_index'],j['method']) for j in jobs] != [
            (0,'adaptive'),(0,'monolithic'),(1,'monolithic'),(1,'adaptive')]:
        raise ValueError('old-input rotated two-arm roster drift')
    return jobs


def request(value, job, head, execution):
    return {**request_for(value,job,head), 'execution_budget_contract':execution['budget']}


def artifacts(directory):
    # Bind partial and late packages too, without accepting them as conclusions.
    return {str(p.relative_to(directory)):_sha256(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name not in ('terminal.json','worker.log')}


def publish_terminal(root, job, started, code, expired, wait, error=None):
    directory=root/job['job_id']
    row={**job,'budget_seconds':300,'wall_seconds':time.monotonic()-started,
         'outer_timeout':expired,'return_code':code,'status':'TIMEOUT' if expired else 'ERROR',
         'package':None,'snapshot_sha256':None,'resource_wait':wait,
         'request_sha256':_sha256(directory/'request.json'),'evidence_level':'HZ_POLICY_ACCEPTED'}
    if error is not None: row['error']=error
    if (directory/'common_facts.json').exists():
        row['snapshot_sha256']=_sha256(directory/'common_facts.json')
    if not expired and code==0 and error is None:
        try:
            p=directory/'package'; status=read(p/'manifest.json')['status']
            if status not in ('SAFE','UNSAFE','UNKNOWN','TIMEOUT'):raise ValueError('bad status')
            row.update(status=status,package=str(p),manifest_sha256=_sha256(p/'manifest.json'))
        except Exception as exc:row['error']=repr(exc)
    row['artifacts']=artifacts(directory)
    atomic_json(directory/'terminal.json',row)
    row['wall_seconds']=time.monotonic()-started
    if row['wall_seconds']>300:
        row.update(status='TIMEOUT',outer_timeout=True,package=None)
    atomic_json(directory/'terminal.json',row)
    with (root/'rows.jsonl').open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return row


def run(root):
    if root.resolve()!=DEFAULT:raise ValueError('one frozen root, no retry')
    if (Path(sys.executable).resolve()!=Path(ACT).resolve()
            or _git_value('branch','--show-current')!='feat/moe-route-verification'
            or _git_value('status','--porcelain')
            or _git_value('rev-parse','HEAD')!=_git_value('rev-parse','@{upstream}')):
        raise ValueError('clean pushed feature checkout and ACT env required')
    value=selection(); jobs=schedule(value); execution=identity(); head=_git_value('rev-parse','HEAD')
    parents=[ROOT/'data/moe/results/conv_three_arm_smoke_20260915_r1',
             ROOT/'data/moe/results/conv_f0_timing_20260915_r1']
    parent_files={str(p.relative_to(ROOT)):_sha256(p) for d in parents for p in sorted(d.rglob('*')) if p.is_file()}
    root.mkdir(exist_ok=False)
    atomic_json(root/'runtime.json',{'schema':'conv_budget_smoke_v2','smoke':True,'full_started':False,
        'git_head':head,'execution':execution,'selection':value,'jobs':jobs,'parents':parent_files,
        'config':{'methods':value['identities']['method_configs']},'started_unix':time.time()})
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0'}
    rows=[]; error=None
    try:
        for job in jobs:
            if identity()!=execution or selection()!=value or _git_value('status','--porcelain'):
                raise ValueError('source or input drift')
            wait=wait_resources(root,job)
            directory=root/job['job_id'];directory.mkdir()
            started=time.monotonic()
            atomic_json(directory/'request.json',request(value,job,head,execution))
            try:
                code,expired=execute([ACT,'-m','scripts.conv_budget_worker_v2','--root',str(directory),
                    '--started',repr(started)],directory/'worker.log',started,env,
                    heartbeat=lambda pid,elapsed:atomic_json(root/'supervisor.json',
                        {'state':'RUNNING','job_id':job['job_id'],'pid':pid,'elapsed_seconds':elapsed,
                         'full_started':False,'updated_unix':time.time()}))
                row=publish_terminal(root,job,started,code,expired,wait)
            except BaseException as exc:
                rows.append(publish_terminal(root,job,started,None,False,wait,repr(exc)))
                raise
            rows.append(row)
            print(f"{len(rows)}/4 {job['job_id']} {row['status']} {row['wall_seconds']:.3f}s",flush=True)
            if row['status']=='ERROR':raise RuntimeError('fail-stop ERROR, no replacement')
    except BaseException as exc:error=repr(exc)
    finally:
        atomic_json(root/'run_terminal.json',{'state':'EXECUTION_ERROR' if error else 'EXECUTION_COMPLETED',
            'error':error,'completed_job_ids':[r['job_id'] for r in rows],
            'unattempted':jobs[len(rows):],'full_started':False})
    audit=subprocess.run([ACT,'-m','scripts.audit_conv_budget_smoke_v2','--root',str(root),
                          '--output',str(root/'audit.final.json')],cwd=ROOT,env=env)
    result=read(root/'audit.final.json') if (root/'audit.final.json').exists() else {}
    atomic_json(root/'supervisor.json',{'state':'SMOKE_PASSED_REVIEW_REQUIRED' if result.get('smoke_gate')=='PASS'
        else 'STOPPED_REVIEW_REQUIRED','audit_returncode':audit.returncode,'full_started':False})
    return 0 if not error and audit.returncode==0 and result.get('smoke_gate')=='PASS' else 1


if __name__=='__main__':
    def stop(signum,frame):raise KeyboardInterrupt(f'supervisor signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args()
    with (ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sys.exit(run(args.root))
