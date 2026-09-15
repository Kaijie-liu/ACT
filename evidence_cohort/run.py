"""Single-writer 60-request supervisor. No resume, tuning or sample replacement."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from portable_proof.runtime import digest
from scripts.optional_evidence_dev_contract import ROOT,ACT,read,save,git
from moe_evidence.execution import LEVELS
from evidence_cohort.contract import OUTPUT,LAUNCH,FREEZE,EXECUTION,verify_freeze,selection,request_for
from evidence_cohort.ownership import info,stop_owned


def environment():
    return {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0'}


def resource():
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    return {'ram_gib':available/2**30,'disk_gib':shutil.disk_usage(ROOT).free/2**30,
            'load_per_core':os.getloadavg()[0]/os.cpu_count()}


def resource_ok(r):
    p=EXECUTION['resource']
    return r['ram_gib']>=p['minimum_ram_gib'] and r['disk_gib']>=p['minimum_disk_gib'] and 0<=r['load_per_core']<=p['maximum_load_per_core']


def wait_resource(record,clock=time.monotonic,sleep=time.sleep,probe=resource):
    start=clock()
    while True:
        r=probe();elapsed=clock()-start;record({'state':'RESOURCE_WAIT','seconds':elapsed,'resource':r})
        if resource_ok(r):return {'seconds':elapsed,'at_launch':r}
        if elapsed>=EXECUTION['resource']['wait_limit_seconds']:raise TimeoutError('resource wait cap reached')
        sleep(min(EXECUTION['resource']['poll_seconds'],EXECUTION['resource']['wait_limit_seconds']-elapsed))


def wait_owned(process,deadline,clock=time.monotonic):
    owned=info(process.pid)
    try:
        code=process.wait(timeout=max(.001,deadline-clock()))
        return {'return_code':code,'killed':False,'owned':owned,'killed_processes':[]}
    except subprocess.TimeoutExpired:
        killed=stop_owned(owned) if owned else []
        process.wait()
        return {'return_code':process.returncode,'killed':True,'owned':owned,'killed_processes':killed}
    except BaseException:
        if owned:stop_owned(owned)
        process.wait();raise


def request(job,req,root,env):
    directory=root/job['job_id'];control=root/'control'/job['job_id']
    started=time.monotonic();directory.mkdir(exist_ok=False);control.mkdir(exist_ok=False)
    save(directory/'request.json',req)
    save(directory/'execution.json',{'arm':job['arm'],'started_monotonic':started,'total_seconds':300})
    cmd=[ACT,'-m','evidence_cohort.driver',str(directory),str(control),job['arm'],'--started',repr(started)]
    with (control/'driver.log').open('x') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(control/'ownership.json',{'pid':p.pid,'identity':info(p.pid),'started_monotonic':started})
        owner=wait_owned(p,started+298)
    owner_elapsed=time.monotonic()-started
    whole_timeout=owner['killed'] or time.monotonic()-started>300
    if not whole_timeout and owner['return_code']==0 and (control/'candidate.json').exists():
        terminal=read(control/'candidate.json')
    else:
        stages=read(directory/'stage_progress.json') if (directory/'stage_progress.json').exists() else {}
        terminal={'arm':job['arm'],'dataset_index':job['dataset_index'],'stages':stages,
            'request_sha256':digest((directory/'request.json').read_bytes()),
            'artifact_sha256':{str(p.relative_to(directory)):digest(p.read_bytes()) for p in directory.rglob('*') if p.is_file()},
            'status':'TIMEOUT' if whole_timeout else 'ERROR','error':None if whole_timeout else 'driver failed or omitted candidate terminal',
            'evidence_level':LEVELS[job['arm']],'budget_seconds':300,'complete_independent_check':False,
            'production_gate_changed':False,'deployed_float_SAFE':False,'outer_timeout':whole_timeout}
    terminal.update(whole_request_timeout=whole_timeout,outer_process=owner,wall_seconds=time.monotonic()-started)
    if whole_timeout or terminal['wall_seconds']>300:
        terminal.update(status='TIMEOUT',complete_independent_check=False,whole_request_timeout=True,outer_timeout=True)
    active=read(control/'active.json') if (control/'active.json').exists() else None
    terminal['censored_phase']=dict(active,observed_phase_window_seconds=max(0.,owner_elapsed-active['entered_seconds']),
        scope='last active phase window through driver return/cleanup; not exact operation time') if whole_timeout and active else None
    save(directory/'terminal.json',terminal)
    terminal['wall_seconds']=time.monotonic()-started
    if terminal['wall_seconds']>300:terminal.update(status='TIMEOUT',whole_request_timeout=True,outer_timeout=True,complete_independent_check=False)
    save(directory/'terminal.json',terminal)
    return {**job,**terminal,'terminal_sha256':digest((directory/'terminal.json').read_bytes())}


def append_row(path,row):
    with path.open('a') as f:
        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def loop(jobs,run_one,publish):
    """Injectable roster loop. All unresolved outcomes continue, ERROR stops."""
    rows=[];error=None;aborted=None
    for job in jobs:
        try:
            row=run_one(job);rows.append(row);publish(row)
            if row['status']=='ERROR':error='request ERROR: '+job['job_id'];break
        except BaseException as exc:
            error=repr(exc)
            if not rows or rows[-1]['job_id']!=job['job_id']:aborted=job
            break
    return {'state':'EXECUTION_COMPLETED' if not error and len(rows)==len(jobs) else 'EXECUTION_ERROR',
        'completed_job_ids':[r['job_id'] for r in rows], 'aborted_job':aborted,
        'unattempted':jobs[len(rows)+(aborted is not None):], 'error':error}


def run():
    if Path(sys.executable).resolve()!=Path(ACT).resolve():raise ValueError('ACT environment required')
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):
        raise ValueError('clean feature checkout required')
    freeze=verify_freeze();v=selection();head=git('rev-parse','HEAD')
    remote=git('ls-remote','origin','refs/heads/feat/moe-route-verification').split()[0]
    if remote!=head:raise ValueError('execution must be published before launch')
    # Revalidate checkpoint, clean inputs, external sources; no verification query.
    from scripts.freeze_general_evidence import audit as clean_audit
    clean=clean_audit()
    if clean['status']!='PASS':raise ValueError('clean preflight failed')
    OUTPUT.mkdir(exist_ok=False);(OUTPUT/'control').mkdir();(OUTPUT/'reviews').mkdir()
    runtime={'schema':EXECUTION['schema'],'git_head':head,'remote_before_launch':remote,
        'freeze_sha256':digest(FREEZE.read_bytes()),'execution':EXECUTION,'selection_sha256':freeze['execution']['selection_sha256'],
        'jobs':v['jobs'],'pid':os.getpid(),'started_monotonic':time.monotonic(),'state':'RUNNING',
        'completed_job_ids':[],'unattempted':v['jobs'],'active_job':None,'clean_preflight':clean}
    save(OUTPUT/'runtime.json',runtime);env=environment()
    def run_one(job):
        runtime['active_job']=job;save(OUTPUT/'runtime.json',runtime)
        wait=wait_resource(lambda x:save(OUTPUT/'progress.json',{**x,'job':job}))
        verify_freeze()
        req=request_for(v,job,head);row=request(job,req,OUTPUT,env);row['resource_wait']=wait
        return row
    def publish(row):
        append_row(OUTPUT/'rows.jsonl',row)
        runtime['completed_job_ids'].append(row['job_id']);runtime['unattempted']=v['jobs'][len(runtime['completed_job_ids']):]
        runtime['active_job']=None;save(OUTPUT/'runtime.json',runtime)
        print(f"{len(runtime['completed_job_ids'])}/60 {row['job_id']} input{row['dataset_index']} {row['status']} {row['wall_seconds']:.3f}s",flush=True)
    end=loop(v['jobs'],run_one,publish);save(OUTPUT/'run_terminal.json',end)
    runtime.update(state=end['state'],active_job=None,completed_job_ids=end['completed_job_ids'],
                   unattempted=end['unattempted'],aborted_job=end['aborted_job']);save(OUTPUT/'runtime.json',runtime)
    # A fresh process owns final review; separate cost, never request rescue.
    start=time.monotonic()
    with (OUTPUT/'audit.log').open('x') as log:
        p=subprocess.Popen([ACT,'-m','evidence_cohort.audit','--root',str(OUTPUT)],cwd=ROOT,env=env,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        audit=wait_owned(p,start+EXECUTION['final_audit_timeout_seconds'])
    save(OUTPUT/'audit_execution.json',{**audit,'seconds':time.monotonic()-start})
    runtime['audit_state']='PASS' if audit['return_code']==0 and (OUTPUT/'audit.final.json').exists() else 'FAILED_OR_INCOMPLETE'
    save(OUTPUT/'runtime.json',runtime)
    print('FINAL',end['state'],'AUDIT',runtime['audit_state'],flush=True)


def launch():
    verify_freeze()
    if OUTPUT.exists() or LAUNCH.exists():raise FileExistsError('one-shot launch, no resume')
    LAUNCH.mkdir(exist_ok=False)
    with (LAUNCH/'supervisor.log').open('x') as log:
        p=subprocess.Popen([ACT,'-m','evidence_cohort.run','--run'],cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(LAUNCH/'launch.json',{'pid':p.pid,'identity':info(p.pid),'head':git('rev-parse','HEAD'),
        'freeze_sha256':digest(FREEZE.read_bytes()),'output':str(OUTPUT),'state':'LAUNCHED_NOT_COMPLETED'})
    print(json.dumps(read(LAUNCH/'launch.json'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--run',action='store_true');m.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:launch()
    else:
        os.nice(10)
        # Locks held across execution AND final audit, including resource waits.
        with (ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run()
