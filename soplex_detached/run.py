"""Private tmux launch, owned controller monitoring and no-resume reconciliation."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from evidence_cohort.ownership import info,stop_owned,descendants
from lp_sandwich.check import strict_json
from soplex_execution.runtime import ROOT,PYTHON,env
from soplex_execution.supervisor import run_jobs,verify_freeze as verify_parent
from soplex_execution.audit import audit_batch
from soplex_fidelity.io import save,sha

PROTOCOL=ROOT/'docs/soplex_detached_v2_protocol.json'
FREEZE=ROOT/'docs/soplex_detached_v2_freeze.json'
OUTPUT=ROOT/'data/moe/results/soplex_diagnostic_real_20260920_v2'
SOCKET=Path('/data1/Kane/MOE/run/soplex-v2.sock')


def boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def atomic(path,value):
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def same_process(record):
    current=info(record['pid'])
    return bool(current and current['start']==record['start'] and current['state']!='Z')


def same_science(f,parent):
    if {k:v for k,v in f.items() if k!='output'}!={k:v for k,v in parent.items() if k!='output'}:
        raise ValueError('only result directory may differ')


def frozen():
    parent,_=verify_parent()
    f=strict_json(PROTOCOL.read_bytes());a=strict_json(FREEZE.read_bytes())
    if a['protocol_sha256']!=sha(PROTOCOL):raise ValueError('new protocol identity')
    same_science(f,parent)
    if f['output']!=str(OUTPUT):raise ValueError('reserved output')
    for path,digest in a['bindings'].items():
        if sha(path)!=digest:raise ValueError('execution dependency drift: '+path)
    return f,a


def launch_tmux(socket,root,command):
    # A private server avoids any interaction with other users' sessions.
    socket.parent.mkdir(parents=True,exist_ok=True)
    if socket.exists():raise FileExistsError('private tmux socket already used')
    cmd=shlex.join(list(map(str,command)))+' >'+shlex.quote(str(root/'guardian.stdout'))+' 2>'+shlex.quote(str(root/'guardian.stderr'))
    subprocess.run(['/usr/bin/tmux','-S',str(socket),'-f','/dev/null','new-session','-d','-s','soplex','-c',str(ROOT),cmd],check=True,timeout=10,env=env())


def postmortem(root,jobs,reason):
    """No proof promotion and no reconstruction of missing terminal costs."""
    rows=[];waits={};malformed_tail=False
    p=root/'resource_wait.jsonl'
    if p.exists():
        lines=p.read_bytes().splitlines()
        for index,line in enumerate(lines):
            try:v=strict_json(line)
            except Exception:
                if index!=len(lines)-1:raise ValueError('non-tail corrupted wait log')
                malformed_tail=True;continue
            waits[v['job_id']]=v
    for job in jobs:
        jr=root/job['job_id'];summary=root/(job['job_id']+'.summary.json')
        entered=(jr/'spec.json').exists()
        state='COMPLETED_RECORD_REQUIRES_AUDIT' if summary.exists() else 'INTERRUPTED_DURING_JOB' if entered else 'NOT_RUN_SUPERVISOR_INTERRUPTION'
        rows.append(dict(job_id=job['job_id'],status=state,upper_bound=None,
            completed_summary=None if not summary.exists() else dict(path=str(summary),sha256=sha(summary)),
            recorded_solver_attempts=int((jr/'solver.command.json').exists()),
            complete_request_seconds=None,cost_status='UNKNOWN_OR_RIGHT_CENSORED',
            last_resource_observation=waits.get(job['job_id']),
            retained_phase_files=[p.name for p in sorted(jr.glob('*.json'))] if jr.exists() else []))
    result=dict(schema='SOPLEX_DETACHED_INTERRUPTION_V2',status='INTERRUPTED_NO_RETRY',reason=reason,
        rows=rows,denominator=len(jobs),observed_unix=time.time(),observed_boot_id=boot(),
        malformed_last_wait_line=malformed_tail,network_SAFE=False,network_UNSAFE=False,
        no_automatic_resume=True,scope='Post-mortem accounting only, not new LP feasibility or terminal proof acceptance.')
    save(root/'interruption.json',result);return result


def guard(root,jobs,command):
    started=time.monotonic();b=boot()
    # Adopt descendants if the controller is killed before it can clean its workers.
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER')
    save(root/'guardian_identity.json',dict(process=info(os.getpid()),boot_id=b,started_monotonic=started,started_unix=time.time()))
    child=None;record=None;error=None;cleanup=[]
    def terminated(signum,frame):raise InterruptedError('guardian signal '+str(signum))
    signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGINT,terminated)
    try:
        with (root/'controller.stdout').open('xb') as out,(root/'controller.stderr').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=env(),stdout=out,stderr=err,start_new_session=True)
            record=info(child.pid)
            if record is None:raise RuntimeError('controller identity missing')
            save(root/'controller_identity.json',dict(process=record,boot_id=b,command=command))
            while child.poll() is None:
                atomic(root/'heartbeat.json',dict(boot_id=b,guardian=info(os.getpid()),controller=record,
                    owned_descendants=descendants(os.getpid()),
                    unix=time.time(),elapsed_seconds=time.monotonic()-started,state='CONTROLLER_RUNNING'))
                time.sleep(1)
            child.wait()
        if child.returncode!=0:error='controller exit '+str(child.returncode)
        elif not (root/'batch.json').exists():error='controller exited without batch terminal'
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        if record:cleanup=stop_owned(record)
        if child:child.wait()
    # Also clean recorded owned children after an abnormal exit if controller still exists.
    if error:
        if record and same_process(record):cleanup+=stop_owned(record)
        for item in descendants(os.getpid()):
            if item['pid']!=os.getpid():cleanup+=stop_owned(item)
        postmortem(root,jobs,error)
    result=dict(status='CONTROLLER_COMPLETE' if error is None else 'INTERRUPTED',error=error,
        controller_returncode=None if child is None else child.returncode,owned_cleanup=cleanup,
        seconds=time.monotonic()-started,scope='Controller lifecycle including resource waiting, not per-LP solver cost')
    save(root/'guardian_terminal.json',result);return result


def controller():
    f,_=frozen();root=Path(f['output'])
    runtime={k:f['runtime'][k] for k in ('soplex','reader','settings')};runtime['checker']=f['independent_checker']
    run_jobs(f['jobs'],root,runtime,f['policy'],frozen)


def run_guardian():
    f,_=frozen();root=Path(f['output'])
    result=guard(root,f['jobs'],[str(PYTHON),'-m','soplex_detached.run','controller'])
    if result['status']=='CONTROLLER_COMPLETE':
        start=time.monotonic()
        try:
            audit_batch(PROTOCOL,FREEZE,root/'final_review.json')
            save(root/'completion.json',dict(status='AUDITED',review_sha256=sha(root/'final_review.json'),
                postterminal_audit_seconds=time.monotonic()-start))
        except Exception as exc:
            save(root/'completion.json',dict(status='AUDIT_ERROR',error=type(exc).__name__+': '+str(exc),
                postterminal_audit_seconds=time.monotonic()-start))


def reconcile(root,jobs):
    if (root/'interruption.json').exists():return 'RECORDED_INTERRUPTION'
    g=root/'guardian_identity.json';c=root/'controller_identity.json'
    for path in (g,c):
        if path.exists():
            v=strict_json(path.read_bytes())
            if v['boot_id']==boot() and same_process(v['process']):return 'LIVE_NO_MUTATION'
    if (root/'completion.json').exists():return 'COMPLETION_PRESENT'
    postmortem(root,jobs,'No live identity-bound guardian/controller; observed after shutdown or session loss. Cause undetermined.')
    return 'RECORDED_INTERRUPTION'


def launch():
    f,_=frozen()
    r=strict_json((ROOT/'docs/soplex_detached_v2_readiness.json').read_bytes())
    if r['status']!='PASS' or r['freeze_sha256']!=sha(FREEZE):raise ValueError('readiness review required')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='feat/moe-route-verification':raise ValueError('branch')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('clean execution tree required')
    root=Path(f['output']);root.mkdir()
    save(root/'execution_identity.json',dict(protocol_sha256=sha(PROTOCOL),addendum_sha256=sha(FREEZE),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),boot_id=boot()))
    try:launch_tmux(SOCKET,root,[PYTHON,'-m','soplex_detached.run','guardian'])
    except Exception as exc:
        postmortem(root,f['jobs'],'tmux launch failed: '+str(exc));raise
    print(json.dumps(dict(status='LAUNCHED_NOT_COMPLETED',root=str(root),socket=str(SOCKET))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','guardian','controller','reconcile'])
    a=p.parse_args()
    if a.action=='launch':launch()
    elif a.action=='guardian':run_guardian()
    elif a.action=='controller':controller()
    else:
        f,_=frozen();print(reconcile(Path(f['output']),f['jobs']))
