"""Single-admission finite batch. No resumes, retries, quota changes or real models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scoped_source.hz_request_store import compact, digest, identity, require, POLICY
from scoped_source.native_protocol import PYTHON, validate_profile, approval_registry
from scoped_source.native_supervised import execute, clean_env
from scoped_proof.owned_bounded import execute as owned

ROOT=Path(__file__).resolve().parents[1]


def save(path,obj):
    raw=compact(obj); require(len(raw)<=2**20,'bounded batch receipt')
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(); p.add_argument('freeze'); p.add_argument('directory'); a=p.parse_args()
    raw=Path(a.freeze).read_bytes(); freeze=json.loads(raw); root=Path(a.directory)
    require(root.is_absolute() and root.resolve()==root and root.is_relative_to('/data1/Kane/MOE')
            and not root.exists(), 'one new batch; no resume')
    require(freeze['schema']=='NATIVE_FINITE_CONTROLS_FREEZE_V1' and len(freeze['records'])==26, 'closed finite roster')
    require(digest((ROOT/'docs/hz_native_controls_protocol_20261018.md').read_bytes())==freeze['protocol_sha256'],
            'frozen control protocol')
    code=freeze['records'][0]['profile']['implementation'] | freeze['control_code']
    require(all(digest((ROOT/n).read_bytes())==h for n,h in code.items()),'frozen batch implementation')
    require(approval_registry()[0]==frozenset(r['sha256'] for r in freeze['records']), 'exact one-batch approval registry')
    for r in freeze['records']:
        require(identity(r['profile'])==r['sha256'],'frozen profile digest'); validate_profile(r['profile'])
    # This is one fresh admission, not a retry loop or GPU availability check.
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    import shutil
    disk=shutil.disk_usage(root.parent).free
    require(available>=4*2**30 and disk>=8*2**30, 'single bounded resource admission')
    jobs=subprocess.run(['ps','-u','kaijieliu','-o','pid,ppid,etime,comm'],capture_output=True,text=True,check=True)
    root.mkdir(); (root/'runs').mkdir(); (root/'receipts').mkdir()
    save(root/'admission.json',dict(freeze_sha256=digest(raw),available_bytes=available,disk_free_bytes=disk,
        admission_attempts=1,jobs=jobs.stdout,as_bytes=8*2**30,rss_bytes=2*2**30,device='cpu',threads=1))
    begin=time.monotonic(); outcomes=[]; stop=None; component=None
    normal=[r for r in freeze['records'] if r['kind']=='normal']; faults=[r for r in freeze['records'] if r['kind']=='fault']
    def request(r):
        nonlocal stop
        t=time.monotonic(); result=None; error=None
        if t-begin+300>freeze['cumulative_seconds']:
            stop='CUMULATIVE_BUDGET'; return
        try:
            result=execute(root/'runs'/r['id'],r['profile'],enabled=True)
            if r['kind']=='normal':
                require(result['online_acceptance'] and result['status']=='COMPLETE','normal complete independent aggregation')
            else:
                require(not result['online_acceptance'],'fault must not be an online positive/complete acceptance')
                from scripts.audit_native_controls import fault_reached
                fault_reached(root/'runs'/r['id'],r,result,time.monotonic()-t)
        except Exception as exc:
            error=dict(kind=type(exc).__name__,message=str(exc)[:500]); stop=r['id']
        item=dict(id=r['id'],kind=r['kind'],profile=r['sha256'],result=result,error=error,
            seconds_before_receipt=time.monotonic()-t,request_started=t)
        save(root/'receipts'/(r['id']+'.json'),item)
        elapsed=time.monotonic()-t
        ok=error is None and elapsed<300 and result is not None and result['overrun_seconds']==0
        if not ok: stop=r['id']
        outcomes.append(dict(id=r['id'],kind=r['kind'],ok=ok,full_caller_seconds=elapsed,
                             receipt_sha256=digest((root/'receipts'/(r['id']+'.json')).read_bytes())))
        print(compact(dict(id=r['id'],ok=ok,status=None if result is None else result['status'],seconds=elapsed)).decode(),flush=True)
    for r in normal:
        if stop: break
        request(r)
    if stop is None:
        start=time.monotonic()
        try:
            command=[PYTHON,'-I','-S','-B',str(ROOT/'scripts/native_control_components.py'),str(root/'components'),str(Path(a.freeze).resolve())]
            component=owned(command,root/'component.log',run_deadline=start+299,cleanup_deadline=start+299.5,
                env=clean_env(),rss_limit=2*2**30,bounded_errors=True)
            require(component['status']=='COMPLETED' and component['cleanup_status']=='LEADER_REAPED_NO_LIVE_GROUP',
                    'complete bounded component batch')
            report=json.loads((root/'components/report.json').read_bytes())
            require(report['status']=='PASS' and len(report['tests'])==32,'complete finite component differential')
            component['report']=report
        except Exception as exc:
            stop='component'; component=dict(component or {},error=dict(kind=type(exc).__name__,message=str(exc)[:500]))
        component['full_caller_seconds']=time.monotonic()-start
        if component['full_caller_seconds']>=300: stop='component_deadline'
        save(root/'component-receipt.json',component)
        print(compact(dict(component=component.get('status','ERROR'),stop=stop)).decode(),flush=True)
    for r in faults:
        if stop: break
        request(r)
    attempted={r['id'] for r in outcomes}
    result=dict(schema='NATIVE_FINITE_BATCH_V1',status='COMPLETE_PENDING_AUDIT' if stop is None else 'STOPPED',
        stop=stop,freeze_sha256=digest(raw),outcomes=outcomes,component=component,
        component_receipt_sha256=None if component is None else digest((root/'component-receipt.json').read_bytes()),
        unexecuted=[r['id'] for r in freeze['records'] if r['id'] not in attempted],
        seconds=time.monotonic()-begin,real_models=0,gpu=0,quota_increases=0,automatic_retries=0)
    save(root/'batch.json',result)
    print(compact(dict(status=result['status'],requests=len(outcomes),stop=stop,seconds=result['seconds'])).decode(),flush=True)
    return 0 if stop is None else 1


if __name__=='__main__': raise SystemExit(main())
