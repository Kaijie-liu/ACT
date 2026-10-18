"""Prepared seven-position runner. Empty batch authority forbids execution.

No resume, source preview, registration, quota expansion, model or GPU path.
The externally approved immutable freeze must be supplied with its expected hash.
"""
import time
ENTRY_STARTED=time.monotonic()  # Script imports onward; launch/exit observer is still unbound.
import argparse
import os
from pathlib import Path
import resource
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__': sys.path.append(str(ROOT))

from scoped_source.hz_request_store import POLICY, compact, identity, require
from scoped_source.native_protocol import read_json, write_json, REPORT_CAP
from scoped_source.native_supervised import execute as native_execute
from scoped_source.endpoint_component_supervised import execute as component_execute
from scoped_source.endpoint_batch import (validate, ORDER, FREEZE_CAP, RECEIPT_CAP,
    LIMITS, FINAL_RESERVE)
from scripts.audit_endpoint_midpoint_controls import inspect, storage_caps
from scoped_source.direct_supervised import census


def admission(deadline):
    """One observation, no queue/retry; never signal or modify another user's jobs."""
    begin=time.monotonic()
    require(begin<deadline,'single admission deadline')
    raw=Path('/proc/meminfo').read_bytes()
    require(len(raw)<=65536,'bounded meminfo')
    available=next(int(s.split()[1])*1024 for s in raw.decode().splitlines() if s.startswith('MemAvailable:'))
    jobs=[]
    for p in Path('/proc').iterdir():
        require(time.monotonic()<deadline,'bounded job observation')
        if not p.name.isdecimal(): continue
        try:
            if p.stat().st_uid!=os.getuid(): continue
            command=(p/'comm').read_bytes()[:128].decode(errors='replace').strip()
            jobs.append(dict(pid=int(p.name),command=command))
            require(len(jobs)<=512,'bounded own-job observation')
        except (FileNotFoundError,ProcessLookupError): continue
    require(time.monotonic()<deadline,'admission completed timely')
    return available,jobs,begin


def selected_observation(result):
    keys=('invocation','owned_execution','started','online_acceptance','status','result',
          'cumulative_work','work_complete','error','overrun_seconds','attempted_terminal_ref',
          'caller_seconds','control_only')
    return {k:result[k] for k in keys if k in result}


def position(root, spec, freeze, expected):
    started=time.monotonic(); deadline=started+300; result=None; error=None; audited=None; storage=None
    run=root/'runs'/spec['id']; ref=None
    try:
        result=(component_execute(run,freeze,expected,started) if spec['id']=='components' else
                native_execute(run,spec['profile'],enabled=True,outer_started=started))
        result=selected_observation(result)
        audited=inspect(run,spec,freeze,result,started,deadline=deadline)
        # Includes partial files/factory templates/copies/logs. For failures,
        # missing Work stays unknown even when the retained file census completes.
        storage=census(run,deadline); storage_caps(storage,spec['id']=='components',reserve_receipt=True)
        require(time.monotonic()<deadline,'whole position reception')
    except BaseException as exc:
        error=dict(kind=type(exc).__name__,message=str(exc)[:400])
        # A quiescent failure prefix may still be inventoried in its ORIGINAL
        # deadline. This is cost observation, never a substitute for inspect().
        if (result is not None and result.get('owned_execution',{}).get('cleanup_status')
                =='LEADER_REAPED_NO_LIVE_GROUP' and time.monotonic()<deadline):
            try:
                storage=census(run,deadline); storage_caps(storage,spec['id']=='components',reserve_receipt=True)
            except Exception:
                storage=None
    before=time.monotonic()
    receipt=dict(schema='ENDPOINT_POSITION_RECEIPT_V1',id=spec['id'],freeze=expected,
        profile=spec['sha256'],started=started,before_publish=before,
        status='PASS' if error is None else 'FAILED',error=error,observed=result,audit=audited,
        storage=storage,work_complete=audited is not None and audited['work_complete'],
        storage_complete=storage is not None,missing_cost_is_not_zero=True)
    try:
        require(run.is_dir(),'no slot receipt before request setup')
        ref=write_json(run/'slot.json',receipt,deadline=deadline,cap=RECEIPT_CAP)
        _,actual=read_json(run/'slot.json',expected=ref['sha256'],deadline=deadline,cap=RECEIPT_CAP)
        require(actual==ref['sha256'],'durable position receipt')
    except BaseException as exc:
        error=dict(kind=type(exc).__name__,message=str(exc)[:400],phase='position_receipt')
    end=time.monotonic()
    return dict(id=spec['id'],ok=error is None and end<deadline,started=started,observed_end=end,
        seconds=end-started,overrun_seconds=max(0.,end-deadline),error=error,
        receipt_sha256=None if ref is None else ref['sha256'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('freeze'); p.add_argument('directory')
    p.add_argument('--sha256',required=True); p.add_argument('--enabled',action='store_true'); args=p.parse_args()
    require(args.enabled,'explicit default-off finite batch')
    freeze,_=read_json(args.freeze,expected=args.sha256,cap=FREEZE_CAP,deadline=ENTRY_STARTED+60)
    validate(freeze,args.sha256,deadline=ENTRY_STARTED+60)  # Denies before source/limits/launch/write.
    root=Path(args.directory)
    require(root.is_absolute() and root.resolve()==root and root.is_relative_to('/data1/Kane/MOE')
            and root!=Path('/data1/Kane/MOE') and not root.exists() and root.parent.is_dir(),
            'new canonical batch; no resume or retries')
    available,jobs,admission_started=admission(ENTRY_STARTED+60)
    disk=shutil.disk_usage(root.parent).free
    require(available>=freeze['admission']['available_bytes'] and disk>=freeze['admission']['disk_free_bytes']
            and time.monotonic()<ENTRY_STARTED+60,'one fresh finite CPU admission')
    # Same process hard limits as actors. This bounds the outer retained ledgers
    # too; sampled aggregate RSS still is not a hard physical-memory upper bound.
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['as_bytes'],LIMITS['as_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['file_bytes'],LIMITS['file_bytes']))
    root.mkdir(); (root/'runs').mkdir()
    write_json(root/'freeze.json',freeze,deadline=ENTRY_STARTED+2100,cap=FREEZE_CAP)
    ar=write_json(root/'admission.json',dict(freeze=args.sha256,attempts=1,available_bytes=available,
        disk_free_bytes=disk,jobs=jobs,started=admission_started,seconds=time.monotonic()-admission_started,
        device='cpu',threads=1,wait_excluded_from_comparisons=True),deadline=ENTRY_STARTED+2100)
    rows=[]; stop=None
    for spec in freeze['records']:
        if time.monotonic()+300+FINAL_RESERVE>=ENTRY_STARTED+2100:
            stop='INSUFFICIENT_FULL_POSITION_AND_FINAL_RESERVE'; break
        row=position(root,spec,freeze,args.sha256); rows.append(row)
        print(compact(dict(id=row['id'],ok=row['ok'],seconds=row['seconds'])).decode(),flush=True)
        if not row['ok']: stop=spec['id']; break
    # Only bounded batch bookkeeping follows. The offline auditor cannot change
    # the online flags or turn a partial/late position into successful completion.
    b=dict(schema='ENDPOINT_FINITE_BATCH_V1',freeze=args.sha256,started=ENTRY_STARTED,
        admission_sha256=ar['sha256'],positions=rows,stop=stop,
        unexecuted=list(ORDER[len(rows):]),before_publish=time.monotonic(),
        models=0,gpu=0,retries=0,quota_increases=0)
    br=write_json(root/'batch.json',b,deadline=ENTRY_STARTED+2100,cap=REPORT_CAP)
    read_json(root/'batch.json',expected=br['sha256'],deadline=ENTRY_STARTED+2100)
    at=time.monotonic()
    final=dict(schema='ENDPOINT_BATCH_RECEPTION_V1',batch_sha256=br['sha256'],freeze=args.sha256,
               observed_at=at,timely=at<ENTRY_STARTED+2100)
    write_json(root/'batch-received.json',final,deadline=ENTRY_STARTED+2100,cap=RECEIPT_CAP)
    # Publication must finish timely as well. A lost/late final receipt is not
    # retrospectively repaired by the offline audit or a second run.
    require(time.monotonic()<ENTRY_STARTED+2100,'batch publication overrun')
    print(compact(dict(status='PENDING_CALLER_OBSERVATION' if stop is None else 'STOPPED',
        seconds=time.monotonic()-ENTRY_STARTED,positions=len(rows),stop=stop,
        online_batch_acceptance=False)).decode(),flush=True)
    return int(stop is not None)


if __name__=='__main__': raise SystemExit(main())
