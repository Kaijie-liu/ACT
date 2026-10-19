"""Prepared seven-position worker. Only trusted observer may launch it.

No resume, source preview, registration, quota expansion, model or GPU path.
The externally approved immutable freeze must be supplied with its expected hash.
"""
import time
import os
from pathlib import Path
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


def selected_observation(result):
    keys=('invocation','owned_execution','started','online_acceptance','status','result',
          'cumulative_work','work_complete','error','overrun_seconds','attempted_terminal_ref',
          'caller_seconds','control_only')
    return {k:result[k] for k in keys if k in result}


def position(root, spec, freeze, expected, owner_session):
    started=time.monotonic(); deadline=started+300; result=None; error=None; audited=None; storage=None
    run=root/'runs'/spec['id']; ref=None
    try:
        result=(component_execute(run,freeze,expected,started,owner_session=owner_session)
                if spec['id']=='components' else native_execute(run,spec['profile'],enabled=True,
                    outer_started=started,owner_session=owner_session))
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


def worker(root, expected, launch_hash):
    from scoped_source.endpoint_batch_observer import launch_check
    root=Path(root); owner_session=os.getsid(0)
    require(owner_session==os.getpid(),'worker must be owned batch session leader')
    a,ah=read_json(root/'admission.json',expected=launch_hash)
    freeze,_=read_json(root/'freeze.json',expected=expected,cap=FREEZE_CAP,deadline=a['run_deadline'])
    launch_check(a,freeze,expected,root)
    validate(freeze,expected,deadline=a['run_deadline'])
    require({p.name for p in root.iterdir()}=={'runs','freeze.json','admission.json'}
            and (root/'runs').is_dir() and not (root/'runs').is_symlink()
            and not any((root/'runs').iterdir()),'fresh observer-owned directory only')
    rows=[]; stop=None
    for spec in freeze['records']:
        if time.monotonic()+300>=a['run_deadline']:
            stop='INSUFFICIENT_FULL_POSITION_AND_FINAL_RESERVE'; break
        row=position(root,spec,freeze,expected,owner_session); rows.append(row)
        print(compact(dict(id=row['id'],ok=row['ok'],seconds=row['seconds'])).decode(),flush=True)
        if not row['ok']: stop=spec['id']; break
    # Only bounded batch bookkeeping follows. The offline auditor cannot change
    # the online flags or turn a partial/late position into successful completion.
    b=dict(schema='ENDPOINT_FINITE_BATCH_V3',freeze=expected,started=a['started'],
        start_sha256=a['start_sha256'],
        admission_sha256=ah,positions=rows,stop=stop,owner_session=owner_session,worker_deadline=a['run_deadline'],
        unexecuted=list(ORDER[len(rows):]),before_publish=time.monotonic(),
        models=0,gpu=0,retries=0,quota_increases=0)
    br=write_json(root/'batch.json',b,deadline=a['run_deadline'],cap=REPORT_CAP)
    read_json(root/'batch.json',expected=br['sha256'],deadline=a['run_deadline'])
    print(compact(dict(status='PENDING_CALLER_OBSERVATION' if stop is None else 'STOPPED',
        seconds=time.monotonic()-a['started'],positions=len(rows),stop=stop,
        online_batch_acceptance=False)).decode(),flush=True)
    return int(stop is not None)


if __name__=='__main__': raise SystemExit('No standalone timer/CLI: use the approved trusted observer API')
