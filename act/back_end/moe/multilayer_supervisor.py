"""Spawn-based hard request deadline, partial evidence and charged final audit.

Preconstructed CPU model/tensors are the API input. Child startup, transfer,
verification, serialization and audit are charged. Parent cleanup and terminal
receipt publication are separately reported, never free solver extensions.
"""
from __future__ import annotations
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import uuid


def _atomic(path, value):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    data=json.dumps(value,sort_keys=True,allow_nan=False).encode()
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def _child(worker, args, deadline, directory, execution_id):
    directory=Path(directory)
    try:
        result=worker(args,deadline,directory)
        package={'execution_id':execution_id,'result':result,'finished_monotonic':time.monotonic()}
        _atomic(directory/'candidate.json',package)
    except BaseException as exc:
        _atomic(directory/'error.json',{'execution_id':execution_id,'type':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'seconds_until_deadline':deadline-time.monotonic()})


def supervise_call(worker, args, *, output_dir, total_seconds, accept):
    """One fresh directory, one child, no retries or silent budget extensions.

    Worker and args must be spawn-pickleable. ``accept`` runs in the parent and
    must be a bounded structural audit, NOT a new unbounded solver invocation.
    """
    started=time.monotonic()
    if not math.isfinite(total_seconds) or total_seconds <= 0:
        raise ValueError('finite positive budget required')
    deadline=started+total_seconds
    directory=Path(output_dir).resolve()
    if not directory.is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('output must be under /data1/Kane/MOE')
    directory.mkdir(parents=True,exist_ok=False)
    execution_id=uuid.uuid4().hex
    _atomic(directory/'request.json',{'execution_id':execution_id,'budget_seconds':total_seconds,
        'start_monotonic':started,'deadline_monotonic':deadline,'schema':'multilayer-supervisor-v1'})
    process=mp.get_context('spawn').Process(target=_child,args=(worker,args,deadline,str(directory),execution_id))
    status,reason='ERROR','child_start_failed'
    audit=None
    result=None
    audit_seconds=0.
    try:
        process.start()
        startup=time.monotonic()-started
        process.join(timeout=max(0.,deadline-time.monotonic()))
        if process.is_alive() or time.monotonic()>=deadline:
            status,reason='TIMEOUT','outer_deadline'
        elif process.exitcode != 0 or (directory/'error.json').exists():
            status,reason='ERROR','child_exception_or_exit'
        elif not (directory/'candidate.json').exists():
            status,reason='ERROR','missing_candidate'
        else:
            package=json.loads((directory/'candidate.json').read_text())
            if package['execution_id']!=execution_id or package['finished_monotonic']>=deadline:
                status,reason='ERROR','candidate_binding_or_lateness'
            else:
                result=package['result']
                began=time.monotonic()
                audit=accept(result)
                audit_seconds=time.monotonic()-began
                if audit['status']!='PASS':
                    status,reason='ERROR','terminal_audit_failed'
                else:
                    status,reason=result['status'],result['reason']
                if time.monotonic()>=deadline:
                    status,reason='TIMEOUT','audit_deadline'
    except Exception as exc:
        status,reason='ERROR','supervisor_exception:'+type(exc).__name__+':'+str(exc)
        startup=time.monotonic()-started
    finally:
        decision_seconds=time.monotonic()-started
        cleanup_start=time.monotonic()
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.)
        cleanup_seconds=time.monotonic()-cleanup_start
    partial=None
    if (directory/'progress.json').exists():
        partial=json.loads((directory/'progress.json').read_text())
    terminal={'schema':'multilayer-supervisor-v1','execution_id':execution_id,'status':status,'reason':reason,
        'budget_seconds':total_seconds,'decision_seconds':decision_seconds,'parent_start_seconds':startup,
        'cleanup_seconds':cleanup_seconds,'audit_seconds':audit_seconds,'audit':audit,
        'worker_exitcode':process.exitcode,'partial_histories':len((partial or {}).get('records',[])),
        'evidence_grade': result['evidence_grade'] if result and status==result['status'] else 'NONE',
        'result_request_id':(result or {}).get('request_id'),
        'artifacts':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ['request.json','progress.json','candidate.json','error.json'] if (directory/name).exists()},
        'elapsed_before_receipt_seconds':time.monotonic()-started}
    _atomic(directory/'terminal.json',terminal)
    # Receipt completion is observable to the caller; it never extends the
    # acceptance deadline or rewrites a timed-out candidate into a positive.
    return {**terminal,'return_wall_seconds':time.monotonic()-started}


def _verification_worker(args, deadline, directory):
    import torch
    from act.util.device_manager import initialize_device
    from act.back_end.moe.multilayer import verify_multilayer_box
    from act.back_end.moe.multilayer_audit import audit_multilayer_result
    from act.back_end.moe.class_separated_top1 import validate_replay
    initialize_device('cpu','float64')
    torch.set_num_threads(1)
    model,kwargs=args
    remaining=deadline-time.monotonic()
    if remaining<=0:
        raise TimeoutError('startup_consumed_budget')
    result=verify_multilayer_box(model, **kwargs, total_seconds=remaining,
        progress=lambda r:_atomic(directory/'progress.json',r))
    began=time.monotonic()
    replayed = result['status']=='UNSAFE_REPLAYED' and validate_replay(model,
        torch.tensor(result['witness'],dtype=torch.float64),kwargs['lower'],kwargs['upper'],kwargs['rows'],kwargs['thresholds'])
    result['independent_replay']=bool(replayed)
    result['child_audit']=audit_multilayer_result(result,expected_request_id=result.get('request_id'),replay=lambda _:replayed)
    result['child_audit_seconds']=time.monotonic()-began
    return result


def verify_multilayer_box_supervised(model, *, output_dir, total_seconds=300., **kwargs):
    """Hard-deadline counterpart; call behind a Python __main__ guard."""
    from act.back_end.moe.multilayer_audit import audit_multilayer_result
    def accept(result):
        # The child binds actual graph/source/state. Parent additionally binds
        # the caller's source model and all requested tensors before auditing.
        from act.back_end.moe.multilayer import _hash_tensor
        identity=result.get('identity',{})
        if identity.get('state')!={k:_hash_tensor(v) for k,v in model.state_dict().items()}:
            return {'status':'FAIL','issues':['parent model binding']}
        if identity.get('tensors')!={k:_hash_tensor(kwargs[k]) for k in ['center','lower','upper','rows','thresholds']}:
            return {'status':'FAIL','issues':['parent tensor binding']}
        if identity.get('sources')!=dict(kwargs.get('source_hashes') or {}):
            return {'status':'FAIL','issues':['parent source binding']}
        if result.get('child_audit',{}).get('status')!='PASS':
            return {'status':'FAIL','issues':['child audit']}
        return audit_multilayer_result(result,expected_request_id=result.get('request_id'),
            replay=lambda _:result.get('independent_replay') is True)
    return supervise_call(_verification_worker,(model,kwargs),output_dir=output_dir,
                          total_seconds=total_seconds,accept=accept)
