"""Read-only audit of the retained V4 pre-admission refusal. No runtime imports.

Hashes bind the original tool-received output, not a new acceptance event.
An early rejection cannot be passed to the completed-batch auditor as success.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
FREEZE='f42a8d2a563764de4a2790c762b222ac4cca9600eb47ef00dffc4ff5233106c8'


def compact(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read(name, cap):
    path=ROOT/name
    assert path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
    with path.open('rb') as stream: raw=stream.read(cap+1)
    assert len(raw)<=cap
    return raw


def audit():
    begin=time.monotonic()
    freeze_raw=read('docs/hz_endpoint_midpoint_freeze_v4_20261019.json',131072)
    assert sha(freeze_raw)==FREEZE
    freeze=json.loads(freeze_raw)
    raw=read('docs/hz_endpoint_midpoint_v4_stdout_20261019.txt',65536)
    lines=raw.splitlines(); assert len(lines)==2
    envelope,export=map(json.loads,lines)
    assert envelope['schema']=='ENDPOINT_PROVISIONED_HOST_RETURN_V1'
    assert export['schema']=='ENDPOINT_ROOT_EXPORT_COST_V1'
    assert export['envelope_sha256']==sha(lines[0])
    caller=envelope['caller']
    assert sha(compact(caller))==envelope['caller_sha256']
    assert caller['schema']=='ENDPOINT_LAUNCH_FAILURE_V1'
    assert caller['stage']=='caller' and caller['error']=='ValueError'
    assert caller['message']=='pidfd capability before admission'
    assert caller['status']=='INCOMPLETE_OR_FAILED'
    assert caller['online_batch_acceptance'] is False and caller['all_work_complete'] is False
    assert caller['missing_cost_is_not_zero'] is True
    assert caller['seconds']==caller['observed_end']-caller['started']
    assert caller['observed_prefix_seconds']==caller['observed_end']-caller['observed_entry']
    assert caller['started']<=caller['observed_entry']<=caller['observed_end']
    assert 0<=caller['seconds']<=envelope['request_seconds']<60
    assert export['request_through_export_seconds']==envelope['request_seconds']+export['export_seconds']
    assert export['within_original_budget'] is True and export['request_through_export_seconds']<2100
    provision=envelope['provisioning']
    assert provision['interpreter_startup_seconds'] is None and provision['full_cold_start_seconds'] is None
    assert provision['scope']=='STANDARD_LIBRARY_ROOT_ONLY_NO_REQUEST_WORK' and provision['missing_is_not_zero']
    assert math.isfinite(provision['observed_prefix_seconds']) and provision['observed_prefix_seconds']>=0
    assert envelope['host']['python']==freeze['records'][0]['profile']['environment']['python']
    assert envelope['host']['flags']==['-I','-S','-B'] and envelope['live_receipt'] is True
    environment_raw=read('docs/hz_endpoint_midpoint_v4_environment_20261019.json',8192)
    environment=json.loads(environment_raw)
    assert environment['os_pidfd_open'] is False and environment['signal_pidfd_send_signal'] is False
    assert environment['executable']==envelope['host']['python']
    # Inspect the executed frozen source: the capability refusal precedes resource
    # admission, output mkdir and session spawn. Do not invoke any of those paths.
    name='scoped_source/endpoint_batch_observer.py'; code=read(name,131072)
    assert sha(code)==freeze['code'][name]
    tree=ast.parse(code)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    capability=next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Constant)
        and n.value=='pidfd capability before admission')
    for n in ast.walk(fn):
        if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in ('admission','owned_session'))
                or isinstance(n.func,ast.Attribute) and n.func.attr=='mkdir'):
            assert n.lineno>capability
    run='/data1/Kane/MOE/runs/hz_endpoint_midpoint_20261019_r2'
    assert not Path(run).exists()
    result=dict(schema='ENDPOINT_V4_EARLY_REJECTION_AUDIT_V1',
        status='VERIFIED_PRE_ADMISSION_REJECTION_NOT_CONTROL_PASS',freeze=FREEZE,
        execution_commit='7cddef31fba7061783fbd286d3e99ae8a04f368f',
        caller_sha256=envelope['caller_sha256'],stdout_sha256=sha(raw),environment_sha256=sha(environment_raw),
        attempted_launches=1,admissions=0,positions_started=0,component_controls_completed=0,
        unexecuted=[r['id'] for r in freeze['records']],models=0,gpu=0,retries=0,
        request_seconds=envelope['request_seconds'],export_seconds=export['export_seconds'],
        request_through_export_seconds=export['request_through_export_seconds'],
        provision_observed_prefix_seconds=provision['observed_prefix_seconds'],
        full_cold_start_seconds=None,online_batch_acceptance=False,work_complete=False,work=None,
        reason='Existing Python omits both pidfd APIs; no unsafe PID fallback or environment replacement',
        guarantee='Retained rejection/source-order audit; no new proof or measured algorithm effect')
    result['offline_audit_seconds']=time.monotonic()-begin
    return result


if __name__=='__main__':
    print(compact(audit()).decode())
