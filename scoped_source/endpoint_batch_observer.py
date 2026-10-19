"""Prepared default-off outer caller. No execution authority is registered.

A registered launcher supplies its LIVE pre-request start anchor and expected
hash before starting this caller. This caller never starts a fresh budget.
File timestamps/process birth are not substitutes for that origin or receiver.
The launcher/clock/OS are trusted; this is not a machine-checked OS proof.
"""
import math
import os
from pathlib import Path
import resource
import shutil
import time
import uuid

from scoped_proof.io import ROOT
from scoped_proof.owned_session import execute as owned_session, LOG_CAP, MEMBERS
from scoped_proof.pidfd_compat import Pidfd
from scoped_source.hz_request_store import compact, digest, identity, require
from scoped_source.native_protocol import PYTHON, read_json, write_json, REPORT_CAP
from scoped_source.native_supervised import clean_env
from scoped_source.endpoint_batch import validate, ORDER, LIMITS, FREEZE_CAP, RECEIPT_CAP, BATCH_META_CAP


def start_record(value, expected_hash):
    """Bounded immutable copy; a hash binds data, NOT an untrusted issuer's truth."""
    keys={'schema','freeze','issuer','root','code_root','clock','origin','started','nonce',
          'boot_id','time_namespace','launcher_pid','launcher_start_ticks'}
    require(type(value) is dict and set(value)==keys,'closed pre-launch anchor')
    value=dict(value)  # Validate/hash/return THIS snapshot, not a later caller-owned version.
    require(set(value)==keys,'closed snapshot after detachment')
    require(type(expected_hash) is str and len(expected_hash)==64
            and all(c in '0123456789abcdef' for c in expected_hash),'externally retained start hash')
    require(value['schema']=='ENDPOINT_TRUSTED_PRELAUNCH_V1'
            and value['clock']=='CLOCK_MONOTONIC'
            and value['origin']=='AFTER_HOST_PROVISIONING_BEFORE_REQUEST','trusted start meaning')
    for k in ('freeze','issuer'):
        require(type(value[k]) is str and len(value[k])==64
                and all(c in '0123456789abcdef' for c in value[k]),'start hash field')
    require(all(type(value[k]) is str and 0<len(value[k])<=512 for k in ('root','code_root'))
            and type(value['nonce']) is str and len(value['nonce'])==32
            and all(c in '0123456789abcdef' for c in value['nonce']),'bounded request binding')
    require(type(value['started']) in (int,float) and math.isfinite(value['started'])
            and 0<=value['started']<2**53,'finite original clock')
    require(type(value['boot_id']) is str and len(value['boot_id'])==36
            and str(uuid.UUID(value['boot_id']))==value['boot_id']
            and type(value['time_namespace']) is str and len(value['time_namespace'])<=64
            and value['time_namespace'].startswith('time:[') and value['time_namespace'].endswith(']')
            and value['time_namespace'][6:-1].isdigit(),'boot and clock namespace identity')
    require(type(value['launcher_pid']) is int and 0<value['launcher_pid']<2**31
            and type(value['launcher_start_ticks']) is int
            and 0<=value['launcher_start_ticks']<2**64,'launcher process incarnation')
    raw=compact(value)
    require(len(raw)<=4096 and digest(raw)==expected_hash,'bounded original start identity')
    # All validated values are immutable scalars.
    return value


def live_clock(start):
    """Local identity check only. /proc ticks are NEVER converted to a start time."""
    info=time.get_clock_info('monotonic')
    require(info.monotonic and not info.adjustable
            and info.implementation=='clock_gettime(CLOCK_MONOTONIC)','same Linux monotonic clock')
    with Path('/proc/sys/kernel/random/boot_id').open('rb') as stream: boot=stream.read(65)
    require(len(boot)<=64 and boot.decode('ascii').strip()==start['boot_id'], 'same boot')
    ns=os.readlink('/proc/self/ns/time')
    p=Path('/proc')/str(start['launcher_pid'])
    require(ns==start['time_namespace'] and p.stat().st_uid==os.getuid()
            and os.readlink(p/'ns/time')==ns,'same local launcher owner and clock namespace')
    with (p/'stat').open('rb') as stream: raw=stream.read(8193)
    require(len(raw)<=8192,'bounded launcher incarnation')
    close=raw.rfind(b') '); tail=raw[close+2:].split()
    require(close>0 and int(raw.split(b' ',1)[0])==start['launcher_pid'] and len(tail)>=20
            and tail[0] not in (b'Z',b'X',b'x')
            and int(tail[19])==start['launcher_start_ticks'],'same live launcher, no PID reuse')


def launch_start(a, freeze, expected, root):
    start=start_record(a['start_anchor'],a['start_sha256'])
    require(start['freeze']==expected and start['issuer']==freeze['startup']['issuer']
            and start['root']==str(root) and start['code_root']==str(ROOT)
            and start['started']==a['started'] and start['nonce']==a['nonce'],
            'one frozen launcher, request and original start')
    require(type(a['observer_entry']) in (int,float) and math.isfinite(a['observer_entry'])
            and a['started']<=a['observer_entry']<=a['admission_started']
            and a['startup_seconds']==a['observer_entry']-a['started'],
            'caller startup is charged, never a new function-entry budget')
    return start


def admission(deadline):
    """One finite observation; no retry/wait queue or signals to other jobs."""
    begin=time.monotonic(); require(begin<deadline,'single admission deadline')
    with Path('/proc/meminfo').open('rb') as f: raw=f.read(65537)
    require(len(raw)<=65536,'bounded meminfo')
    available=next(int(s.split()[1])*1024 for s in raw.splitlines() if s.startswith(b'MemAvailable:'))
    jobs=[]
    for p in Path('/proc').iterdir():
        require(time.monotonic()<deadline,'bounded job observation')
        if not p.name.isdecimal(): continue
        try:
            if p.stat().st_uid!=os.getuid(): continue
            with (p/'comm').open('rb') as f: raw=f.read(129)
            require(len(raw)<=128 and len(jobs)<512,'bounded own-job observation')
            jobs.append(dict(pid=int(p.name),command=raw.decode(errors='replace').strip()))
        except (FileNotFoundError,ProcessLookupError): continue
    require(time.monotonic()<deadline,'admission completed timely')
    return available,jobs,begin


def launch_check(a, freeze, expected, root):
    launch_start(a,freeze,expected,root)
    require(a['schema']=='ENDPOINT_BATCH_LAUNCH_V2' and a['freeze']==expected
            and a['root']==str(root) and a['code_root']==str(ROOT)
            and type(a['started']) in (int,float) and math.isfinite(a['started'])
            and a['started']>=0 and a['run_deadline']==a['started']+LIMITS['batch_run_seconds']
            and a['cleanup_deadline']==a['started']+LIMITS['batch_cleanup_seconds']
            and a['deadline']==a['started']+LIMITS['cumulative_seconds']
            and type(a['nonce']) is str and len(a['nonce'])==32,'bound observer launch schedule')
    require(a['attempts']==1 and 0<=a['seconds']<=60
            and a['started']<=a['admission_started']<=a['admission_end']<a['started']+60
            and a['seconds']==a['admission_end']-a['admission_started']
            and a['available_bytes']>=freeze['admission']['available_bytes']
            and a['disk_free_bytes']>=freeze['admission']['disk_free_bytes']
            and a['device']=='cpu' and a['threads']==1
            and a['admission_in_total_budget'] is True,'one charged resource admission')


def command(root, expected, launch_hash):
    # Starting this interpreter, its bootstrap, imports and every nested process
    # are charged to the parent's earlier monotonic start, never a fresh budget.
    bootstrap='''import os,resource,sys,time
resource.setrlimit(resource.RLIMIT_AS,(8589934592,8589934592))
resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728))
import hashlib,json
from pathlib import Path
root=Path(sys.argv[1]); repo=Path(sys.argv[4])
def read(name,expected,cap):
    p=root/name
    if p.is_symlink() or not p.is_file() or p.stat().st_nlink!=1 or p.stat().st_size>cap: raise ValueError("batch bootstrap file")
    with p.open("rb") as f: raw=f.read(cap+1)
    if len(raw)>cap or hashlib.sha256(raw).hexdigest()!=expected: raise ValueError("batch bootstrap identity")
    return json.loads(raw)
freeze=read("freeze.json",sys.argv[2],131072)
a=read("admission.json",sys.argv[3],1048576)
if a["root"]!=str(root) or a["code_root"]!=str(repo) or a["freeze"]!=sys.argv[2] or os.getsid(0)!=os.getpid(): raise ValueError("batch owner binding")
code=freeze["code"]
if type(code) is not dict or not 1<=len(code)<=256: raise ValueError("batch roster")
for n,h in code.items():
    if time.monotonic()>=a["run_deadline"]: raise TimeoutError("batch bootstrap deadline")
    p=repo/n
    if Path(n).is_absolute() or ".." in Path(n).parts or any(q.is_symlink() for q in (p,*p.parents)): raise ValueError("batch code path")
    if not p.is_file() or p.stat().st_nlink!=1 or p.stat().st_size>131072 or hashlib.sha256(p.read_bytes()).hexdigest()!=h: raise ValueError("batch code identity")
for n in ("scripts","scoped_source","scoped_proof","source_enclosure","upstream_source","router_source"):
    if (repo/n/"__init__.py").exists(): raise ValueError("unbound initializer")
if "scripts/run_endpoint_midpoint_controls.py" not in code: raise ValueError("batch entry missing")
sys.path.append(str(repo))
from scripts.run_endpoint_midpoint_controls import worker
raise SystemExit(worker(root,sys.argv[2],sys.argv[3]))
'''
    return [PYTHON,'-I','-S','-B','-c',bootstrap,str(root),expected,launch_hash,str(ROOT)]


def batch_check(root, freeze, expected, a, ah, sid, deadline=None):
    """Bounded summary/receipt reception. Slot checks ran inside each 300s slot.

    Offline audit independently rechecks the full saved evidence. This is not a
    second exact checker or another all-artifact census outside request budgets.
    """
    b,bh=read_json(root/'batch.json',deadline=deadline)
    require(b['schema']=='ENDPOINT_FINITE_BATCH_V3' and b['freeze']==expected
            and b['started']==a['started'] and b['admission_sha256']==ah
            and b['start_sha256']==a['start_sha256']
            and b['owner_session']==sid and b['worker_deadline']==a['run_deadline']
            and all(type(b[k]) is int and b[k]==0 for k in ('models','gpu','retries','quota_increases')),
            'worker identity, session and original clock')
    rows=b['positions']; require(type(rows) is list and len(rows)<=len(ORDER),'finite observed prefix')
    require([r['id'] for r in rows]==list(ORDER[:len(rows)])
            and b['unexecuted']==list(ORDER[len(rows):]),'no missing position hidden')
    previous=a['started']; good=True
    for row,spec in zip(rows,freeze['records']):
        if deadline is not None: require(time.monotonic()<deadline,'batch summary deadline')
        require(previous<=row['started']<=row['observed_end']
                and row['seconds']==row['observed_end']-row['started'],'nonoverlapping observed times')
        previous=row['observed_end']
        if row['ok'] is not True:
            require(row is rows[-1],'stop at first failed slot'); good=False; continue
        r,_=read_json(root/'runs'/row['id']/'slot.json',expected=row['receipt_sha256'],
                      cap=RECEIPT_CAP,deadline=deadline)
        require(r['freeze']==expected and r['profile']==spec['sha256'] and r['id']==row['id']
                and r['status']=='PASS' and r['error'] is None and r['started']==row['started']
                and r['observed']['owned_execution']['owner_session']==sid
                and r['before_publish']<=row['observed_end']<row['started']+300
                and row['overrun_seconds']==0,'durable timely slot bound to enclosing owner')
    require(previous<=b['before_publish']<a['run_deadline'],'worker final publication order')
    return b,bh,good and len(rows)==7 and b['stop'] is None and not b['unexecuted']


def observe_return(proposal, *, start_anchor, start_sha256):
    """Trusted caller invokes this AFTER execute actually returned.

    This observation is the receiver's trust boundary, not another disk receipt.
    Retain the returned anchor separately. Never call from the batch worker or
    recreate a missing observation later to upgrade an earlier partial run.
    """
    start=start_record(start_anchor,start_sha256); live_clock(start)
    require(proposal['schema']=='ENDPOINT_CALLER_PROPOSAL_V2'
            and proposal['start_sha256']==start_sha256 and proposal['freeze']==start['freeze']
            and proposal['nonce']==start['nonce'] and proposal['started']==start['started'],
            'live observer proposal bound to retained original start')
    require(type(proposal['producer_finished_at']) in (int,float)
            and math.isfinite(proposal['producer_finished_at'])
            and proposal['started']<=proposal['producer_finished_at'],
            'finite producer completion on original clock')
    # Charge start/clock validation before recording completed reception, rather
    # than reporting an earlier arrival that omits these (possibly delayed) reads.
    observed=time.monotonic()
    require(proposal['producer_finished_at']<=observed,'actual post-return observation, same clock')
    result={k:proposal[k] for k in ('freeze','launch_sha256','terminal_sha256','batch_sha256',
                                   'nonce','started','producer_finished_at','start_sha256',
                                   'observer_entry','startup_seconds')}
    result.update(schema='ENDPOINT_CALLER_RETURN_V2',received_at=observed,seconds=observed-proposal['started'],
        online_batch_acceptance=proposal['candidate_complete'] is True and observed<proposal['started']+2100,
        elapsed_observed=True,all_work_complete=False)
    # The actual root-of-trust event is receipt above, not this function's own
    # return/fsync. Binding is bounded bookkeeping and must also finish timely.
    if time.monotonic()>=proposal['started']+2100:
        result['online_batch_acceptance']=False
    return result


def receive(anchor, expected_hash, terminal, terminal_hash, a, ah, batch_hash):
    """Check a caller-provided LIVE return anchor, never synthesize it from files.

    Hash is an integrity binding, not authentication of an arbitrary new caller.
    The frozen trusted receiver and its original clock are still assumptions.
    """
    require(set(anchor)=={'schema','freeze','launch_sha256','terminal_sha256','batch_sha256',
        'nonce','started','producer_finished_at','received_at','seconds','online_batch_acceptance',
        'elapsed_observed','all_work_complete','start_sha256','observer_entry','startup_seconds'}
        and identity(anchor)==expected_hash and anchor['schema']=='ENDPOINT_CALLER_RETURN_V2'
        and anchor['freeze']==a['freeze'] and anchor['nonce']==a['nonce']
        and anchor['launch_sha256']==ah and anchor['terminal_sha256']==terminal_hash
        and anchor['batch_sha256']==batch_hash and anchor['started']==a['started']
        and anchor['start_sha256']==a['start_sha256']==terminal['start_sha256']
        and anchor['observer_entry']==a['observer_entry']
        and anchor['startup_seconds']==a['startup_seconds'],
        'externally retained caller return identity')
    require(type(anchor['received_at']) in (int,float) and math.isfinite(anchor['received_at'])
        and terminal['before_publish']<=anchor['producer_finished_at']<=anchor['received_at']
        and anchor['seconds']==anchor['received_at']-a['started']
        and type(anchor['online_batch_acceptance']) is bool and anchor['elapsed_observed'] is True
        and anchor['all_work_complete'] is False,'post-publication caller reception, not complete Work')
    if anchor['online_batch_acceptance']:
        require(anchor['received_at']<a['deadline'] and terminal['worker_summary_received'] is True
            and terminal['candidate_complete'] is True and terminal['error'] is None,
            'positive caller reception inside original budget')


def execute(root, freeze_path, expected, *, start_anchor, start_sha256, enabled=False):
    observer_entry=time.monotonic()  # Observation only, NEVER the budget origin.
    require(enabled is True,'default-off trusted batch observer')
    start=start_record(start_anchor,start_sha256); started=start['started']
    require(started<=observer_entry<started+60,'original startup/admission window')
    # The frozen measurement root imposes these before request start/imports.
    # Checking now cannot prove their past installation, and must not repair it.
    for kind,cap in ((resource.RLIMIT_AS,LIMITS['as_bytes']),(resource.RLIMIT_FSIZE,LIMITS['file_bytes'])):
        require(resource.getrlimit(kind)==(cap,cap),'required pre-request caller limits, never raise/repair')
    live_clock(start)
    freeze,_=read_json(freeze_path,expected=expected,cap=FREEZE_CAP,deadline=started+60)
    validate(freeze,expected,deadline=started+60)  # Empty authority refuses before side effects.
    require(LOG_CAP==LIMITS['batch_log_bytes'] and MEMBERS==LIMITS['batch_session_members'],
            'frozen observation storage bounds')
    Pidfd().probe()  # Both operations, before admission or any child creation.
    root=Path(root)
    require(root.is_absolute() and root.resolve()==root and root.is_relative_to('/data1/Kane/MOE')
            and root!=Path('/data1/Kane/MOE') and not root.exists() and root.parent.is_dir(),
            'new canonical batch, no resume')
    # Check request binding before observation/setup; authority never comes from this anchor alone.
    require(start['freeze']==expected and start['issuer']==freeze['startup']['issuer']
            and start['root']==str(root) and start['code_root']==str(ROOT),'registered pre-launch request')
    available,jobs,begin=admission(started+60); disk=shutil.disk_usage(root.parent).free; at=time.monotonic()
    a=dict(schema='ENDPOINT_BATCH_LAUNCH_V2',freeze=expected,root=str(root),code_root=str(ROOT),
        started=started,run_deadline=started+LIMITS['batch_run_seconds'],
        cleanup_deadline=started+LIMITS['batch_cleanup_seconds'],deadline=started+2100,
        nonce=start['nonce'],start_anchor=start,start_sha256=start_sha256,
        observer_entry=observer_entry,startup_seconds=observer_entry-started,
        attempts=1,available_bytes=available,disk_free_bytes=disk,jobs=jobs,
        admission_started=begin,admission_end=at,seconds=at-begin,device='cpu',threads=1,
        admission_in_total_budget=True)
    launch_check(a,freeze,expected,root)
    root.mkdir(); (root/'runs').mkdir()
    write_json(root/'freeze.json',freeze,deadline=a['run_deadline'],cap=FREEZE_CAP)
    ar=write_json(root/'admission.json',a,deadline=a['run_deadline'])
    execution=owned_session(command(root,expected,ar['sha256']),run_deadline=a['run_deadline'],
        cleanup_deadline=a['cleanup_deadline'],env=clean_env(),rss_limit=LIMITS['rss_bytes'])
    candidate=False; bh=None; error=None; publication=None; batch=None
    try:
        require(execution['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION',
                'quiescent session before reading a completed or failed prefix')
        batch,bh,candidate=batch_check(root,freeze,expected,a,ar['sha256'],execution['session'],a['deadline'])
        require(execution['status']=='COMPLETED','complete session exit for acceptance')
        require(time.monotonic()<a['deadline'],'observer reception deadline')
    except BaseException as exc:
        error=dict(kind=type(exc).__name__,message=str(exc)[:400]); candidate=False
    terminal=dict(schema='ENDPOINT_BATCH_OBSERVER_V2',freeze=expected,launch_sha256=ar['sha256'],
        start_sha256=start_sha256,
        batch_sha256=bh,owned_execution=execution,candidate_complete=candidate,error=error,
        before_publish=time.monotonic(),online_batch_acceptance=False,
        worker_summary_received=batch is not None,
        unexecuted=None if batch is None else batch['unexecuted'],missing_cost_is_not_zero=True)
    try:
        # 128KiB + 1MiB + 1MiB + 64KiB remain below the OLD four-file 4MiB reserve.
        require(FREEZE_CAP+2*REPORT_CAP+RECEIPT_CAP<=BATCH_META_CAP,'same metadata reserve')
        publication=write_json(root/'batch-received.json',terminal,deadline=a['deadline'],cap=RECEIPT_CAP)
        read_json(root/'batch-received.json',expected=publication['sha256'],deadline=a['deadline'],cap=RECEIPT_CAP)
    except BaseException as exc:
        error=dict(kind=type(exc).__name__,message=str(exc)[:400]); candidate=False
    # Still only a proposal. Its receiver must call observe_return AFTER this
    # function returns; producer timestamps cannot prove that event occurred.
    end=time.monotonic()
    proposal=dict(schema='ENDPOINT_CALLER_PROPOSAL_V2',freeze=expected,launch_sha256=ar['sha256'],
        start_sha256=start_sha256,observer_entry=observer_entry,startup_seconds=observer_entry-started,
        terminal_sha256=None if publication is None else publication['sha256'],batch_sha256=bh,
        nonce=a['nonce'],started=started,producer_finished_at=end,
        candidate_complete=candidate and error is None and end<a['deadline'],
        online_batch_acceptance=False)
    return proposal
