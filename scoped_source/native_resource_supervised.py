"""Default-off V2 joint intake, supervision and final caller reception.

No execution envelope is approved. Runtime controls are a later, separate
freeze. V1 modules, their policies and their historical approvals are untouched.
"""
import hashlib
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import uuid

from scoped_source.hz_request_store import POLICY, require
from scoped_source.native_protocol import (PYTHON, REPORT_CAP, LOG_CAP, META_RESERVE,
    META_MEMBERS, META_CAPS, STAGES, COUNTERS, counts, restore, hard_limits,
    validate_receipt as original_validate_receipt)
from scoped_source.native_resource_protocol import Session, ROOT, verify_code
from scoped_source.native_resources import bounded_read, parse, publish, code, tick, measure
from scoped_source.native_supervised import environment, receive, clean_env, append_suffix
from scoped_source.direct_supervised import census
from scoped_proof.owned_bounded import exited_unreaped
from scoped_proof.owned_resource_v2 import execute as run_owned, observe, complete


def metadata_caps(session):
    slots = session.resources.slots
    caps = {n:min(v,slots['metadata']['encoded']) for n,v in META_CAPS.items()}
    # This file is redirected by the owner, not the bounded actor pipe. Reserve
    # its actual old RLIMIT_FSIZE ceiling even if metadata JSON is smaller.
    caps['controller.log'] = min(REPORT_CAP,session.profile['file_bytes'])
    caps['terminal.json'] = min(REPORT_CAP,slots['terminal']['encoded'])
    caps['caller.json'] = min(REPORT_CAP,slots['caller']['encoded'])
    caps['envelope.json'] = len(session.raw)
    require(sum(caps.values()) <= META_RESERVE and len(caps) <= META_MEMBERS,
            'new metadata fits INSIDE original bytes/members reserve')
    return caps


def write(session, name, builder, *, slot='metadata'):
    cap = metadata_caps(session)[name]
    return publish(session.root/name,builder,session.resources,slot,cap=cap)


def read(session, name, expected=None, *, deadline=None):
    resources = session.resources
    with resources.hold('metadata') as slot:
        cap = metadata_caps(session)[name]
        raw, h = bounded_read(session.root/name,cap,deadline or resources.deadline,expected)
        value, _, _ = parse(raw,resources,'metadata')
        return value,h  # caller's phase reservation owns the retained object


def validate_receipt(value, **kwargs):
    require(value.get('schema') == 'NATIVE_RESOURCE_ACTOR_V2', 'V2 actor receipt')
    # All old chain, counter, deadline and COMPLETE gates are retained verbatim.
    copy = dict(value,schema='NATIVE_ACTOR_RECEIPT_V1')
    return original_validate_receipt(copy,**kwargs)


def command(role, session, invocation):
    require(role in ('controller',*STAGES), 'closed V2 actor')
    bootstrap = '''import resource,sys,time
started=time.monotonic()
a=int(sys.argv[1]); f=int(sys.argv[2]); role=sys.argv[3]
resource.setrlimit(resource.RLIMIT_AS,(a,a))
resource.setrlimit(resource.RLIMIT_FSIZE,(min(1048576,f) if role=="controller" else f,f))
import hashlib,json,os,stat
from pathlib import Path
code=Path(sys.argv[4]); root=Path(sys.argv[5]); deadline=float(sys.argv[9])
def read(p,cap,expected):
    if time.monotonic()>=deadline: raise TimeoutError("bootstrap deadline")
    if not p.is_absolute() or p.resolve()!=p or any(q.is_symlink() for q in (p,*p.parents)):
        raise ValueError("bootstrap path")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>cap:
            raise ValueError("bootstrap bounded regular input")
        with os.fdopen(fd,"rb",closefd=False) as stream: raw=stream.read(cap+1)
        after=os.fstat(fd); named=p.stat(follow_symlinks=False)
        key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
        if len(raw)>cap or len(raw)!=before.st_size or key(before)!=key(after) or key(after)!=key(named):
            raise ValueError("bootstrap changed input")
        if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError("bootstrap hash")
        if time.monotonic()>=deadline: raise TimeoutError("bootstrap read deadline")
        return raw
    finally: os.close(fd)
raw=read(root/"invocation.json",1048576,sys.argv[6]); inv=json.loads(raw)
envelope=read(root/"envelope.json",1048576,sys.argv[7])
roster=inv["profile"]["implementation"]
if type(roster) is not dict or not 1<=len(roster)<=256: raise ValueError("bootstrap roster")
for name,h in roster.items():
    if type(name) is not str or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("bootstrap relative code")
    read(code/name,131072,h)
for name in ("scoped_source","scoped_proof","source_enclosure","upstream_source","router_source"):
    if (code/name/"__init__.py").exists(): raise ValueError("unbound package initializer")
if "scoped_source/native_resource_supervised.py" not in roster: raise ValueError("V2 supervisor absent")
sys.path.append(str(code))
from scoped_source.native_resource_supervised import actor_main
actor_main(role,sys.argv[5],sys.argv[6],envelope,sys.argv[7],float(sys.argv[8]),started)
'''
    p = session.profile
    return [PYTHON,'-I','-S','-B','-c',bootstrap,str(p['as_bytes']),str(p['file_bytes']),role,
        str(ROOT),str(session.root),invocation,session.expected,str(session.started),str(session.started+299)]


def prior_chain(session, inv, expected, stop):
    initial = {k:0 for k in COUNTERS}; previous = expected; records = []
    for stage in STAGES[:STAGES.index(stop)]:
        report,h = read(session,stage+'.json',deadline=inv['run_deadline'])
        initial = validate_receipt(report,stage=stage,invocation=expected,previous=previous,initial=initial,
            deadline=inv['run_deadline'],started=inv['started'])
        require(report['bindings'] == session.refs(), 'all actors use same C/L/E/R')
        previous = h; records.append(report)
    return initial,previous,records


def actor_main(role, root, invhash, envelope, expected, started, boot_started):
    session = Session(envelope,expected,role,started)
    try:
        with session.resources.hold('phase'):
            inv,_ = read(session,'invocation.json',invhash,deadline=started+299)
            session.verify_invocation(inv,Path(root))
            limits = hard_limits(session.profile,role)
            with session.resources.hold('metadata'):
                verify_code(ROOT,session.profile['implementation'],inv['run_deadline'])
            if role == 'controller': return controller(session,inv,invhash,limits,boot_started)
            require(role in STAGES, 'V2 phase')
            initial,previous,prior = prior_chain(session,inv,invhash,role)
            work = restore(initial,inv['run_deadline']); output = None; error = None
            status = 'ERROR'; setup = None; progress_bytes = 0

            def progress(name,w):
                nonlocal progress_bytes
                from scoped_source.native_resources import materialize
                w.tick()
                raw = materialize(dict(phase=name,work=dict(w.counts),at=time.monotonic(),complete=False),
                    session.resources,'metadata',LOG_CAP)+b'\n'
                require(progress_bytes+len(raw) <= LOG_CAP, 'bounded cumulative progress')
                with (session.root/(role+'.progress')).open('xb' if progress_bytes == 0 else 'ab',buffering=0) as f:
                    require(f.write(raw) == len(raw), 'progress short write')
                progress_bytes += len(raw); w.tick()

            try:
                progress('actor_start',work)
                if role == 'production':
                    setup = environment(session.profile,work)
                    from scoped_source.native_resource_production import produce
                    output = produce(session.root,session.profile['source'],session.profile['method'],invhash,
                        ROOT,session.profile['implementation'],work,session=session,invocation_record=inv,
                        enabled=True,progress=progress)
                elif role == 'replay':
                    from scoped_source.native_resource_portable import check_bound_package
                    production = prior[0]['output']
                    checked = check_bound_package(session.root/'artifacts/package',
                        expected_bundle=production['package']['sha256'],
                        expected_capsule=production['snapshots']['resource_binding']['sha256'],
                        expected_envelope=session.expected,expected_invocation=invhash,
                        resources=session.resources,work=work,enabled=True)
                    output = checked['result']
                else:
                    # Original independent aggregation, package seals and census.
                    output = receive(session.root,session.profile,invhash,prior,work)
                if role != 'production':
                    require(not any(n.split('.')[0] in ('torch','numpy','scipy','act') for n in sys.modules),
                            'fresh standard-library checker/reception')
                progress('actor_complete',work); work.tick(); status = 'COMPLETE'
            except BaseException as exc:
                error = code(exc); status = 'TIMEOUT' if isinstance(exc,TimeoutError) else 'ERROR'
            final = counts(work.counts,complete=status == 'COMPLETE')
            write(session,role+'.json',lambda:dict(schema='NATIVE_RESOURCE_ACTOR_V2',stage=role,
                status=status,error=error,invocation=invhash,bindings=session.refs(),previous=previous,
                initial_work=initial,work=final,delta_work={k:final[k]-initial[k] for k in COUNTERS},
                begin=boot_started,end=time.monotonic(),deadline=inv['run_deadline'],output=output,
                limits=limits,native_setup=setup,progress_bytes=progress_bytes,
                resources=session.resources.receipt(),actor_seconds=time.monotonic()-boot_started))
            if status != 'COMPLETE': raise RuntimeError('INCOMPLETE_ACTOR')
    finally:
        session.close()


def run_actor(session,role,invhash,deadline):
    begin = time.monotonic(); p = None; log_bytes = 0; eof = False
    selector = selectors.DefaultSelector()
    try:
        tick(deadline)
        p = subprocess.Popen(command(role,session,invhash),cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=clean_env(),start_new_session=False)
        os.set_blocking(p.stdout.fileno(),False); selector.register(p.stdout,selectors.EVENT_READ)
        cap = metadata_caps(session)[role+'.log']
        with (session.root/(role+'.log')).open('xb',buffering=0) as log:
            while True:
                tick(deadline)
                for key,_ in selector.select(min(.01,max(0.,deadline-time.monotonic()))):
                    raw = os.read(key.fileobj.fileno(),min(16384,cap-log_bytes+1))
                    if not raw:
                        selector.unregister(key.fileobj); eof = True; continue
                    require(log_bytes+len(raw) <= cap,'actor log capacity')
                    require(log.write(raw) == len(raw),'actor log short write'); log_bytes += len(raw)
                if exited_unreaped(p.pid) is not None and eof:
                    rc = p.wait(timeout=max(.001,deadline-time.monotonic()))
                    tick(deadline); require(rc == 0,'actor exit')
                    with session.resources.hold('observer'),session.resources.hold('observer_next'):
                        group = observe(os.getpgrp(),invhash,'between_actors',session.resources,deadline)
                        require(group['complete'] and group['live'] == [os.getpid()], 'complete previous actor exit')
                    return dict(stage=role,begin=begin,end=time.monotonic(),seconds=time.monotonic()-begin,
                        log_bytes=log_bytes,returncode=rc,remaining_live_actor_processes=[])
    finally:
        selector.close()
        if p is not None and p.stdout is not None: p.stdout.close()
        # Outer owner still holds the unreaped controller leader and cleans the
        # group after any failure. Do not launch the next phase on failure.


def controller(session,inv,invhash,limits,started):
    metadata_caps(session)
    stages = []; initial = {k:0 for k in COUNTERS}; previous = invhash
    for stage in STAGES:
        stages.append(run_actor(session,stage,invhash,inv['run_deadline']))
        r,h = read(session,stage+'.json',deadline=inv['run_deadline'])
        initial = validate_receipt(r,stage=stage,invocation=invhash,previous=previous,initial=initial,
            deadline=inv['run_deadline'],started=inv['started'])
        require(r['bindings'] == session.refs(),'same actor bindings'); previous = h
    storage = census(session.root,inv['run_deadline'])
    caps = metadata_caps(session)
    # Caller receipt is now explicit and counted, not an uncharged outside file.
    reserve = sum(caps[n] for n in ('controller.json','terminal.json','caller.json'))
    require(storage['bytes']+reserve <= POLICY['total_bytes']
            and len(storage['files'])+3 <= POLICY['members'], 'all final records reserved')
    write(session,'controller.json',lambda:dict(schema='NATIVE_RESOURCE_CONTROLLER_V2',status='COMPLETE',
        invocation=invhash,bindings=session.refs(),last_receipt=previous,result=r['output']['result'],
        work=initial,phases=stages,limits=limits,storage_before_terminals=storage,
        started=started,end=time.monotonic(),deadline=inv['run_deadline'],received_all_obligations=True,
        resources=session.resources.receipt()))


def suffix(session,name,deadline):
    cap = metadata_caps(session)[name]
    with session.resources.hold('metadata') as slot:
        # Archive allowance is not permission to exceed the live read backing.
        # An unusually large controller log remains failure evidence, not an
        # implicit expansion of the JSON/read reservation.
        raw,h = bounded_read(session.root/name,min(cap,slot['encoded']),deadline)
        return dict(bytes=len(raw),sha256=h)


def finalize(session,invhash,pending):
    """Publish pending evidence only; the caller must close before accepting."""
    resources = session.resources; deadline = session.deadline
    durable = caller = None; error = pending['error']; storage = pending.get('storage')
    eligible = pending['eligible']
    try:
        durable = write(session,'terminal.json',lambda:dict(schema='NATIVE_RESOURCE_TERMINAL_V2',
            invocation=invhash,bindings=session.refs(),pending=pending,
            observed_at=time.monotonic(),online_report_received=False,online_positive=False),slot='terminal')
        if storage is not None: append_suffix(storage,'terminal.json',durable)
        caller = write(session,'caller.json',lambda:dict(schema='NATIVE_RESOURCE_CALLER_V2',
            invocation=invhash,bindings=session.refs(),terminal=durable,error=error,
            pending_eligible=eligible,storage_before_self=storage,started=session.started,
            observed_before_publication=time.monotonic(),resources=resources.receipt()),slot='caller')
        if storage is not None: append_suffix(storage,'caller.json',caller)
        tick(deadline)
    except BaseException as exc:
        eligible = False; error = code(exc)
        # Publication may have created a partial file. Never report the older
        # successful prefix as a complete storage census. Recount if timely;
        # otherwise explicitly leave it unknown, without an offline upgrade.
        try: storage = census(session.root,deadline)
        except BaseException: storage = None
    out = dict(schema='NATIVE_RESOURCE_RETURN_V2',invocation=invhash,bindings=session.refs(),
        terminal_ref=durable,caller_ref=caller,error=error,storage=storage,
        storage_complete=storage is not None and caller is not None and error is None,
        result=pending.get('result'),work=pending.get('work'),owned_execution=pending.get('owned_execution'),
        online_report_received=False,online_positive=False,started=session.started,
        guarantee='DECLARED_SOURCE_NOT_CHECKPOINT_OR_NATIVE_FLOAT_PROOF')
    out.update(pending_eligible=eligible and error is None and durable is not None and caller is not None,
               status=pending['status'])
    return out


def execute(envelope, expected, *, started, enabled=False):
    # The separately frozen launcher must capture t0 BEFORE importing this
    # module / materializing invocation input. No free startup or clock reset.
    now = time.monotonic()
    require(type(started) in (int,float) and math.isfinite(started)
            and started <= now < started+300, 'externally observed original request start')
    require(enabled is True, 'V2 explicit opt-in')
    # Empty V2 registry refuses here, before mkdir, imports, limits or children.
    session = Session(envelope,expected,'caller',started)
    ih = None; created = False; out = None
    pending = dict(status='ERROR',eligible=False,error=None,result=None,work=None,
        storage=None,owned_execution=None,scope='ALL_REQUEST_COSTS_INCLUDING_FAILURES')
    try:
        with session.resources.hold('phase'):
            try:
                root = session.root
                require(not root.exists() and not root.is_symlink(),'fresh V2 directory only')
                metadata_caps(session)
                with session.resources.hold('metadata'):
                    verify_code(ROOT,session.profile['implementation'],started+299)
                root.mkdir(parents=False,exist_ok=False); created = True
                write(session,'envelope.json',lambda:session.envelope)
                inv = dict(schema='NATIVE_RESOURCE_INVOCATION_V2',root=str(root),code_root=str(ROOT),
                    profile=session.profile,bindings=session.refs(),envelope=session.expected,nonce=uuid.uuid4().hex,
                    started=started,run_deadline=started+299,cleanup_deadline=started+299.5,deadline=started+300)
                ref = write(session,'invocation.json',lambda:inv); ih = ref['sha256']
                run = run_owned(command('controller',session,ih),root/'controller.log',cwd=ROOT,
                    run_deadline=inv['run_deadline'],cleanup_deadline=inv['cleanup_deadline'],env=clean_env(),
                    rss_limit=session.profile['rss_bytes'],resources=session.resources,binding=ih)
                pending.update(status=run['status'],owned_execution=run)
                require(complete(run,inv['cleanup_deadline']) and time.monotonic() < inv['deadline'],
                        'complete execution/reap/empty-live observation and timely reception')
                r,h = read(session,'controller.json',deadline=inv['deadline'])
                require(r.get('schema') == 'NATIVE_RESOURCE_CONTROLLER_V2' and r.get('status') == 'COMPLETE'
                    and r.get('invocation') == ih and r.get('bindings') == session.refs()
                    and r.get('received_all_obligations') is True and r['deadline'] == inv['run_deadline']
                    and r['end'] < inv['run_deadline'],'bound timely controller')
                counts(r['work'])
                storage = dict(r['storage_before_terminals'])
                storage['files'] = dict(storage['files']); storage['hashes'] = dict(storage['hashes'])
                for name in ('controller.json','controller.log'):
                    tail = suffix(session,name,inv['deadline'])
                    if name == 'controller.json': require(tail['sha256'] == h,'durable controller identity')
                    append_suffix(storage,name,tail)
                pending.update(status='COMPLETE',eligible=True,result=r['result'],work=r['work'],storage=storage)
            except BaseException as exc:
                pending.update(eligible=False,error=code(exc))
                if isinstance(exc,TimeoutError): pending['status'] = 'TIMEOUT'
                elif pending['status'] == 'COMPLETED': pending['status'] = 'ERROR'
                # A failed prefix is never inferred from a successful earlier
                # census. Recount only our newly created root, time permitting.
                if created:
                    try: pending['storage'] = census(session.root,session.deadline)
                    except BaseException: pending['storage'] = None
            if created:
                out = finalize(session,ih,pending)
            else:
                # Never write into a directory that this invocation did not
                # exclusively create, even to record why it was refused.
                out = dict(schema='NATIVE_RESOURCE_RETURN_V2',invocation=ih,bindings=session.refs(),
                    terminal_ref=None,caller_ref=None,error=pending['error'],storage=None,
                    storage_complete=False,result=None,work=None,owned_execution=None,
                    online_report_received=False,online_positive=False,pending_eligible=False,
                    started=started,status='EARLY_REJECTED',
                    guarantee='DECLARED_SOURCE_NOT_CHECKPOINT_OR_NATIVE_FLOAT_PROOF')
            # Retained return data is owned by the caller result contract, not
            # evidence that all interpreter objects/RSS have been freed.
    except BaseException as exc:
        if out is None:
            out = dict(schema='NATIVE_RESOURCE_RETURN_V2',invocation=ih,bindings=session.refs(),
                terminal_ref=None,caller_ref=None,storage=None,storage_complete=False,
                result=None,work=None,owned_execution=pending['owned_execution'],started=started)
        out.update(error=code(exc),pending_eligible=False,online_report_received=False,
                   online_positive=False,status='ERROR')
    finally:
        try: session.close()
        except BaseException as exc:
            if out is not None:
                out.update(error=code(exc),pending_eligible=False,online_report_received=False,
                           online_positive=False,status='ERROR')
    # Explicit phase/session release and owned-group cleanup have finished.
    # Only THIS return anchor can accept; durable files remain pending records.
    end = time.monotonic(); deadline = started+300
    accepted = bool(out.pop('pending_eligible',False) and out.get('error') is None and end < deadline)
    result = out.get('result')
    out.update(observed_at=end,seconds=end-started,overrun_seconds=max(0.,end-deadline),
        online_report_received=accepted,
        online_positive=bool(accepted and result is not None
            and result['status'] == 'CHECKED_POSITIVE_CAPTURED_REAL_SOURCE'
            and result['positive'] == result['required']),
        status='COMPLETE' if accepted else 'TIMEOUT' if end >= deadline else
            out['status'] if out['status'] not in ('COMPLETE','COMPLETED') else 'ERROR')
    return out
