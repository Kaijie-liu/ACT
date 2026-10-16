"""Optional fresh-actor native pipeline, with no approved runnable profile.

One owned process group contains a stdlib controller and sequential production,
replay and reception actors. Only production imports Torch. The outer caller
confirms group cleanup and timely observation; disk COMPLETE alone never accepts.
Per-process AS is hard-enforced; aggregate RSS remains sampled, not an AS proof.
Actors are trusted not to daemonize or escape this group, as in the old supervisor.
"""
import hashlib
import os
from pathlib import Path
import selectors
import stat
import subprocess
import sys
import time
import uuid

from scoped_source.hz_request_store import POLICY, compact, identity, require
from scoped_source.native_protocol import (PYTHON, SITE, REPORT_CAP, LOG_CAP, META_RESERVE,
    COUNTERS, STAGES, counts, restore, read_json, write_json, validate_profile,
    verify_code, hard_limits, validate_receipt, approval_registry, META_CAPS, META_MEMBERS)
from scoped_source.native_package import CODE_FILES, manifest, check_package
from scoped_source.direct_supervised import census
from scoped_proof.io import ROOT
from scoped_proof.owned_bounded import execute as run_owned, observe, exited_unreaped


def command(role, root, invocation, profile):
    require(role in ('controller', *STAGES), 'closed native actor')
    # Limits precede all project and numerical imports in each fresh interpreter.
    # The stdlib bootstrap and OS startup are trusted and fully time-charged.
    bootstrap = '''import resource,sys,time
started=time.monotonic()
a=int(sys.argv[1]); f=int(sys.argv[2]); role=sys.argv[3]
resource.setrlimit(resource.RLIMIT_AS,(a,a))
resource.setrlimit(resource.RLIMIT_FSIZE,(min(1048576,f) if role=="controller" else f,f))
import hashlib,json
from pathlib import Path
code=Path(sys.argv[4]); request=Path(sys.argv[5])/"invocation.json"
if request.is_symlink() or request.stat().st_size>1048576: raise ValueError("bootstrap invocation")
raw=request.read_bytes()
if len(raw)>1048576 or hashlib.sha256(raw).hexdigest()!=sys.argv[6]: raise ValueError("bootstrap identity")
inv=json.loads(raw); roster=inv["profile"]["implementation"]; deadline=inv["run_deadline"]
if type(roster) is not dict or not 1<=len(roster)<=256: raise ValueError("bootstrap roster")
for name,h in roster.items():
    if time.monotonic()>=deadline: raise TimeoutError("bootstrap deadline")
    p=code/name
    if Path(name).is_absolute() or ".." in Path(name).parts or any(q.is_symlink() for q in (p,*p.parents)):
        raise ValueError("bootstrap code path")
    if not p.is_file() or p.stat().st_nlink!=1 or p.stat().st_size>131072: raise ValueError("bootstrap code capacity")
    if hashlib.sha256(p.read_bytes()).hexdigest()!=h: raise ValueError("bootstrap code changed")
for name in ("scoped_source","scoped_proof","source_enclosure","upstream_source","router_source"):
    if (code/name/"__init__.py").exists(): raise ValueError("unbound package initializer")
if "scoped_source/native_supervised.py" not in roster: raise ValueError("bootstrap supervisor missing")
if time.monotonic()>=deadline: raise TimeoutError("bootstrap final deadline")
sys.path.append(str(code))
from scoped_source.native_supervised import actor_main
actor_main(role,sys.argv[5],sys.argv[6],started)
'''
    return [PYTHON,'-I','-S','-B','-c',bootstrap,str(profile['as_bytes']),str(profile['file_bytes']),
            role,str(ROOT),str(root),invocation]


def invocation(root, expected):
    value, h = read_json(Path(root)/'invocation.json',expected=expected)
    require(type(value) is dict and set(value) == {'schema','root','code_root','profile','approval','nonce','started',
        'run_deadline','cleanup_deadline','deadline'} and value['schema'] == 'NATIVE_INVOCATION_V1'
        and value['root'] == str(root) and value['code_root'] == str(ROOT), 'bound native invocation')
    profile = validate_profile(value['profile'])
    require(value['approval'] == approval_registry()[1], 'unchanged externally approved execution registry')
    require(value['run_deadline'] == value['started']+299
            and value['cleanup_deadline'] == value['started']+299.5
            and value['deadline'] == value['started']+300 and time.monotonic() < value['run_deadline'],
            'unchanged cumulative deadline schedule')
    return value, profile


def prior_chain(root, inv, expected, stop):
    initial = {k:0 for k in COUNTERS}; previous = expected; records = []
    for stage in STAGES[:STAGES.index(stop)]:
        report, h = read_json(Path(root)/(stage+'.json'),deadline=inv['run_deadline'])
        initial = validate_receipt(report,stage=stage,invocation=expected,previous=previous,initial=initial,
            deadline=inv['run_deadline'],started=inv['started'])
        previous = h; records.append(report)
    return initial, previous, records


def environment(profile, work):
    started = time.monotonic()
    # Frozen files are primary artifacts, not a claim to attest the full OS stack.
    for r in profile['environment']['files']:
        work.tick(); p = Path(r['file'])
        require(p.is_file() and p.stat().st_size == r['bytes'], 'native environment file length')
        h = hashlib.sha256(); size = 0
        with p.open('rb') as stream:
            for raw in iter(lambda:stream.read(65536), b''):
                work.tick(); size += len(raw); h.update(raw)
                require(size <= r['bytes'], 'native environment file grew')
        require(size == r['bytes'] and h.hexdigest() == r['sha256'], 'native environment identity')
    require(SITE not in sys.path and not any(n.split('.')[0] in ('torch','numpy','scipy','act') for n in sys.modules),
            'native setup happens only after limits and bindings')
    sys.path.append(SITE)
    import torch
    require(str(torch.__version__) == profile['environment']['torch_version'], 'frozen native version')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'one CPU native thread')
    work.tick()
    return dict(seconds=time.monotonic()-started,torch_version=str(torch.__version__),device='cpu',threads=1,
                environment_scope='BOUND_PRIMARY_FILES_NOT_FULL_OS_ATTESTATION')


def receive(root, profile, invhash, prior, work):
    production, replay = (r['output'] for r in prior)
    package = Path(root)/'artifacts/package'
    code = {n:profile['implementation'][n] for n in CODE_FILES}
    expected = production['package']['sha256']
    m, _ = manifest(package,expected,profile['method'],invhash,work,code,profile['source']['source'])
    require(production['result'] == replay['result'] and production['result_ref'] == replay['result_ref'] == m['result']
            and production['request'] == replay['request'] and replay['package_sha256'] == expected
            and production['source'] == replay['source'] == profile['source']['source'], 'three-phase evidence binding')
    from scoped_source.check_result_stream import aggregate
    result = aggregate(package,m['result'],m['tiles'],replay['request'],replay['context'],work)
    require(result == replay['result'], 'independent complete obligation aggregation')
    inventory = census(root,work.deadline)
    require(inventory['bytes'] <= POLICY['total_bytes'] and len(inventory['files']) <= POLICY['members'],
            'whole request storage including all phases')
    return dict(result=result,result_ref=m['result'],package_sha256=expected,
                storage_before_final_metadata=inventory,source=profile['source']['source'],
                declared_source_only=True,checkpoint_correspondence_checked=False,native_float_proof=False)


def actor_main(role, root, invhash, boot_started):
    root = Path(root); inv, profile = invocation(root,invhash)
    limits = hard_limits(profile,role)
    verify_code(ROOT,profile['implementation'],inv['run_deadline'])
    if role == 'controller':
        return controller(root,inv,invhash,profile,limits,boot_started)
    require(role in STAGES, 'native phase')
    initial, previous, prior = prior_chain(root,inv,invhash,role)
    work = restore(initial,inv['run_deadline']); output = None; error = None; status = 'ERROR'
    setup = None; progress_bytes = 0

    def progress(name, w):
        nonlocal progress_bytes
        w.tick(); raw = compact(dict(phase=name,work=dict(w.counts),at=time.monotonic(),complete=False))+b'\n'
        require(progress_bytes+len(raw) <= LOG_CAP, 'bounded partial progress')
        with (root/(role+'.progress')).open('xb' if progress_bytes == 0 else 'ab',buffering=0) as stream:
            require(stream.write(raw) == len(raw), 'progress short write')
        progress_bytes += len(raw); w.tick()

    try:
        progress('actor_start',work)
        if role == 'production':
            setup = environment(profile,work)
            from scoped_source.native_production import produce
            output = produce(root,profile['source'],profile['method'],invhash,ROOT,profile['implementation'],
                             work,enabled=True,progress=progress)
        elif role == 'replay':
            output = check_package(root/'artifacts/package',expected=prior[0]['output']['package']['sha256'],
                expected_method=profile['method'],invocation=invhash,expected_source=profile['source']['source'],
                expected_code={n:profile['implementation'][n] for n in CODE_FILES},work=work,enabled=True)
        else:
            output = receive(root,profile,invhash,prior,work)
        if role != 'production':
            require(not any(n.split('.')[0] in ('torch','numpy','scipy','act') for n in sys.modules),
                    'fresh standard-library independent actor')
        progress('actor_complete',work); work.tick(); status = 'COMPLETE'
    except BaseException as exc:
        error = dict(kind=type(exc).__name__,message=str(exc)[:400])
        status = 'TIMEOUT' if isinstance(exc,TimeoutError) else 'ERROR'
    final = counts(work.counts,complete=status == 'COMPLETE')
    report = dict(schema='NATIVE_ACTOR_RECEIPT_V1',stage=role,status=status,error=error,
        invocation=invhash,previous=previous,initial_work=initial,work=final,
        delta_work={k:final[k]-initial[k] for k in COUNTERS},begin=boot_started,end=time.monotonic(),
        deadline=inv['run_deadline'],output=output,limits=limits,native_setup=setup,
        progress_bytes=progress_bytes,actor_seconds=time.monotonic()-boot_started)
    write_json(root/(role+'.json'),report,deadline=inv['run_deadline'])
    if status != 'COMPLETE': raise RuntimeError('native actor incomplete; preserved evidence is not acceptance')


def run_actor(root, role, invhash, profile, deadline):
    begin = time.monotonic(); p = None; log_bytes = 0; eof = False
    selector = selectors.DefaultSelector()
    try:
        require(begin < deadline, 'actor launch deadline')
        p = subprocess.Popen(command(role,root,invhash,profile),cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=clean_env(),start_new_session=False)
        os.set_blocking(p.stdout.fileno(),False); selector.register(p.stdout,selectors.EVENT_READ)
        with (root/(role+'.log')).open('xb',buffering=0) as log:
            while True:
                require(time.monotonic() < deadline, 'one actor deadline')
                for key,_ in selector.select(min(.01,max(0.,deadline-time.monotonic()))):
                    raw = os.read(key.fileobj.fileno(),min(16384,LOG_CAP-log_bytes+1))
                    if not raw:
                        selector.unregister(key.fileobj); eof = True; continue
                    require(log_bytes+len(raw) <= LOG_CAP, 'actor log limit, not truncated success')
                    require(log.write(raw) == len(raw), 'actor log short write'); log_bytes += len(raw)
                exited = exited_unreaped(p.pid)
                if exited is not None and eof:
                    rc = p.wait(timeout=max(.001,deadline-time.monotonic()))
                    require(time.monotonic() < deadline and rc == 0, 'timely successful actor exit')
                    group = observe(os.getpgrp())
                    require(group['live'] == [os.getpid()], 'previous actor fully exited before next actor')
                    return dict(stage=role,begin=begin,end=time.monotonic(),seconds=time.monotonic()-begin,
                                log_bytes=log_bytes,returncode=rc,remaining_live_actor_processes=[])
    finally:
        selector.close()
        if p is not None and p.stdout is not None: p.stdout.close()
        # Never start a new phase after failure. The outer owner retains the
        # controller leader PID and kills/reaps the entire group under its deadline.


def controller(root, inv, invhash, profile, limits, started):
    require(sum(META_CAPS.values()) <= META_RESERVE and len(META_CAPS) <= META_MEMBERS,
            'closed reserved metadata roster')
    stages = []; initial = {k:0 for k in COUNTERS}; previous = invhash
    for stage in STAGES:
        record = run_actor(root,stage,invhash,profile,inv['run_deadline']); stages.append(record)
        r,h = read_json(root/(stage+'.json'),deadline=inv['run_deadline'])
        initial = validate_receipt(r,stage=stage,invocation=invhash,previous=previous,initial=initial,
            deadline=inv['run_deadline'],started=inv['started']); previous = h
    storage = census(root,inv['run_deadline'])
    require(storage['bytes']+2*REPORT_CAP <= POLICY['total_bytes']
            and len(storage['files'])+2 <= POLICY['members'], 'controller/caller terminal reserve')
    final = dict(schema='NATIVE_CONTROLLER_RECEIPT_V1',status='COMPLETE',invocation=invhash,
        last_receipt=previous,result=r['output']['result'],work=initial,phases=stages,limits=limits,
        storage_before_two_terminals=storage,started=started,end=time.monotonic(),
        deadline=inv['run_deadline'],received_all_obligations=True)
    write_json(root/'controller.json',final,deadline=inv['run_deadline'])


def clean_env():
    # No PYTHONPATH/user site or inherited multithread overrides.
    return {k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','TZ')} | {
        'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}


def execute(root, profile, *, enabled=False):
    started = time.monotonic()
    require(enabled is True, 'native supervision opt-in')
    profile = validate_profile(profile)  # Empty registry: refuses before side effects.
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_relative_to('/data1/Kane/MOE')
            and root != Path('/data1/Kane/MOE') and not root.exists() and not root.is_symlink(), 'new native run directory')
    inv = dict(schema='NATIVE_INVOCATION_V1',root=str(root),code_root=str(ROOT),profile=profile,approval=approval_registry()[1],
        nonce=uuid.uuid4().hex,started=started,run_deadline=started+299,cleanup_deadline=started+299.5,deadline=started+300)
    # Full dependency traversal is performed in the hard-limited controller and
    # actors, not in this caller. This small bootstrap remains trusted.
    root.mkdir(parents=False,exist_ok=False)
    ref = write_json(root/'invocation.json',inv,deadline=inv['run_deadline']); ih = ref['sha256']
    run = run_owned(command('controller',root,ih,profile),root/'controller.log',run_deadline=inv['run_deadline'],
        cleanup_deadline=inv['cleanup_deadline'],env=clean_env(),rss_limit=profile['rss_bytes'],bounded_errors=True)
    accepted = False; receipt = None; error = None
    try:
        require(run['status'] == 'COMPLETED' and run['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_GROUP',
                'complete native group execution and cleanup')
        receipt,controller_hash = read_json(root/'controller.json',deadline=inv['deadline'])
        require(receipt.get('schema') == 'NATIVE_CONTROLLER_RECEIPT_V1' and receipt.get('status') == 'COMPLETE'
                and receipt.get('invocation') == ih and receipt.get('deadline') == inv['run_deadline']
                and receipt.get('received_all_obligations') is True and receipt['end'] < inv['run_deadline'],
                'bound timely controller reception')
        counts(receipt['work'])
        storage = dict(receipt['storage_before_two_terminals'])
        storage['files'] = dict(storage['files']); storage['hashes'] = dict(storage['hashes'])
        for name in ('controller.json','controller.log'):
            suffix = suffix_ref(root/name,inv['deadline'])
            if name == 'controller.json': require(suffix['sha256'] == controller_hash, 'durable controller receipt')
            append_suffix(storage,name,suffix)
        accepted = True
    except Exception as exc:
        error = dict(kind=type(exc).__name__,message=str(exc)[:400])
    terminal = dict(schema='NATIVE_SUPERVISED_TERMINAL_V1',invocation=ih,
        status='COMPLETE' if accepted else run['status'] if run['status'] != 'COMPLETED' else 'ERROR',
        result=receipt['result'] if accepted else None,error=error,owned_execution=run,
        cumulative_work=receipt['work'] if accepted else None,work_complete=accepted,
        storage=storage if accepted else None,storage_complete=False,
        storage_scope='SUPERVISED_CENSUS_PLUS_QUIESCENT_SUFFIX_BEFORE_SELF',
        started=started,observed_at=time.monotonic(),seconds=time.monotonic()-started,
        source_scope='DECLARED_SOURCE_NOT_CHECKPOINT_OR_NATIVE_FLOAT_PROOF',online_acceptance=False)
    terminal_ref = None
    try:
        terminal_ref = write_json(root/'terminal.json',terminal,deadline=inv['deadline'])
        durable = suffix_ref(root/'terminal.json',inv['deadline'])
        require(durable == terminal_ref, 'durable final publication')
        if accepted:
            append_suffix(storage,'terminal.json',durable)
            terminal.update(storage=storage,storage_complete=True,terminal_ref=durable,
                            storage_scope='SUPERVISED_CENSUS_PLUS_QUIESCENT_FINAL_SUFFIX')
        accepted = accepted and time.monotonic() < inv['deadline']
    except Exception as exc:
        accepted = False
        terminal['error'] = dict(kind=type(exc).__name__,message=str(exc)[:400],phase='caller_finalization')
        terminal['status'] = 'TIMEOUT' if isinstance(exc,TimeoutError) or time.monotonic() >= inv['deadline'] else 'ERROR'
    terminal.update(online_acceptance=accepted,caller_seconds=time.monotonic()-started,
                    attempted_terminal_ref=terminal_ref,overrun_seconds=max(0.,time.monotonic()-inv['deadline']))
    if not accepted and terminal['status'] == 'COMPLETE':
        terminal['status'] = 'TIMEOUT' if time.monotonic() >= inv['deadline'] else 'ERROR'
    return terminal


def suffix_ref(path, deadline):
    """At most three bounded tail reads, not another unsupervised full census.

    Only after successful group quiescence. Caller I/O can be OS-delayed; an
    overrun rejects acceptance and is reported, never hidden as physical 300s.
    """
    require(time.monotonic() < deadline and path.name in ('controller.json','controller.log','terminal.json'),
            'bounded quiescent suffix')
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    h = hashlib.sha256(); count = 0
    with os.fdopen(fd,'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= REPORT_CAP,
                'bounded regular final suffix')
        for raw in iter(lambda:stream.read(65536),b''):
            require(time.monotonic() < deadline, 'final suffix read deadline')
            count += len(raw); require(count <= REPORT_CAP, 'suffix cap'); h.update(raw)
        after = os.fstat(stream.fileno())
        require(count == before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
                'quiescent final suffix unchanged')
    require(time.monotonic() < deadline, 'final suffix timely')
    return dict(bytes=count,sha256=h.hexdigest())


def append_suffix(storage, name, ref):
    storage['bytes'] += ref['bytes']-storage['files'].get(name,0)
    storage['files'][name] = ref['bytes']; storage['hashes'][name] = ref['sha256']
    require(storage['bytes'] == sum(storage['files'].values()) <= POLICY['total_bytes']
            and len(storage['files']) <= POLICY['members'], 'complete actual prefix plus suffix storage')
