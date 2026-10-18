"""V2 bounded observation and owned-group supervision, not a global reaper.

Never signal a PID learned from /proc. Keep the leader unreaped until killpg,
then reap once; never signal that group number again after leader reaping.
Complete observation is not an atomic OS snapshot. Workers must not escape.
"""
import os
from pathlib import Path
import signal
import subprocess
import time

from scoped_proof.owned_bounded import exited_unreaped, bounded_reap
from scoped_source.hz_request_store import require
from scoped_source.native_resources import code, measure, tick

MAX_OBSERVATION_INTEGER = 2**63-1


def proc_read(path, cap, deadline):
    # proc pseudo-files commonly report size zero. Bound the actual read.
    tick(deadline)
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        with os.fdopen(fd,'rb',closefd=False) as stream: raw = stream.read(cap+1)
        require(len(raw) <= cap, 'proc record reservation')
        tick(deadline); return raw.decode('ascii')
    finally:
        os.close(fd)


def number(token):
    require(type(token) is str and 0 < len(token) <= 20
            and all('0' <= c <= '9' for c in token), 'bounded proc integer')
    return int(token)


def rss_value(text, *, zombie=False):
    found = None
    for line in text.splitlines():
        if line.startswith('VmRSS:'):
            fields = line.split()
            require(found is None and len(fields) == 3 and fields[2] == 'kB', 'proc RSS fields')
            found = number(fields[1])*1024
    require(found is not None or zombie, 'RSS unavailable')
    return 0 if found is None else found


def observe(pgid, binding, phase, resources, deadline):
    """Two observation slots cover old/new overlap; lists never default to proof."""
    slot = resources.active('observer')
    resources.active('observer_next')
    # Four bounded text/copy buffers plus both observation records are reserved
    # by the externally frozen slot formula; this is not a new PID quota.
    cap = min(65536,slot['encoded']//4)
    require(cap > 0 and type(pgid) is int and 0 < pgid <= MAX_OBSERVATION_INTEGER
            and type(binding) is str and len(binding) == 64
            and all(c in '0123456789abcdef' for c in binding)
            and phase in ('running','leader_exit','cleanup','between_actors'), 'observer preallocation')
    # Reserve the entire fixed schema, longest finite reason, maximum counters
    # and slack for a finite binary64 timestamp BEFORE retaining any PID.
    worst = dict(binding=binding,pgid=MAX_OBSERVATION_INTEGER,phase='between_actors',
        complete=False,reason='NOT_SCANNED',live=[],zombies=[],rss=MAX_OBSERVATION_INTEGER,
        entries=MAX_OBSERVATION_INTEGER,bytes=MAX_OBSERVATION_INTEGER,at=0.0)
    overhead = measure(worst,deadline,slot['encoded'],slot['depth'])['bytes']+64
    require(overhead <= slot['encoded'], 'complete and partial observation header reserve')
    del worst
    result = dict(binding=binding,pgid=pgid,phase=phase,complete=False,reason='NOT_SCANNED',
        live=[],zombies=[],rss=None,entries=0,bytes=0,at=time.monotonic())
    total = 0; remaining = slot['encoded']-overhead
    try:
        with os.scandir('/proc') as paths:
            for entry in paths:
                tick(deadline)
                require(result['entries'] < MAX_OBSERVATION_INTEGER,'observation counter representation')
                result['entries'] += 1
                name = entry.name
                if not name.isascii() or not name.isdecimal(): continue
                pid = number(name)
                try:
                    text = proc_read(entry.path+'/stat',cap,deadline)
                    require(result['bytes']+len(text) <= MAX_OBSERVATION_INTEGER,'observation bytes representation')
                    result['bytes'] += len(text)
                    fields = text.rsplit(')',1)[1].split()
                    require(len(fields) >= 3, 'proc stat structure')
                    if number(fields[2]) != pgid: continue
                    state = fields[0]; require(len(state) == 1, 'proc state')
                    # Charge representation before list append; full measure
                    # below independently accounts every field and escaping.
                    cost = len(name)+1
                    require(cost <= remaining, 'proc roster capacity')
                    remaining -= cost
                    status = proc_read(entry.path+'/status',cap,deadline)
                    require(result['bytes']+len(status) <= MAX_OBSERVATION_INTEGER,'observation bytes representation')
                    result['bytes'] += len(status)
                    value = rss_value(status,zombie=state == 'Z')
                    require(total+value <= MAX_OBSERVATION_INTEGER,'observation RSS representation')
                    (result['zombies'] if state == 'Z' else result['live']).append(pid)
                    total += value
                except (FileNotFoundError,ProcessLookupError):
                    continue  # vanished entries only, not permission/parse errors
        result['live'].sort(); result['zombies'].sort()
        result.update(complete=True,reason=None,rss=total,at=time.monotonic())
        tick(deadline)
        measure(result,deadline,slot['encoded'],slot['depth'])
    except Exception as exc:
        # The pre-reserved header+PID bound also covers this failure form even
        # when a second measure cannot run after a deadline. Never infer empty
        # membership or zero RSS from an incomplete prefix.
        result.update(complete=False,reason=code(exc),rss=None,at=time.monotonic())
    return result


def parent_rss(resources,deadline):
    slot = resources.active('observer_next')
    cap = min(65536,slot['encoded']//4)
    return rss_value(proc_read('/proc/self/status',cap,deadline))


def execute(command, log, *, cwd, run_deadline, cleanup_deadline, env, rss_limit,
            resources, binding):
    started = time.monotonic()
    require(started < run_deadline < cleanup_deadline <= resources.deadline, 'one owned deadline')
    p = None; final = None; sample = None; failed_observation = None
    peak = None; reaped = False; rc = None
    status = 'ERROR'; cleanup = 'NO_PROCESS'; error = None; exit_at = None
    cleanup_at = started
    def require_observation(value):
        nonlocal failed_observation
        if not value['complete']:
            failed_observation = dict(phase=value['phase'],reason=value['reason'],at=value['at'],
                entries=value['entries'],bytes=value['bytes'],live_prefix_count=len(value['live']),
                zombie_prefix_count=len(value['zombies']),complete=False)
            if value['reason'] == 'DEADLINE': raise TimeoutError('INCOMPLETE_OBSERVATION')
            raise ValueError('INCOMPLETE_OBSERVATION')
    with resources.hold('observer'), resources.hold('observer_next'):
        try:
            tick(run_deadline)
            with Path(log).open('xb') as stream:
                p = subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                while True:
                    tick(run_deadline)
                    sample = observe(p.pid,binding,'running',resources,run_deadline)
                    require_observation(sample)
                    now_rss = sample['rss']+parent_rss(resources,run_deadline)
                    peak = now_rss if peak is None else max(peak,now_rss)
                    if now_rss > rss_limit:
                        status = 'RESOURCE_LIMIT'; break
                    exit_info = exited_unreaped(p.pid)
                    if exit_info is not None:
                        sample = observe(p.pid,binding,'leader_exit',resources,run_deadline)
                        require_observation(sample)
                        exit_at = time.monotonic(); tick(run_deadline)
                        status = ('COMPLETED' if exit_info.si_code == os.CLD_EXITED
                            and exit_info.si_status == 0 and not any(v != p.pid for v in sample['live']) else 'ERROR')
                        break
                    time.sleep(min(.01,max(0.,run_deadline-time.monotonic())))
        except BaseException as exc:
            status = 'TIMEOUT' if isinstance(exc,TimeoutError) else 'ERROR'; error = code(exc)
        finally:
            # Do not retain a running roster alongside old+new cleanup rosters.
            # Only a fixed-size failed-observation summary survives in phase.
            sample = None
            cleanup_at = time.monotonic()
            if p is not None:
                cleanup = 'CLEANUP_INCOMPLETE'
                try:
                    # The still-unreaped leader pins the PGID until THIS signal.
                    os.killpg(p.pid,signal.SIGKILL)
                    reaped = bounded_reap(p,cleanup_deadline); rc = p.returncode
                    while time.monotonic() < cleanup_deadline:
                        final = observe(p.pid,binding,'cleanup',resources,cleanup_deadline)
                        if reaped and final['complete'] and not final['live']:
                            cleanup = 'LEADER_REAPED_NO_LIVE_GROUP'; break
                        if not final['complete']: break
                        time.sleep(min(.005,max(0.,cleanup_deadline-time.monotonic())))
                except BaseException as exc:
                    error = code(exc)
                if cleanup != 'LEADER_REAPED_NO_LIVE_GROUP': status = 'CLEANUP_INCOMPLETE'
    end = time.monotonic()
    if end >= cleanup_deadline and status == 'COMPLETED': status = 'TIMEOUT'
    return dict(schema='OWNED_RESOURCE_V2',status=status,error=error,pid=None if p is None else p.pid,
        returncode=rc,leader_reaped=reaped,cleanup_status=cleanup,remaining_group=final,
        seconds=end-started,execution_seconds=cleanup_at-started,cleanup_seconds=end-cleanup_at,
        sampled_peak_rss=peak,failed_observation=failed_observation,exit_observed_at=exit_at,
        cleanup_observed_at=end,run_deadline=run_deadline,
        cleanup_deadline=cleanup_deadline,escaped_descendants_or_driver_cleanup=False)


def complete(run,deadline):
    final = run.get('remaining_group')
    return (run.get('status') == 'COMPLETED' and run.get('leader_reaped') is True
        and run.get('cleanup_status') == 'LEADER_REAPED_NO_LIVE_GROUP'
        and type(final) is dict and final.get('complete') is True and final.get('live') == []
        and final.get('at',deadline) < deadline and run.get('cleanup_observed_at',deadline) < deadline)
