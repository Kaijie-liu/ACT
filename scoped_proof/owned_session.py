"""Finite Linux session observation for the endpoint batch, not a general reaper.

The unreaped direct-child leader reserves the SID. Trusted workers may create
process groups but may not leave this session. pidfds prevent PID reuse signals.
No GPU/driver cleanup or hard aggregate RSS guarantee is claimed.
"""
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from scoped_proof.io import ROOT
from scoped_proof.owned_bounded import exited_unreaped, bounded_reap, parent_rss

LOG_CAP=16384
MEMBERS=512


def gate(deadline):
    if time.monotonic()>=deadline: raise TimeoutError('owned session deadline')


def process(pid):
    p=Path('/proc')/str(pid)
    with (p/'stat').open('rb') as f: raw=f.read(8193)
    if len(raw)>8192: raise ValueError('bounded process identity')
    fields=raw.rsplit(b')',1)[1].split()
    return fields[0].decode(),int(fields[3])


def observe(sid,deadline):
    live=[]; zombies=[]; rss=0
    for p in Path('/proc').iterdir():
        gate(deadline)
        if not p.name.isdecimal(): continue
        try:
            state,current=process(int(p.name))
            if current!=sid: continue
            (zombies if state=='Z' else live).append(int(p.name))
            if len(live)+len(zombies)>MEMBERS: raise ValueError('bounded session roster')
            with (p/'status').open('rb') as f: raw=f.read(65537)
            if len(raw)>65536: raise ValueError('bounded process status')
            value=next((int(line.split()[1])*1024 for line in raw.splitlines()
                        if line.startswith(b'VmRSS:')),None)
            if value is None and state!='Z': raise ValueError('live process RSS unavailable')
            rss+=value or 0
        except (FileNotFoundError,ProcessLookupError): continue
    gate(deadline)
    return dict(live=sorted(live),zombies=sorted(zombies),rss=rss)


def kill_owned(pid,sid,deadline):
    """Open first, then recheck SID. Never signal a bare discovered PID/PGID."""
    gate(deadline); fd=None
    try:
        fd=os.pidfd_open(pid,0)
        _,current=process(pid)
        if current!=sid: return False
        gate(deadline); signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
        return True
    except (FileNotFoundError,ProcessLookupError): return False
    finally:
        if fd is not None: os.close(fd)


def execute(argv, *, run_deadline, cleanup_deadline, env, rss_limit):
    """One spawn; caller has already authorized source, code, limits and times."""
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support required; no unsafe fallback')
    if run_deadline>cleanup_deadline: raise ValueError('shared session cleanup budget')
    started=time.monotonic(); p=None; selector=selectors.DefaultSelector(); chunks=[]; size=0
    peak=0; status='ERROR'; error=None; exit_at=None; rc=None; cleanup='NO_PROCESS'; eof=False
    final=dict(live=[],zombies=[],rss=0); before_cleanup=started; signals=0; leader_fd=None
    try:
        gate(run_deadline)
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,start_new_session=True)
        leader_fd=os.pidfd_open(p.pid,0)
        os.set_blocking(p.stdout.fileno(),False); selector.register(p.stdout,selectors.EVENT_READ)
        while True:
            gate(run_deadline)
            current=observe(p.pid,run_deadline); peak=max(peak,current['rss']+parent_rss())
            if peak>rss_limit: status='RESOURCE_LIMIT'; break
            for key,_ in selector.select(min(.01,max(0.,run_deadline-time.monotonic()))):
                raw=os.read(key.fileobj.fileno(),min(4096,LOG_CAP-size+1))
                if not raw: selector.unregister(key.fileobj); eof=True; continue
                size+=len(raw)
                if size>LOG_CAP: raise ValueError('batch output cap, not truncated success')
                chunks.append(raw)
            info=exited_unreaped(p.pid)
            if info is not None and eof:
                final=observe(p.pid,run_deadline); exit_at=time.monotonic()
                status=('COMPLETED' if info.si_code==os.CLD_EXITED and info.si_status==0
                        and not final['live'] else 'ERROR')
                break
    except TimeoutError:
        status='TIMEOUT'; error='SESSION_DEADLINE'
    except BaseException as exc:
        status='ERROR'; error=type(exc).__name__
    finally:
        before_cleanup=time.monotonic(); selector.close()
        if p is not None:
            cleanup='CLEANUP_INCOMPLETE'
            try:
                # Direct child identity remains reserved by waitid(WNOWAIT).
                # Also kill a pre-exec leader whose setsid has not yet happened.
                gate(cleanup_deadline)
                if leader_fd is None: leader_fd=os.pidfd_open(p.pid,0)
                try:
                    signal.pidfd_send_signal(leader_fd,signal.SIGKILL,None,0); signals+=1
                except ProcessLookupError: pass
                # Leader stays unreaped throughout scans/signals, including when
                # it exited normally. Its SID cannot be given to another job.
                while time.monotonic()<cleanup_deadline:
                    final=observe(p.pid,cleanup_deadline)
                    if not final['live']:
                        if bounded_reap(p,cleanup_deadline):
                            rc=p.returncode; cleanup='LEADER_REAPED_NO_LIVE_SESSION'
                        break
                    for pid in final['live']:
                        signals+=kill_owned(pid,p.pid,cleanup_deadline)
                    time.sleep(min(.005,max(0.,cleanup_deadline-time.monotonic())))
            except BaseException as exc:
                error='SESSION_CLEANUP_'+type(exc).__name__
            if p.stdout is not None: p.stdout.close()
            if cleanup!='LEADER_REAPED_NO_LIVE_SESSION': status='CLEANUP_INCOMPLETE'
        if leader_fd is not None: os.close(leader_fd)
    end=time.monotonic()
    if status=='COMPLETED' and end>=cleanup_deadline: status='TIMEOUT'
    import base64
    return dict(schema='OWNED_ENDPOINT_SESSION_V1',status=status,error=error,
        session=None if p is None else p.pid,returncode=rc,started_at=started,finished_at=end,
        execution_seconds=before_cleanup-started,cleanup_seconds=end-before_cleanup,
        cleanup_status=cleanup,remaining_session=final,exit_observed_at=exit_at,
        run_deadline=run_deadline,cleanup_deadline=cleanup_deadline,kill_signals=signals,
        sampled_session_plus_caller_rss=peak,log_bytes=size,log_complete=size<=LOG_CAP,
        log_base64=base64.b64encode(b''.join(chunks)).decode(),
        escaped_sessions_checked=False,zombie_reaping_complete=False)
