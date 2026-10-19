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
from scoped_proof.pidfd_compat import Pidfd

LOG_CAP=16384
MEMBERS=512
ERROR_TEXT=256


def gate(deadline):
    if time.monotonic()>=deadline: raise TimeoutError('owned session deadline')


class _ProcReader:
    """Read-only production source; deterministic controls use a private double."""
    def pids(self):
        for p in Path('/proc').iterdir():
            if p.name.isdecimal(): yield int(p.name)

    def read(self,pid,name,cap):
        with (Path('/proc')/str(pid)/name).open('rb') as f:
            return f.read(cap+1)


def _identity(pid,reader):
    raw=reader.read(pid,'stat',8192)
    if len(raw)>8192: raise ValueError('bounded process identity')
    if int(raw.split(b' (',1)[0])!=pid: raise ValueError('process PID mismatch')
    fields=raw.rsplit(b')',1)[1].split()
    state=fields[0].decode('ascii'); sid=int(fields[3]); start=int(fields[19])
    if len(state)!=1 or sid<0 or start<0: raise ValueError('invalid process identity')
    return state,sid,start


def process(pid):
    state,sid,_=_identity(pid,_ProcReader())
    return state,sid


def _error_detail(exc,phase,pid):
    if isinstance(exc,_ObservationFailure): return dict(exc.detail)
    # Two bounded first-error records, not unbounded tracebacks or proc contents.
    message=''.join(c if 32<=ord(c)<127 else '?' for c in str(exc)[:ERROR_TEXT])
    return dict(kind=type(exc).__name__[:64],message=message,phase=phase[:64],pid=pid)


class _ObservationFailure(RuntimeError):
    def __init__(self,exc,phase,pid):
        self.detail=_error_detail(exc,phase,pid)
        self.timed_out=isinstance(exc,TimeoutError)
        super().__init__(self.detail['message'])


def _observe(sid,deadline,reader):
    live=[]; zombies=[]; rss=0; pid=None; phase='scan_begin'
    try:
        gate(deadline)
        for pid in reader.pids():
            phase='initial_stat'; gate(deadline)
            try: first=_identity(pid,reader)
            except (FileNotFoundError,ProcessLookupError): continue
            if first[1]!=sid: continue
            phase='status'; gate(deadline)
            try: raw=reader.read(pid,'status',65536)
            except (FileNotFoundError,ProcessLookupError): raw=b''
            if len(raw)>65536: raise ValueError('bounded process status')
            values=[line.split() for line in raw.splitlines() if line.startswith(b'VmRSS:')]
            if len(values)>1: raise ValueError('duplicate RSS')
            value=None
            if values:
                if len(values[0])!=3 or values[0][2]!=b'kB': raise ValueError('invalid RSS unit')
                value=int(values[0][1])*1024
                if value<0: raise ValueError('negative RSS')
            # One bounded reread also prevents combining two incarnations when
            # RSS exists. Missing status is NOT itself evidence of process exit.
            phase='confirm_stat'; gate(deadline)
            try: last=_identity(pid,reader)
            except (FileNotFoundError,ProcessLookupError): continue
            if first[1:]!=last[1:]: raise ValueError('process identity changed during observation')
            if last[0] in ('Z','X','x'):
                zombies.append(pid)  # terminal/non-running; not a claim of reaping
            else:
                if value is None: raise ValueError('live process RSS unavailable')
                live.append(pid); rss+=value
            if len(live)+len(zombies)>MEMBERS: raise ValueError('bounded session roster')
        phase='scan_end'; pid=None; gate(deadline)
        return dict(live=sorted(live),zombies=sorted(zombies),rss=rss)
    except Exception as exc:
        raise _ObservationFailure(exc,phase,pid) from exc


def observe(sid,deadline):
    return _observe(sid,deadline,_ProcReader())


class _SessionEvidence:
    def __init__(self):
        self.remaining=None
        self.execution_error=None; self.cleanup_error=None

    def scan(self,sid,deadline,observer):
        self.remaining=None  # A failed/unfinished scan cannot retain an old roster.
        self.remaining=observer(sid,deadline)
        return self.remaining

    def failed(self,stage,exc,phase,pid):
        field=stage+'_error'
        if getattr(self,field) is None:
            detail=_error_detail(exc,phase,pid)
            detail['observation_known']=self.remaining is not None
            setattr(self,field,detail)

    def fields(self):
        first=self.execution_error or self.cleanup_error
        error=None if first is None else (
            'SESSION_DEADLINE' if self.execution_error and first['kind']=='TimeoutError' else
            ('' if self.execution_error else 'SESSION_CLEANUP_')+first['kind'])
        return dict(error=error,execution_error=self.execution_error,cleanup_error=self.cleanup_error,
                    remaining_session=self.remaining,remaining_session_known=self.remaining is not None,
                    observation_contract='CONFIRMED_IDENTITY_V2')


def kill_owned(pid,sid,deadline, *, pidfd=None):
    """Open first, then recheck SID. Never signal a bare discovered PID/PGID."""
    gate(deadline); fd=None
    pidfd=Pidfd() if pidfd is None else pidfd
    try:
        fd=pidfd.pidfd_open(pid,0)
        _,current=process(pid)
        if current!=sid: return False
        gate(deadline); pidfd.pidfd_send_signal(fd,int(signal.SIGKILL),None,0)
        return True
    except (FileNotFoundError,ProcessLookupError): return False
    finally:
        if fd is not None: os.close(fd)


def execute(argv, *, run_deadline, cleanup_deadline, env, rss_limit):
    """One spawn; caller has already authorized source, code, limits and times."""
    return _execute(argv,run_deadline=run_deadline,cleanup_deadline=cleanup_deadline,
                    env=env,rss_limit=rss_limit,_observer=observe)


def _execute(argv, *, run_deadline, cleanup_deadline, env, rss_limit, _observer):
    """Private deterministic fault seam. Public execute always binds observe."""
    if run_deadline>cleanup_deadline: raise ValueError('shared session cleanup budget')
    started=time.monotonic()
    gate(run_deadline)
    pidfd=Pidfd(); pidfd.probe()  # Before Popen; unavailability cannot strand a child.
    gate(run_deadline)
    p=None; selector=selectors.DefaultSelector(); chunks=[]; size=0
    peak=0; status='ERROR'; exit_at=None; rc=None; cleanup='NO_PROCESS'; eof=False
    evidence=_SessionEvidence(); before_cleanup=started; signals=0; leader_fd=None; phase='spawn'
    try:
        gate(run_deadline)
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,start_new_session=True)
        leader_fd=pidfd.pidfd_open(p.pid,0)
        os.set_blocking(p.stdout.fileno(),False); selector.register(p.stdout,selectors.EVENT_READ)
        while True:
            phase='execution_observe'
            gate(run_deadline)
            current=evidence.scan(p.pid,run_deadline,_observer)
            peak=max(peak,current['rss']+parent_rss())
            if peak>rss_limit: status='RESOURCE_LIMIT'; break
            phase='execution_output'
            for key,_ in selector.select(min(.01,max(0.,run_deadline-time.monotonic()))):
                raw=os.read(key.fileobj.fileno(),min(4096,LOG_CAP-size+1))
                if not raw: selector.unregister(key.fileobj); eof=True; continue
                size+=len(raw)
                if size>LOG_CAP: raise ValueError('batch output cap, not truncated success')
                chunks.append(raw)
            info=exited_unreaped(p.pid)
            if info is not None and eof:
                phase='execution_final_observe'
                final=evidence.scan(p.pid,run_deadline,_observer); exit_at=time.monotonic()
                status=('COMPLETED' if info.si_code==os.CLD_EXITED and info.si_status==0
                        and not final['live'] else 'ERROR')
                break
    except BaseException as exc:
        evidence.failed('execution',exc,phase,None if p is None else p.pid)
        status='TIMEOUT' if isinstance(exc,TimeoutError) or getattr(exc,'timed_out',False) else 'ERROR'
    finally:
        before_cleanup=time.monotonic(); evidence.remaining=None
        try: selector.close()
        except BaseException as exc: evidence.failed('cleanup',exc,'selector_close',None if p is None else p.pid)
        if p is not None:
            cleanup='CLEANUP_INCOMPLETE'
            phase='cleanup_leader_signal'
            try:
                # Direct child identity remains reserved by waitid(WNOWAIT).
                # Also kill a pre-exec leader whose setsid has not yet happened.
                gate(cleanup_deadline)
                if leader_fd is None: leader_fd=pidfd.pidfd_open(p.pid,0)
                try:
                    pidfd.pidfd_send_signal(leader_fd,int(signal.SIGKILL),None,0); signals+=1
                except ProcessLookupError: pass
                # Leader stays unreaped throughout scans/signals, including when
                # it exited normally. Its SID cannot be given to another job.
                while time.monotonic()<cleanup_deadline:
                    phase='cleanup_observe'
                    final=evidence.scan(p.pid,cleanup_deadline,_observer)
                    if not final['live']:
                        phase='cleanup_reap'
                        if bounded_reap(p,cleanup_deadline):
                            rc=p.returncode; cleanup='LEADER_REAPED_NO_LIVE_SESSION'
                        break
                    for pid in final['live']:
                        phase='cleanup_member_signal'
                        signals+=kill_owned(pid,p.pid,cleanup_deadline,pidfd=pidfd)
                    time.sleep(min(.005,max(0.,cleanup_deadline-time.monotonic())))
            except BaseException as exc:
                evidence.failed('cleanup',exc,phase,p.pid if phase!='cleanup_member_signal' else pid)
            try:
                if p.stdout is not None: p.stdout.close()
            except BaseException as exc: evidence.failed('cleanup',exc,'stdout_close',p.pid)
        try:
            if leader_fd is not None: os.close(leader_fd)
        except BaseException as exc: evidence.failed('cleanup',exc,'leader_fd_close',None if p is None else p.pid)
        if evidence.cleanup_error is not None: cleanup='CLEANUP_INCOMPLETE'
        if p is not None and cleanup!='LEADER_REAPED_NO_LIVE_SESSION': status='CLEANUP_INCOMPLETE'
    end=time.monotonic()
    if status=='COMPLETED' and end>=cleanup_deadline: status='TIMEOUT'
    import base64
    return dict(schema='OWNED_ENDPOINT_SESSION_V1',status=status,**evidence.fields(),
        session=None if p is None else p.pid,returncode=rc,started_at=started,finished_at=end,
        execution_seconds=before_cleanup-started,cleanup_seconds=end-before_cleanup,
        cleanup_status=cleanup,exit_observed_at=exit_at,
        run_deadline=run_deadline,cleanup_deadline=cleanup_deadline,kill_signals=signals,
        sampled_session_plus_caller_rss=peak,log_bytes=size,log_complete=size<=LOG_CAP,
        log_base64=base64.b64encode(b''.join(chunks)).decode(),
        escaped_sessions_checked=False,zombie_reaping_complete=False,pidfd_backend=pidfd.backend)
