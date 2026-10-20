"""Finite Linux session observation for the endpoint batch, not a general reaper.

The unreaped direct-child leader reserves the SID. Trusted workers may create
process groups but may not leave this session. pidfds prevent PID reuse signals.
No GPU/driver cleanup or hard aggregate RSS guarantee is claimed.
"""
import os
from pathlib import Path
import selectors
import select
import signal
import subprocess
import time

from scoped_proof.io import ROOT
from scoped_proof.owned_bounded import exited_unreaped, bounded_reap, parent_rss
from scoped_proof.pidfd_compat import Pidfd

LOG_CAP=16384
MEMBERS=512
ERROR_TEXT=256
TERMINAL_POLL_MS=10


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
    if len(state)!=1 or not 0<=sid<2**63 or not 0<=start<2**63:
        raise ValueError('invalid process identity')
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
    def __init__(self,exc,phase,pid,observation=None):
        self.detail=_error_detail(exc,phase,pid)
        if observation is not None: self.detail['observation']=dict(observation)
        self.timed_out=isinstance(exc,TimeoutError)
        super().__init__(self.detail['message'])


def _identity_record(value):
    return dict(state=value[0],sid=value[1],start_ticks=value[2])


def _poll_terminal(fd,milliseconds):
    poll=select.poll()
    poll.register(fd,select.POLLIN)
    return poll.poll(milliseconds)


def _terminal_without_rss(pid,identity,deadline,reader,context,*,pidfd=None,poll=None):
    """One identity-bound terminal query; never reconstructs a missing RSS."""
    gate(deadline)
    api=Pidfd() if pidfd is None else pidfd
    poll=_poll_terminal if poll is None else poll
    fd=None
    try:
        fd=api.pidfd_open(pid,0)
        gate(deadline)
        bound=_identity(pid,reader)
        if bound[1:]!=identity[1:]:
            raise ValueError('terminal process identity changed')
        gate(deadline)
        wait_ms=min(TERMINAL_POLL_MS,max(0,int((deadline-time.monotonic())*1000)))
        events=poll(fd,wait_ms)
        gate(deadline)
        if (len(events)!=1 or events[0][0]!=fd
                or not events[0][1]&select.POLLIN
                or events[0][1]&~(select.POLLIN|select.POLLHUP)):
            raise ValueError('live process RSS unavailable: terminal unconfirmed')
        return dict(pid=pid,identity=_identity_record(bound),
                    missing_observation=dict(context),rss=None,
                    evidence='IDENTITY_BOUND_PIDFD_POLLIN',pidfd_backend=api.backend,
                    poll_events=events[0][1],poll_limit_ms=wait_ms,confirmed_at=time.monotonic())
    finally:
        if fd is not None: os.close(fd)


def _observe(sid,deadline,reader,*,terminal_confirmation=False,_pidfd=None,_poll=None):
    if type(terminal_confirmation) is not bool: raise ValueError('terminal confirmation flag')
    live=[]; zombies=[]; terminals=[]; rss=0; pid=None; phase='scan_begin'
    context=None
    try:
        gate(deadline)
        for pid in reader.pids():
            context=dict(first_identity=None,last_identity=None,status_read='NOT_ATTEMPTED',
                         status_bytes=None,rss_present=None)
            phase='initial_stat'; gate(deadline)
            try: first=_identity(pid,reader)
            except (FileNotFoundError,ProcessLookupError): continue
            context['first_identity']=_identity_record(first)
            if first[1]!=sid: continue
            phase='status'; gate(deadline)
            context['status_read']='ERROR'
            try:
                raw=reader.read(pid,'status',65536); context['status_read']='READ'
            except (FileNotFoundError,ProcessLookupError):
                raw=b''; context['status_read']='DISAPPEARED'
            context['status_bytes']=len(raw)
            if len(raw)>65536: raise ValueError('bounded process status')
            values=[line.split() for line in raw.splitlines() if line.startswith(b'VmRSS:')]
            context['rss_present']=bool(values)
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
            context['last_identity']=_identity_record(last)
            if first[1:]!=last[1:]: raise ValueError('process identity changed during observation')
            if last[0] in ('Z','X','x'):
                zombies.append(pid)  # terminal/non-running; not a claim of reaping
            else:
                if value is None:
                    if not terminal_confirmation: raise ValueError('live process RSS unavailable')
                    phase='terminal_pidfd_confirmation'
                    terminals.append(_terminal_without_rss(pid,last,deadline,reader,context,
                                                          pidfd=_pidfd,poll=_poll))
                else:
                    live.append(pid); rss+=value
            if len(live)+len(zombies)+len(terminals)>MEMBERS: raise ValueError('bounded session roster')
        phase='scan_end'; pid=None; context=None; gate(deadline)
        result=dict(live=sorted(live),zombies=sorted(zombies),rss=rss)
        if terminal_confirmation:
            result.update(terminal_without_rss=terminals,rss_scope='OBSERVED_LIVE_ONLY')
        return result
    except Exception as exc:
        failure=_ObservationFailure(exc,phase,pid,context)
        if terminals: failure.detail['terminal_without_rss']=terminals
        raise failure from exc


def observe(sid,deadline,*,terminal_confirmation=False):
    return _observe(sid,deadline,_ProcReader(),terminal_confirmation=terminal_confirmation)


def _observe_cleanup(sid,deadline,reader):
    """Identity roster for cleanup only. Never supplies memory evidence."""
    live=[]; zombies=[]; identities=[]; pid=None; phase='cleanup_scan_begin'; context=None
    try:
        gate(deadline)
        for pid in reader.pids():
            context=dict(first_identity=None,last_identity=None)
            phase='cleanup_initial_stat'; gate(deadline)
            try: first=_identity(pid,reader)
            except (FileNotFoundError,ProcessLookupError): continue
            context['first_identity']=_identity_record(first)
            if first[1]!=sid: continue
            phase='cleanup_confirm_stat'; gate(deadline)
            try: last=_identity(pid,reader)
            except (FileNotFoundError,ProcessLookupError): continue
            context['last_identity']=_identity_record(last)
            if first[1:]!=last[1:]: raise ValueError('cleanup process identity changed')
            if last[0] in ('Z','X','x'): zombies.append(pid)
            else:
                live.append(pid)
                identities.append(dict(pid=pid,sid=sid,start_ticks=last[2]))
            if len(live)+len(zombies)>MEMBERS: raise ValueError('bounded session roster')
        phase='cleanup_scan_end'; pid=None; context=None; gate(deadline)
        return dict(live=sorted(live),zombies=sorted(zombies),rss=None,rss_known=False,
                    observation_scope='IDENTITY_ONLY_CLEANUP',
                    identities=sorted(identities,key=lambda v:v['pid']))
    except Exception as exc:
        raise _ObservationFailure(exc,phase,pid,context) from exc


def observe_cleanup(sid,deadline):
    return _observe_cleanup(sid,deadline,_ProcReader())


class _SessionEvidence:
    def __init__(self,terminal_confirmation=False):
        self.remaining=None
        self.execution_error=None; self.cleanup_error=None
        self.terminal_confirmation=terminal_confirmation
        self.terminals={}

    def retain_terminals(self,records):
        for record in records:
            key=(record['pid'],record['identity']['sid'],record['identity']['start_ticks'])
            if key not in self.terminals:
                if len(self.terminals)>=MEMBERS: raise ValueError('bounded terminal evidence')
                self.terminals[key]=record

    def scan(self,sid,deadline,observer):
        self.remaining=None  # A failed/unfinished scan cannot retain an old roster.
        try:
            current=observer(sid,deadline)
        except _ObservationFailure as exc:
            try: self.retain_terminals(exc.detail.get('terminal_without_rss',[]))
            except Exception as retention:
                # Evidence capacity failure cannot erase the original scan or
                # deadline error. This scan remains rejected in either case.
                exc.detail['terminal_retention_error']=_error_detail(retention,'terminal_retention',None)
            raise
        self.retain_terminals(current.get('terminal_without_rss',[]))
        self.remaining=current
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
        result=dict(error=error,execution_error=self.execution_error,cleanup_error=self.cleanup_error,
                    remaining_session=self.remaining,remaining_session_known=self.remaining is not None,
                    observation_contract=('CONFIRMED_IDENTITY_V3' if self.terminal_confirmation
                                          else 'CONFIRMED_IDENTITY_V2'),cleanup_contract='IDENTITY_ONLY_V1')
        if self.terminal_confirmation: result['terminal_without_rss']=list(self.terminals.values())
        return result


def kill_owned(pid,sid,deadline, *, pidfd=None,expected_start=None):
    return _kill_owned(pid,sid,deadline,pidfd=pidfd,expected_start=expected_start,reader=_ProcReader())


def _kill_owned(pid,sid,deadline, *, pidfd,expected_start,reader):
    """Open first, then recheck SID. Never signal a bare discovered PID/PGID."""
    gate(deadline); fd=None
    pidfd=Pidfd() if pidfd is None else pidfd
    try:
        fd=pidfd.pidfd_open(pid,0)
        _,current,start=_identity(pid,reader)
        if current!=sid or (expected_start is not None and start!=expected_start):
            if expected_start is not None: raise ValueError('signal process identity changed')
            return False
        gate(deadline); pidfd.pidfd_send_signal(fd,int(signal.SIGKILL),None,0)
        return True
    except (FileNotFoundError,ProcessLookupError): return False
    finally:
        if fd is not None: os.close(fd)


def execute(argv, *, run_deadline, cleanup_deadline, env, rss_limit,terminal_confirmation=False):
    """One spawn; caller has already authorized source, code, limits and times."""
    if type(terminal_confirmation) is not bool: raise ValueError('terminal confirmation flag')
    observer=observe if not terminal_confirmation else (
        lambda sid,deadline:observe(sid,deadline,terminal_confirmation=True))
    return _execute(argv,run_deadline=run_deadline,cleanup_deadline=cleanup_deadline,
                    env=env,rss_limit=rss_limit,_observer=observer,_cleanup_observer=observe_cleanup,
                    _terminal_confirmation=terminal_confirmation)


def _execute(argv, *, run_deadline, cleanup_deadline, env, rss_limit, _observer,
             _cleanup_observer=observe_cleanup,_terminal_confirmation=False):
    """Private deterministic fault seam. Public execute always binds observe."""
    if run_deadline>cleanup_deadline: raise ValueError('shared session cleanup budget')
    started=time.monotonic()
    gate(run_deadline)
    pidfd=Pidfd(); pidfd.probe()  # Before Popen; unavailability cannot strand a child.
    gate(run_deadline)
    p=None; selector=selectors.DefaultSelector(); chunks=[]; size=0
    peak=0; status='ERROR'; exit_at=None; rc=None; cleanup='NO_PROCESS'; eof=False
    evidence=_SessionEvidence(_terminal_confirmation)
    before_cleanup=started; signals=0; leader_fd=None; phase='spawn'
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
                    final=evidence.scan(p.pid,cleanup_deadline,_cleanup_observer)
                    if not final['live']:
                        phase='cleanup_reap'
                        if bounded_reap(p,cleanup_deadline):
                            rc=p.returncode; cleanup='LEADER_REAPED_NO_LIVE_SESSION'
                        break
                    for member in final['identities']:
                        pid=member['pid']
                        phase='cleanup_member_signal'
                        signals+=kill_owned(pid,p.pid,cleanup_deadline,pidfd=pidfd,
                                            expected_start=member['start_ticks'])
                    evidence.remaining=None  # Signals invalidate the pre-signal roster until a fresh scan.
                    time.sleep(min(.005,max(0.,cleanup_deadline-time.monotonic())))
            except BaseException as exc:
                evidence.remaining=None  # A signal/reap failure can invalidate the last identity roster.
                evidence.failed('cleanup',exc,phase,p.pid if phase!='cleanup_member_signal' else pid)
            try:
                if p.stdout is not None: p.stdout.close()
            except BaseException as exc: evidence.failed('cleanup',exc,'stdout_close',p.pid)
        try:
            if leader_fd is not None: os.close(leader_fd)
        except BaseException as exc: evidence.failed('cleanup',exc,'leader_fd_close',None if p is None else p.pid)
        if evidence.cleanup_error is not None: cleanup='CLEANUP_INCOMPLETE'
        if p is not None and cleanup!='LEADER_REAPED_NO_LIVE_SESSION':
            evidence.remaining=None
            status='CLEANUP_INCOMPLETE'
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
