"""Bounded CPU process-group execution, V1; not a general job/GPU reaper.

Workers must not daemonize or change session. Hold the leader unreaped until
signalling its group, so its PID cannot be reused for an unrelated group. This
confirms leader reaping and absence of LIVE group members, not orphan-zombie
reaping or CUDA-driver resource release. Observation errors fail closed.
"""
import os
from pathlib import Path
import signal
import subprocess
import time

from scoped_proof.io import ROOT


def observe(pgid):
    live, zombies, rss = [], [], 0
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            (zombies if fields[0] == 'Z' else live).append(int(path.name))
            for line in (path/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    rss += int(line.split()[1])*1024
        except (FileNotFoundError, ProcessLookupError):
            continue  # a vanished entry, not a permission/parse failure
    return {'live': sorted(live), 'zombies': sorted(zombies), 'rss': rss}


def parent_rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    raise ValueError('parent RSS unavailable')


def exited_unreaped(pid):
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def bounded_reap(process, deadline):
    # No fresh relative budget and no unbounded wait, including error paths.
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        return False
    try:
        process.wait(timeout=remaining)
        return True
    except subprocess.TimeoutExpired:
        return False


def execute(command, log, *, run_deadline, cleanup_deadline, env, rss_limit):
    start = time.monotonic()
    if run_deadline > cleanup_deadline:
        raise ValueError('cleanup must fit inside the phase deadline')
    p, peak, rc, error = None, 0, None, None
    status, cleanup = 'TIMEOUT', 'NO_PROCESS'
    before_cleanup = start
    final = {'live': [], 'zombies': [], 'rss': 0}
    descendant, exit_observed = False, None
    try:
        if start >= run_deadline:
            raise TimeoutError('no launch budget')
        with Path(log).open('xb') as stream:
            p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                now = time.monotonic()
                if now >= run_deadline:
                    status = 'TIMEOUT'
                    break
                group = observe(p.pid)
                peak = max(peak, group['rss']+parent_rss())
                if peak > rss_limit:
                    status = 'RESOURCE_LIMIT'
                    break
                exit_info = exited_unreaped(p.pid)
                if exit_info is not None:
                    group = observe(p.pid)  # include forks just before leader exit
                    exit_observed = time.monotonic()
                    descendant = bool(set(group['live'])-{p.pid})
                    status = ('TIMEOUT' if exit_observed >= run_deadline else
                              'COMPLETED' if exit_info.si_code == os.CLD_EXITED
                              and exit_info.si_status == 0 and not descendant else 'ERROR')
                    break
                time.sleep(min(.01, max(0., run_deadline-time.monotonic())))
    except TimeoutError as exc:
        status, error = 'TIMEOUT', repr(exc)
    except Exception as exc:
        status, error = 'ERROR', repr(exc)
    finally:
        before_cleanup = time.monotonic()
        if p is not None:
            cleanup = 'CLEANUP_INCOMPLETE'
            try:
                # The unreaped leader owns/reserves this pgid until this signal.
                os.killpg(p.pid, signal.SIGKILL)
                reaped = bounded_reap(p, cleanup_deadline)
                rc = p.returncode
                while time.monotonic() < cleanup_deadline:
                    final = observe(p.pid)
                    if reaped and not final['live']:
                        cleanup = 'LEADER_REAPED_NO_LIVE_GROUP'
                        break
                    time.sleep(min(.005, max(0., cleanup_deadline-time.monotonic())))
            except Exception as exc:
                error = repr(exc)
            if cleanup != 'LEADER_REAPED_NO_LIVE_GROUP':
                status = 'CLEANUP_INCOMPLETE'
    end = time.monotonic()
    if end >= cleanup_deadline and status == 'COMPLETED':
        status = 'TIMEOUT'
    return {'schema': 'OWNED_BOUNDED_CPU_V1', 'status': status, 'error': error,
            'pid': None if p is None else p.pid, 'returncode': rc,
            'seconds': end-start, 'execution_seconds': before_cleanup-start,
            'cleanup_seconds': end-before_cleanup, 'cleanup_status': cleanup,
            'remaining_group': final, 'descendant_on_leader_exit': descendant,
            'exit_observed_at': exit_observed,
            'sampled_peak_rss': peak, 'run_deadline': run_deadline,
            'cleanup_deadline': cleanup_deadline,
            'escaped_descendants_or_driver_cleanup': False}
