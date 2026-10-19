"""One frozen compatibility-only batch; no algorithm, model or GPU execution.

This provisioned stdlib measurement root uses the existing owned-session runner,
not another supervisor. Each test child has a short self-expiry as a second
safety net. No external PID is ever signalled. Unexpected failure stops suffix.
"""
import time
PROVISION_ENTRY = time.monotonic()
import hashlib
import json
import os
import resource
import signal
import sys

REPO = '/data1/Kane/MOE/ACT'
PYTHON = '/data1/Kane/miniconda3/envs/act-py312/bin/python'
FREEZE = REPO+'/docs/hz_pidfd_control_freeze_20261019.json'
RUN = '/data1/Kane/MOE/results/pidfd_compat_20261019_r1'
IDS = ('self_capability', 'invalid_and_closed_fd', 'local_abi_errors',
       'normal_and_error_exit', 'owned_signal', 'foreign_session_refusal',
       'deadline_and_descendant_cleanup', 'descriptor_balance')
CAPS = dict(seconds=300, as_bytes=8*2**30, rss_bytes=2*2**30,
            artifact_bytes=128*2**20, work=8000000)


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    # Provisioning is explicit; no project imports/native probes before t0.
    for kind, cap in ((resource.RLIMIT_AS, CAPS['as_bytes']),
                      (resource.RLIMIT_FSIZE, CAPS['artifact_bytes'])):
        limits = resource.getrlimit(kind)
        require(all(v == resource.RLIM_INFINITY or v >= cap for v in limits),
                'cannot raise tighter inherited limit')
        resource.setrlimit(kind, (cap, cap))
    started = time.monotonic()
    deadline = started+300

    def hard_stop(_sig, _frame):
        # Only this process exits. No fallback signal to an unbound PID/group.
        os.write(2, b'PIDFD_CONTROL_HARD_DEADLINE_NO_ACCEPTANCE\n')
        os._exit(124)

    signal.signal(signal.SIGALRM, hard_stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    rows = []; records = []; tracked = []; events = []
    result = dict(schema='PIDFD_FINITE_CONTROL_V1', status='FAIL', started=started,
                  full_cold_start_seconds=None, provisioning_seconds=started-PROVISION_ENTRY,
                  root_pid=os.getpid(), caps=CAPS, cases=rows, sessions=records,
                  child_events=events, algorithm_executed=False, model_executed=False,
                  gpu_executed=False, endpoint_batch_authorized=False)
    directory_created = False
    try:
        require(len(sys.argv) == 3 and sys.argv[1] == '--execute-approved-once', 'explicit opt in')
        expected = sys.argv[2]
        require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'freeze hash')
        require(sys.executable == PYTHON, 'frozen interpreter')
        with open(FREEZE, 'rb') as stream:
            raw = stream.read(131073)
        require(len(raw) <= 131072 and hashlib.sha256(raw).hexdigest() == expected, 'freeze identity')
        freeze = json.loads(raw)
        require(freeze['schema'] == 'PIDFD_FINITE_FREEZE_V1' and freeze['caps'] == CAPS
                and freeze['cases'] == list(IDS) and freeze['attempts'] == 1
                and freeze['result_directory'] == RUN, 'closed finite roster')
        result['freeze_sha256'] = expected
        for path, value in freeze['files'].items():
            require(sha(path) == value, 'frozen file: '+path)
        require(time.monotonic() < started+60, 'bounded preflight')
        from pathlib import Path
        import shutil
        import subprocess
        import ctypes
        import errno
        import base64
        sys.path.append(REPO)
        from scoped_proof.pidfd_compat import Pidfd, _Libc
        from scoped_proof import owned_session as owned
        from scoped_proof.io import save
        require(not Path(RUN).exists(), 'no retry or overwrite')
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        free_disk = shutil.disk_usage(Path(RUN).parent).free
        require(available >= 4*2**30 and free_disk >= CAPS['artifact_bytes'], 'fresh resource admission')
        result['admission'] = dict(available_bytes=available, disk_free_bytes=free_disk,
                                   seconds=time.monotonic()-started)
        Path(RUN).mkdir()  # Exclusive one-shot claim, including failed attempts.
        directory_created = True
        api = Pidfd()
        initial_fds = len(os.listdir('/proc/self/fd'))
        initial_children = Path('/proc/self/task/'+str(os.getpid())+'/children').read_text().split()
        result['backend'] = api.backend
        result['python_native_api'] = dict(open=hasattr(os, 'pidfd_open'),
                                          send=hasattr(signal, 'pidfd_send_signal'))
        env = dict(PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        peak = owned.parent_rss()

        def tick():
            nonlocal peak
            require(time.monotonic() < started+240, 'control work deadline; reserve reception/cleanup')
            rss = owned.parent_rss()
            for p in tracked:
                # Direct Popen children remain our objects until bounded wait.
                if p.returncode is None:
                    rss += owned.observe(p.pid, started+245)['rss']
            peak = max(peak, rss)
            require(peak <= CAPS['rss_bytes'], 'sampled RSS stop')

        def fd_count():
            return len(os.listdir('/proc/self/fd'))

        def expect(kind, action, code=None):
            try:
                action()
            except kind as exc:
                if code is not None:
                    require(exc.errno == code, 'exact errno')
                return
            raise AssertionError('expected '+kind.__name__)

        def spawn(seconds=2):
            tick(); api.probe(); tick()
            code = 'import signal,time;signal.alarm(6);time.sleep('+str(seconds)+')'
            p = subprocess.Popen([PYTHON, '-I', '-S', '-B', '-c', code], cwd=REPO,
                                 env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
            tracked.append(p)
            events.append(dict(event='SPAWN', pid=p.pid, sid=p.pid, at=time.monotonic()))
            return p

        def wait(p):
            # No signal fallback: bounded wait on self-expiring, owned fixtures.
            p.wait(timeout=min(8, max(.001, started+250-time.monotonic())))
            events.append(dict(event='REAP', pid=p.pid, returncode=p.returncode, at=time.monotonic()))
            return p.returncode

        def session(code, seconds):
            nonlocal peak
            tick()
            end = min(started+230, time.monotonic()+seconds)
            receipt = owned.execute([PYTHON, '-I', '-S', '-B', '-c', code],
                run_deadline=end, cleanup_deadline=min(started+240, end+6), env=env,
                rss_limit=CAPS['rss_bytes'])
            records.append(receipt)
            peak = max(peak, receipt['sampled_session_plus_caller_rss'])
            require(receipt['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_SESSION'
                    and not receipt['remaining_session']['live'], 'owned session fully quiescent')
            require(receipt['finished_at'] < receipt['cleanup_deadline'], 'same cleanup deadline')
            tick()
            return receipt

        def self_capability():
            before = fd_count(); require(api.probe() == api.backend, 'self capability')
            fd = api.pidfd_open(os.getpid())
            try:
                require(not os.get_inheritable(fd), 'CLOEXEC')
                api.pidfd_send_signal(fd, 0)
            finally:
                os.close(fd)
            require(fd_count() == before, 'probe FD release')

        def invalid_and_closed_fd():
            for value in (True, 0, -1, 2**31, 1.5):
                expect(ValueError, lambda: api.pidfd_open(value))
            for value in (True, -1, 1, 2**32):
                expect(ValueError, lambda: api.pidfd_open(os.getpid(), value))
            for args in ((-1, 0), (True, 0), (2**31, 0), (0, 1), (0, True),
                         (0, 0, 1), (0, 0, None, 1)):
                expect(ValueError, lambda: api.pidfd_send_signal(*args))
            fd = api.pidfd_open(os.getpid()); os.close(fd)
            expect(OSError, lambda: api.pidfd_send_signal(fd, 0), errno.EBADF)

        def local_abi_errors():
            class Function:
                def __init__(self, result, code):
                    self.result = result; self.code = code; self.args = None
                def __call__(self, *args):
                    self.args = args; ctypes.set_errno(self.code); return self.result
            class Symbols:
                pass
            expect(AttributeError, lambda: _Libc(Symbols()))
            for code in (errno.ESRCH, errno.EPERM, errno.ENOSYS, errno.EMFILE, errno.EINVAL):
                symbols = Symbols()
                symbols.pidfd_open = Function(-1, code)
                symbols.pidfd_send_signal = Function(-1, code)
                bridge = _Libc(symbols)
                require(symbols.pidfd_open.argtypes == [ctypes.c_int, ctypes.c_uint]
                        and symbols.pidfd_send_signal.argtypes ==
                        [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
                        and symbols.pidfd_open.restype is ctypes.c_int
                        and symbols.pidfd_send_signal.restype is ctypes.c_int, 'explicit ABI')
                expect(OSError, lambda: bridge.pidfd_open(123, 0), code)
                expect(OSError, lambda: bridge.pidfd_send_signal(42, 0, None, 0), code)
                require(symbols.pidfd_open.args == (123, 0)
                        and symbols.pidfd_send_signal.args == (42, 0, None, 0), 'ABI arguments')
            symbols.pidfd_open = Function(-1, 0); symbols.pidfd_send_signal = Function(1, 0)
            bridge = _Libc(symbols)
            expect(RuntimeError, lambda: bridge.pidfd_open(123, 0))
            expect(RuntimeError, lambda: bridge.pidfd_send_signal(42, 0, None, 0))

        def normal_and_error_exit():
            a = session('import signal;signal.alarm(6);print("owned normal")', 5)
            b = session('import signal;signal.alarm(6);raise SystemExit(7)', 5)
            require(a['status'] == 'COMPLETED' and a['returncode'] == 0, 'normal child')
            require(b['status'] == 'ERROR' and b['returncode'] == 7, 'nonzero exit not accepted')

        def owned_signal():
            p = spawn(4)
            require(owned.kill_owned(p.pid, p.pid, started+240, pidfd=api), 'owned pidfd signal')
            require(wait(p) == -int(signal.SIGKILL), 'owned child killed/reaped')

        def foreign_session_refusal():
            p = spawn(2); before = fd_count()
            require(not owned.kill_owned(p.pid, os.getsid(0), started+240, pidfd=api), 'foreign SID rejected')
            require(fd_count() == before and p.poll() is None, 'foreign target untouched; fd closed')
            require(wait(p) == 0, 'foreign sentinel natural expiry, no signal')

        def deadline_and_descendant_cleanup():
            before = Path('/proc/self/task/'+str(os.getpid())+'/children').read_text().split()
            expect(TimeoutError, lambda: owned.execute([], run_deadline=started-1,
                cleanup_deadline=started+240, env=env, rss_limit=CAPS['rss_bytes']))
            expect(TimeoutError, lambda: owned.kill_owned(os.getpid(), os.getsid(0), started-1, pidfd=api))
            require(before == Path('/proc/self/task/'+str(os.getpid())+'/children').read_text().split(),
                    'expired deadline created no child')
            child = 'import signal,time;signal.alarm(6);time.sleep(5)'
            code = ('import os,signal,subprocess,sys,time;signal.alarm(6);'
                    'p=subprocess.Popen([sys.executable,"-I","-S","-B","-c",'+repr(child)+'],process_group=0);'
                    'print(p.pid,os.getsid(p.pid),os.getpgid(p.pid),flush=True);time.sleep(5)')
            receipt = session(code, .5)
            pid, sid, pgid = map(int, base64.b64decode(receipt['log_base64']).decode().split())
            require(receipt['status'] == 'TIMEOUT' and sid == receipt['session']
                    and pgid == pid != sid and receipt['kill_signals'] >= 2,
                    'different process group in same session cleaned on deadline')

        def descriptor_balance():
            for _ in range(3):
                api.probe()
            require(fd_count() == initial_fds, 'no leaked descriptor after all branches')
            require(all(p.returncode is not None for p in tracked), 'owned fixtures reaped')
            require(initial_children == Path('/proc/self/task/'+str(os.getpid())+'/children').read_text().split(),
                    'no unreaped direct fixture')

        functions = (self_capability, invalid_and_closed_fd, local_abi_errors,
                     normal_and_error_exit, owned_signal, foreign_session_refusal,
                     deadline_and_descendant_cleanup, descriptor_balance)
        for name, function in zip(IDS, functions, strict=True):
            tick(); begin = time.monotonic(); before = fd_count()
            try:
                function(); tick()
                require(fd_count() == before, 'per-case descriptor balance')
            except BaseException as exc:
                rows.append(dict(id=name, status='FAIL', seconds=time.monotonic()-begin,
                                 error=type(exc).__name__, message=str(exc)[:300]))
                raise
            rows.append(dict(id=name, status='PASS', seconds=time.monotonic()-begin))
        result.update(status='PASS', sampled_batch_rss_bytes=peak,
                      initial_fd_count=initial_fds, final_fd_count=fd_count())
    except BaseException as exc:
        result.update(status='FAIL', error=type(exc).__name__, message=str(exc)[:500])
    finally:
        # A failed owned_session may still hold an unreaped direct leader.
        # Keep that incarnation reserved while observing this session; do not
        # signal a numeric PID, or observe a reused SID after reaping it.
        for row in records:
            if row['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_SESSION' or row['session'] is None:
                continue
            sid = row['session']; end = min(started+260, time.monotonic()+8)
            recovered = False
            try:
                while time.monotonic() < end:
                    info = os.waitid(os.P_PID, sid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    current = owned.observe(sid, end)
                    if info is not None and not current['live']:
                        waited, status = os.waitpid(sid, os.WNOHANG)
                        require(waited == sid, 'own leader reaped after self expiry')
                        events.append(dict(event='FAILED_SESSION_REAP', pid=sid, wait_status=status,
                                           at=time.monotonic()))
                        recovered = True; break
                    time.sleep(.01)
            except BaseException as exc:
                events.append(dict(event='FAILED_SESSION_CLEANUP_ERROR', pid=sid,
                                   error=type(exc).__name__, at=time.monotonic()))
            if not recovered:
                result.update(status='FAIL', cleanup_incomplete=True)
        # Only direct self-expiring fixtures, no unowned cleanup/fallback.
        for p in tracked:
            if p.returncode is None:
                try:
                    p.wait(timeout=min(8, max(.001, started+260-time.monotonic())))
                    events.append(dict(event='FAILURE_REAP', pid=p.pid, returncode=p.returncode,
                                       at=time.monotonic()))
                except BaseException:
                    result.update(status='FAIL', cleanup_incomplete=True)
    result['unexecuted'] = [name for name in IDS if name not in {row['id'] for row in rows}]
    result['before_export'] = time.monotonic()
    result['seconds_before_export'] = result['before_export']-started
    result['online_acceptance'] = result['status'] == 'PASS' and result['before_export'] < deadline
    if directory_created:
        save(RUN+'/result.json', result)
    exported = time.monotonic()
    receipt = dict(schema='PIDFD_FINITE_RECEPTION_V1', result_sha256=hashlib.sha256(packed(result)).hexdigest(),
                   started=started, received_at=exported, seconds=exported-started,
                   export_seconds=exported-result['before_export'],
                   online_acceptance=result['online_acceptance'] and exported < deadline,
                   result_directory=RUN if directory_created else None)
    if directory_created:
        save(RUN+'/reception.json', receipt)
    finished = time.monotonic()
    terminal = dict(schema='PIDFD_FINITE_RETURN_V1', result=result, reception=receipt,
                    finished_at=finished, seconds=finished-started,
                    online_acceptance=receipt['online_acceptance'] and finished < deadline)
    print(json.dumps(terminal, sort_keys=True), flush=True)
    after_export = time.monotonic()
    exported = dict(schema='PIDFD_FINITE_STDOUT_EXPORTED_V1', after_export=after_export,
                    seconds=after_export-started, stdout_export_seconds=after_export-finished,
                    online_acceptance=terminal['online_acceptance'] and after_export < deadline)
    print(json.dumps(exported, sort_keys=True), file=sys.stderr, flush=True)
    # Successful process exit is also REQUIRED by the saved-record auditor.
    # It cannot be inferred from a candidate acceptance flag written earlier.
    timely = time.monotonic() < deadline
    signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if exported['online_acceptance'] and timely else 1


if __name__ == '__main__':
    raise SystemExit(main())
