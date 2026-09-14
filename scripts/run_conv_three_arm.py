"""Smoke-only supervisor. No full launch, resume, overwrite or automatic Git writes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256, _git_value
from scripts.conv_three_arm_contract import (ACT, ARMS, ROOT, DEFAULT_ROOT,
    SELECTION_HASH, PROTOCOL_HASH, selection, wrapper_hashes, request_for, read)


def resource_state():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {'available_ram_gib': int(memory['MemAvailable'].split()[0])*1024/2**30,
            'free_disk_gib': shutil.disk_usage(ROOT).free/2**30,
            'load_per_core': os.getloadavg()[0]/(os.cpu_count() or 1)}


def resources_ok(state):
    return (state['available_ram_gib'] >= 16 and state['free_disk_gib'] >= 5
            and state['load_per_core'] <= .5)


def wait_resources(root, job):
    start = time.monotonic()
    while True:
        state = resource_state()
        wait = time.monotonic()-start
        atomic_json(root/'supervisor.json', {'state': 'RESOURCE_READY' if resources_ok(state) else 'RESOURCE_WAIT',
                    'job_id': job['job_id'], 'resource': state, 'wait_seconds': wait, 'updated_unix': time.time()})
        if resources_ok(state):
            return {'seconds': wait, 'at_launch': state}
        if wait >= 86400:
            raise TimeoutError('resource wait exhausted; no worker launched')
        time.sleep(min(30, 86400-wait))


def execute(command, log, started, env, budget=300, heartbeat=None):
    """Own a new process group; clean it on deadline and supervisor exceptions."""
    expired = False
    with Path(log).open('x') as handle:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                remaining = budget-(time.monotonic()-started)
                if remaining <= 0:
                    expired = True
                    break
                try:
                    proc.wait(timeout=min(10, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if heartbeat:
                        heartbeat(proc.pid, time.monotonic()-started)
        finally:
            # Also remove a cross-environment descendant left after leader exit.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    return proc.returncode, expired or time.monotonic()-started > budget


def terminal(root, job, started, code, expired, wait, error=None):
    directory = root/job['job_id']
    row = {**job, 'budget_seconds': 300, 'wall_seconds': time.monotonic()-started,
           'outer_timeout': expired, 'return_code': code, 'status': 'TIMEOUT' if expired else 'ERROR',
           'package': None, 'snapshot_sha256': None, 'resource_wait': wait,
           'request_sha256': _sha256(directory/'request.json'),
           'evidence_level': 'CROWN_NUMERICAL_FILTER' if job['method']=='crown' else 'HZ_POLICY_ACCEPTED'}
    if error is not None:
        row['error'] = error
    for name, key in [('common_facts.json', 'snapshot_sha256'), ('routes.json', 'routes_sha256'),
                      ('external.json', 'external_sha256')]:
        if (directory/name).exists():
            row[key] = _sha256(directory/name)
    if not expired and code == 0 and error is None:
        try:
            if job['method'] == 'crown':
                row['status'] = read(directory/'external.json')['status']
                if row['status'] not in ('POSITIVE', 'UNSAFE', 'UNKNOWN'):
                    raise ValueError('invalid external terminal')
            else:
                package = directory/'package'
                row.update(status=read(package/'manifest.json')['status'], package=str(package),
                           manifest_sha256=_sha256(package/'manifest.json'))
                if row['status'] not in ('SAFE', 'UNSAFE', 'UNKNOWN', 'TIMEOUT'):
                    raise ValueError('invalid HZ terminal')
        except Exception as exc:
            row.update(status='ERROR', package=None, error=repr(exc))
    # Publish before measuring submission completion; if it crosses the deadline,
    # replace ONLY this provisional record by a conservative TIMEOUT terminal.
    atomic_json(directory/'terminal.json', row)
    row['wall_seconds'] = time.monotonic()-started
    if row['wall_seconds'] > 300:
        row.update(status='TIMEOUT', outer_timeout=True, package=None)
    atomic_json(directory/'terminal.json', row)
    with (root/'rows.jsonl').open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    return row


def run(root):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT/'data/moe/results'):
        raise ValueError('output must be a new local results directory')
    if Path(sys.executable).resolve() != Path(ACT).resolve():
        raise ValueError('act-py312 required')
    if (_git_value('branch', '--show-current') != 'feat/moe-route-verification'
            or _git_value('status', '--porcelain')
            or _git_value('rev-parse', 'HEAD') != _git_value('rev-parse', '@{upstream}')):
        raise ValueError('clean, committed and pushed feature checkout required')
    value = selection(); wrappers = wrapper_hashes(); head = _git_value('rev-parse', 'HEAD')
    root.mkdir(exist_ok=False)
    runtime = {'schema': 'conv_smoke_execution_v1', 'smoke': True, 'full_authorized': False,
               'git_head': head, 'selection_sha256': SELECTION_HASH, 'protocol_sha256': PROTOCOL_HASH,
               'wrapper_sha256': wrappers, 'selection': value, 'started_unix': time.time(),
               'config': {'methods': value['identities']['method_configs']}}
    atomic_json(root/'runtime.json', runtime)
    schedule = value['smoke_jobs']; completed = []
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0'}
    error = None
    try:
        for job in schedule:
            if wrapper_hashes()!=wrappers or _git_value('status','--porcelain') or selection()!=value:
                raise ValueError('source/input drift during execution')
            wait = wait_resources(root, job)
            directory = root/job['job_id']; directory.mkdir()
            started = time.monotonic()
            atomic_json(directory/'request.json', request_for(value, job, head))
            try:
                code, expired = execute([ACT, '-m', 'act.pipeline.moe.conv_three_arm_worker',
                     '--root', str(directory), '--started', repr(started)], directory/'worker.log', started, env,
                     heartbeat=lambda pid, elapsed: atomic_json(root/'supervisor.json',
                         {'state':'RUNNING', 'job_id':job['job_id'], 'pid':pid,
                          'elapsed_seconds':elapsed, 'updated_unix':time.time()}))
                row = terminal(root, job, started, code, expired, wait)
            except BaseException as exc:
                row = terminal(root, job, started, None, False, wait, repr(exc))
                completed.append(row)
                raise
            completed.append(row)
            print(f"{len(completed)}/6 {job['job_id']} {row['status']} {row['wall_seconds']:.2f}s", flush=True)
            if row['status']=='ERROR':
                raise RuntimeError('execution error: fail-stop, no replacement')
    except BaseException as exc:
        error = repr(exc)
    finally:
        atomic_json(root/'run_terminal.json', {'state':'EXECUTION_ERROR' if error else 'EXECUTION_COMPLETED',
            'error':error, 'completed_job_ids':[r['job_id'] for r in completed],
            'unattempted':schedule[len(completed):], 'full_started':False})
    # Separate interpreter, including after a fail-stop. Audit conformance is
    # distinct from the smoke gate and can document an intact failed attempt.
    result = subprocess.run([ACT, '-m', 'scripts.audit_conv_three_arm', '--root', str(root),
                             '--output', str(root/'audit.final.json')], cwd=ROOT, env=env)
    checked = read(root/'audit.final.json') if (root/'audit.final.json').exists() else None
    atomic_json(root/'supervisor.json', {'state': 'SMOKE_PASSED' if checked and checked['smoke_gate']=='PASS' else 'STOPPED_REVIEW_REQUIRED',
                'audit_returncode':result.returncode, 'full_started':False, 'updated_unix':time.time()})
    return 0 if not error and result.returncode==0 and checked['smoke_gate']=='PASS' else 1


if __name__=='__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'supervisor signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args=parser.parse_args()
    with (ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        sys.exit(run(args.root))
