"""One frozen old-smoke monolithic request, 300s, diagnostic spans only."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.conv_three_arm_contract import (ACT, ROOT, read, selection, request_for,
                                            wrapper_hashes)
from scripts.run_conv_three_arm import execute, terminal, wait_resources
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256, _git_value

PROTOCOL = ROOT/'scripts/conv_f0_timing_protocol_r1.json'
FILES = ('scripts/conv_f0_timing_protocol_r1.json', 'scripts/f0_timing_trace.py',
         'scripts/run_conv_f0_timing.py', 'scripts/audit_conv_f0_timing.py',
         'scripts/test_conv_f0_timing.py')
DEFAULT = ROOT/'data/moe/results/conv_f0_timing_20260915_r1'


def worker(root, started):
    from scripts.f0_timing_trace import Recorder, install, restore
    recorder = Recorder(root/'trace.jsonl', started,
                        {'request_sha256': _sha256(root/'request.json'),
                         'runtime_sha256': _sha256(root.parent/'runtime.json')})
    patches = []
    try:
        recorder.emit('INSTALL_BEGIN')
        patches = install(recorder)
        from act.pipeline.moe.conv_three_arm_worker import worker as original
        recorder.wrap(original, 'conv_request.worker')(root, started)
        recorder.emit('WORKER_COMPLETE')
    finally:
        restore(patches)
        recorder.close()


def run(root):
    root = root.resolve()
    if root != DEFAULT:
        raise ValueError('R1 has exactly one frozen output directory; no retry/alternate input')
    if (Path(sys.executable).resolve() != Path(ACT).resolve()
            or _git_value('branch', '--show-current') != 'feat/moe-route-verification'
            or _git_value('status', '--porcelain')
            or _git_value('rev-parse', 'HEAD') != _git_value('rev-parse', '@{upstream}')):
        raise ValueError('clean pushed feature branch and ACT env required')
    value = selection()
    protocol = read(PROTOCOL)
    job = next(j for j in value['smoke_jobs'] if j['job_id'] == protocol['job_id'])
    if job['method'] != 'monolithic' or job['dataset_index'] != 0 or protocol['budget_seconds'] != 300:
        raise ValueError('diagnostic identity drift')
    parent = ROOT/protocol['parent_root']
    root.mkdir(exist_ok=False)
    runtime = {'schema': 'conv_f0_timing_r1', 'protocol': protocol,
               'git_head': _git_value('rev-parse', 'HEAD'), 'selection': value,
               'sources': {p: _sha256(ROOT/p) for p in FILES},
               'old_wrappers': wrapper_hashes(),
               'parent_request_sha256': _sha256(parent/job['job_id']/'request.json'),
               'parent_terminal_sha256': _sha256(parent/job['job_id']/'terminal.json'),
               'started_unix': time.time(), 'full_started': False}
    request = request_for(value, job, runtime['git_head'])
    old = read(parent/job['job_id']/'request.json')
    if {k:v for k,v in request.items() if k != 'head'} != {k:v for k,v in old.items() if k != 'head'}:
        raise ValueError('diagnostic changed frozen request')
    atomic_json(root/'runtime.json', runtime)
    wait = wait_resources(root, job)
    directory = root/job['job_id']; directory.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
           'MKL_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':'', 'PYTHONHASHSEED':'0'}
    started = time.monotonic()
    atomic_json(directory/'request.json', request)
    error = None
    try:
        code, expired = execute([ACT, '-m', 'scripts.run_conv_f0_timing', '--worker',
            '--root', str(directory), '--started', repr(started)], directory/'worker.log', started, env,
            heartbeat=lambda pid, elapsed: atomic_json(root/'supervisor.json',
                {'state':'RUNNING', 'pid':pid, 'elapsed_seconds':elapsed, 'full_started':False}))
        row = terminal(root, job, started, code, expired, wait)
    except BaseException as exc:
        error = repr(exc)
        row = terminal(root, job, started, None, False, wait, error)
    atomic_json(root/'run_terminal.json', {'state':'DIAGNOSTIC_STOPPED', 'error':error,
        'full_started':False, 'row':row, 'trace_sha256':_sha256(directory/'trace.jsonl')
        if (directory/'trace.jsonl').exists() else None})
    result = subprocess.run([ACT, '-m', 'scripts.audit_conv_f0_timing', '--root', str(root),
                             '--output', str(root/'audit.final.json')], cwd=ROOT, env=env)
    atomic_json(root/'supervisor.json', {'state':'REVIEW_REQUIRED', 'audit_returncode':result.returncode,
                                       'full_started':False})
    return result.returncode if not error and row['status'] != 'ERROR' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--started', type=float)
    args = parser.parse_args()
    if args.worker:
        worker(args.root, args.started)
    else:
        def stop(signum, frame):
            raise KeyboardInterrupt(f'supervisor signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        with (ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sys.exit(run(args.root))
