"""R3 fixed-resource parent. Old R2 CLI keeps its default legacy policy.

Wall deadline plus sampled own-process-group RSS; not a strict instantaneous
allocation limit. Both arms pay imports, validation, translation and solving.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_paired_execution_r2 import validate, worker


def group_rss(pgid):
    total = 0
    for item in Path('/proc').iterdir():
        if not item.name.isdecimal():
            continue
        try:
            fields = (item/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:
                total += int(fields[21])*os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError, ProcessLookupError):
            continue
    return total


def supervise(command, cwd, folder, seconds, rss_limit):
    if not 0 < seconds <= 300 or type(rss_limit) is not int or rss_limit <= 0:
        raise ValueError('finite bounded supervision required')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg')
    for name in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'TMPDIR'):
        path = folder/name.lower()
        path.mkdir()
        env[name] = str(path.resolve())
    peak, state, error, code, proc = 0, 'ERROR', None, None, None
    with (folder/'stdout.txt').open('x') as out, (folder/'stderr.txt').open('x') as err:
        try:
            if time.monotonic()-start >= seconds:
                raise TimeoutError('preflight exhausted deadline before spawn')
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
            while True:
                peak = max(peak, group_rss(proc.pid))
                elapsed = time.monotonic()-start
                if elapsed >= seconds:
                    state = 'TIMEOUT'
                    break
                if peak > rss_limit:
                    state = 'RESOURCE_LIMIT'
                    break
                code = proc.poll()
                if code is not None:
                    state = 'COMPLETED' if code == 0 else 'ERROR'
                    break
                time.sleep(min(.05, max(.001, seconds-elapsed)))
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                state = 'TIMEOUT'
            error = repr(exc)
        finally:
            if proc is not None:
                # Clean only the group created by this call, including children
                # left behind after parent exit. Never target another user's job.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = proc.wait()
    elapsed = time.monotonic()-start
    if state == 'COMPLETED' and elapsed >= seconds:
        state = 'TIMEOUT'
    receipt = {'status': state, 'exit_code': code, 'error': error, 'command': command,
               'deadline_seconds': seconds, 'execution_including_preflight_seconds': elapsed,
               'peak_sampled_group_rss_bytes': peak, 'group_rss_limit_bytes': rss_limit,
               'rss_poll_seconds': .05, 'rss_is_sampled_not_instantaneous_cap': True,
               'postflight_in_execution_budget': False, 'source_unchanged': None,
               'stdout_sha256': sha256(folder/'stdout.txt'), 'stderr_sha256': sha256(folder/'stderr.txt')}
    receipt['total_with_postflight_seconds'] = time.monotonic()-start
    receipt['receipt_own_write_excluded_from_this_clock'] = True
    write(folder/'receipt.json', receipt)
    return receipt


def terminal(folder, receipt):
    """Outer failure wins, even with a truncated or late candidate file."""
    folder = Path(folder)
    result, parse_error = None, None
    file = folder/'result.json'
    if file.exists():
        try:
            result = json.loads(file.read_text())
            if not isinstance(result, dict) or not isinstance(result.get('status'), str):
                raise ValueError('missing candidate status')
        except (ValueError, OSError) as exc:
            parse_error = repr(exc)
            result = None
    status = (result['status'] if result else 'ERROR') if receipt['status'] == 'COMPLETED' else receipt['status']
    return {'status': status, 'result_sha256': sha256(file) if file.exists() else None,
            'result_parse_error': parse_error}


def resource_config(cfg):
    if cfg['hybridz'] != {'sparse_resource_policy': 'csr_bytes_v1', 'sparse_representation_bytes': 2**31}:
        raise ValueError('frozen CSR resource policy')
    if cfg['group_rss_limit_bytes'] != 8*2**30:
        raise ValueError('frozen sampled process-group RSS policy')


def run(path):
    batch_start = time.monotonic()
    cfg = json.loads(path.read_text())
    resource_config(cfg)
    validate(cfg)
    root = Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'launch.json', {'config_sha256': sha256(path), 'pid': os.getpid(),
                             'execution_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()})
    rows, blocked = [], False
    for rank, request in enumerate(cfg['requests']):
        for arm in (['act', 'author'] if rank % 2 == 0 else ['author', 'act']):
            if blocked:
                rows.append({'id': request['id'], 'arm': arm, 'status': 'NOT_STARTED_AFTER_ERROR'})
                continue
            folder = root/f"{request['id']}_{arm}"
            receipt = supervise([cfg['python'][arm], str(Path(__file__).resolve()), '--config', str(path.resolve()),
                                 '--worker', request['id'], '--arm', arm], str(Path(__file__).resolve().parents[1]),
                                folder, cfg['seconds'], cfg['group_rss_limit_bytes'])
            candidate = terminal(folder, receipt)
            row = {'id': request['id'], 'arm': arm, **candidate,
                   'seconds': receipt['execution_including_preflight_seconds'],
                   'receipt_sha256': sha256(folder/'receipt.json')}
            write(folder/'terminal.json', row)
            rows.append(row)
            blocked = row['status'] in ('ERROR', 'SOURCE_CHANGED')
            write(root/f'progress_{len(rows):03d}.json', {'rows': rows, 'config_sha256': sha256(path)})
    write(root/'summary.json', {'config_sha256': sha256(path), 'rows': rows})
    write(root/'batch_cost.json', {'config_sha256': sha256(path),
        'batch_wall_through_summary_seconds': time.monotonic()-batch_start,
        'charged_request_seconds': sum(r.get('seconds', 0.) for r in rows),
        'includes': 'top-level validation, launch, all receipts/terminals/progress, summary write',
        'excludes': 'this cost file write and independent archive/audit'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--worker')
    parser.add_argument('--arm', choices=['act', 'author'])
    args = parser.parse_args()
    if args.worker:
        from act.config.config import HybridZConfig
        cfg = json.loads(args.config.read_text())
        resource_config(cfg)
        worker(cfg, args.config, args.worker, args.arm, hybridz_config=HybridZConfig(**cfg['hybridz']))
    else:
        run(args.config)
