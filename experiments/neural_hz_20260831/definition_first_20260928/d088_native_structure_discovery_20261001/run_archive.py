"""Exclusive stdlib supervisor for the one D088 archived-native probe.

One worker process, total subprocess timeout240s; no retry or mathematical
rerun. The worker authenticates every inherited local source before imports.
This supervisor also saves a terminal receipt if the worker is forcibly killed.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
RUN = EXP / 'results/d088_native_structure_discovery_20261001_v1'
OUT = RUN / 'archive_probe_supervisor'
RESERVE, MEMORY_CAP, AS_CAP = 65536, 1024**3, 16*1024**3


def read_json(path, limit=RESERVE):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError('missing/oversized/linked supervisor input')
    return json.loads(path.read_text())


def sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing/linked supervision source')
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def rss():
    with Path('/proc/self/status').open() as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
    raise ValueError('own RSS unavailable')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    if not RUN.is_dir():
        raise ValueError('the one mathematical run must exist first')
    OUT.mkdir(exist_ok=False)  # First attempt consumes the archive-stage version.
    started, initial = time.monotonic(), rss()
    tracemalloc.start()
    record = dict(schema='d088_archive_supervisor_v1', worker_launched=False,
        timeout=False, worker_exit=None, archive_probe_qualified=False,
        complete_physical_qualification=False, actual_model_binding_qualified=False,
        gpu_computation_completed=False, formal_gain=0, source_drift=[], artifacts={})
    identities = {}
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        if os.environ.get('LD_PRELOAD'):
            raise ValueError('unexpected LD_PRELOAD')
        if (RUN/'archive_probe').exists():
            raise ValueError('archive worker version is already consumed')
        frozen = read_json(HERE/'freeze.json')
        done = read_json(RUN/'exit.json')
        if (frozen.get('schema') != 'd088_native_structure_discovery_v1'
                or done.get('all_stages_passed') is not True
                or done.get('supervisor_exit') != 0 or done.get('tests_count') != 3825
                or done.get('test_files') != 183):
            raise ValueError('complete frozen mathematical gate has not passed')
        identities = frozen['source_sha256']
        if not all(str(HERE/name) in identities for name in ('archive_probe.py','run_archive.py')):
            raise ValueError('supervision source is not frozen')
        for name, digest in identities.items():
            if sha(Path(name)) != digest:
                raise ValueError('frozen supervision source mismatch: '+name)
        env = dict(os.environ)
        env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        for name in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH',
                     'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
            path = OUT/name.lower()
            path.mkdir()
            env[name] = str(path)
        command = [sys.executable, '-B', str(HERE/'archive_probe.py'), '--enabled']
        record.update(command=command, cpu_affinity=sorted(os.sched_getaffinity(0)),
            worker_timeout_s=240, as_cap_bytes=AS_CAP, mathematical_run=str(RUN))
        with (OUT/'worker.log').open('xb') as stream:
            launch = time.monotonic()
            record['worker_launched'] = True
            try:
                completed = subprocess.run(command, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=240,
                    check=False, cwd=str(HERE.parents[3]))
                record['worker_exit'] = completed.returncode
            except subprocess.TimeoutExpired:
                # subprocess.run kills and waits for THIS child before raising.
                record['timeout'] = True
            finally:
                record['subprocess_wall_s'] = time.monotonic()-launch
        receipt = RUN/'archive_probe/worker.json'
        if receipt.is_file():
            worker = read_json(receipt)
            record['worker_receipt'] = dict(file=str(receipt), sha256=sha(receipt),
                bytes=receipt.stat().st_size, qualified=worker.get('archive_probe_qualified') is True,
                failure=worker.get('failure'))
            record['archive_probe_qualified'] = bool(record['worker_exit'] == 0
                and not record['timeout'] and worker.get('archive_probe_qualified') is True
                and worker.get('transformed') is True and worker.get('all_groups_applied') is True)
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:2048])
    finally:
        sealing = time.monotonic()
        for name, digest in identities.items():
            try:
                if sha(Path(name)) != digest:
                    record['source_drift'].append(name)
            except Exception as exc:
                record.setdefault('identities_unchecked', []).append(dict(path=name, reason=str(exc)[:256]))
        log = OUT/'worker.log'
        if log.is_file():
            try:
                record['artifacts']['worker.log'] = dict(bytes=log.stat().st_size, sha256=sha(log))
            except Exception as exc:
                record['sealing_error'] = str(exc)[:512]
        current, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        growth = max(0, peak_rss-initial)
        memory_ok = (peak_rss+RESERVE <= MEMORY_CAP and growth+RESERVE <= MEMORY_CAP
                     and peak+metadata+RESERVE <= MEMORY_CAP)
        record.update(supervisor_wall_s=time.monotonic()-started,
            finalization_wall_s=time.monotonic()-sealing, initial_rss_bytes=initial,
            peak_rss_bytes=peak_rss, rss_growth_bytes=growth, tracemalloc_current_bytes=current,
            tracemalloc_peak_bytes=peak, tracemalloc_metadata_bytes=metadata,
            summary_reserve_bytes=RESERVE, supervisor_memory_gate_passed=memory_ok)
        record['archive_probe_qualified'] = bool(record['archive_probe_qualified'] and memory_ok
            and not record['source_drift'] and not record.get('identities_unchecked')
            and not record.get('failure') and not record.get('sealing_error'))
        encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
        if len(encoded.encode()) > RESERVE:
            record['archive_probe_qualified'] = False
            encoded = json.dumps(dict(schema=record['schema'], archive_probe_qualified=False,
                formal_gain=0, failure={'type':'SummaryReserveExceeded'},
                worker_exit=record['worker_exit'], timeout=record['timeout']))
        with (OUT/'supervisor.json').open('x') as stream:
            stream.write(encoded+'\n')
    if not record['archive_probe_qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
