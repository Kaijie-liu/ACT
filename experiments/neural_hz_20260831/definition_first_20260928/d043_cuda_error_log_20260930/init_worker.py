"""One torch.cuda.is_available call, driver logs only; never a GPU fixture."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
RUN = HERE.parent.parent / 'results/d043_cuda_error_log_20260930_v1'
AS_CAP, MEMORY_CAP, RESERVE, LOG_CAP = 16 * 1024**3, 1024**3, 65536, 16 * 1024**2


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmSize:', 'VmRSS:', 'VmHWM:')):
            key, value, unit = line.split()
            if unit != 'kB':
                raise ValueError('unexpected process-memory unit')
            values[key[:-1] + '_bytes'] = int(value) * 1024
    if len(values) != 3:
        raise ValueError('incomplete own-process memory telemetry')
    return values


def event(name):
    print(json.dumps(dict(event=name, epoch_s=time.time(),
                         monotonic_s=time.monotonic(), memory=memory())), flush=True)


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
    resource.setrlimit(resource.RLIMIT_FSIZE, (LOG_CAP, LOG_CAP))
    sys.dont_write_bytecode = True
    started = time.monotonic()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tracemalloc.start()
    record = dict(worker_pid=os.getpid(), api='torch.cuda.is_available', api_calls=0,
        api_returned=False, cuda_available=None, gpu_computation_completed=False,
        source_census_completed=False, native_HZ_admitted=False,
        complete_physical_qualification=False, observed_context_bytes=None,
        combined_physical_gate='unknown', formal_gain=0,
        diagnostic_solver_calls=0, model_forward_calls=0, new_benchmark_solves=0,
        numerical_fixture_executed=False, driver_cause_inferred=False,
        host_observations_within_caps=False)
    try:
        record['initial_memory'] = memory()
        if len(os.sched_getaffinity(0)) != 1 or not __debug__:
            raise ValueError('CPU1 and enabled assertions required')
        freeze_path, runner_path = HERE / 'freeze.json', HERE / 'run_diagnostic.py'
        if (freeze_path.is_symlink() or not 0 < freeze_path.stat().st_size <= RESERVE
                or runner_path.is_symlink()):
            raise ValueError('invalid small freeze or runner path')
        frozen = json.loads(freeze_path.read_text())
        if sha(runner_path) != frozen['source_sha256'][str(runner_path)]:
            raise ValueError('new runner identity drift')
        spec = importlib.util.spec_from_file_location('d043_readonly_runner', runner_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner.check_freeze()
        registered = runner.read_json(RUN / 'preregistered.json')
        if (registered['freeze_sha256'] != sha(freeze_path)
                or registered['worker_api'] != 'torch.cuda.is_available_once'
                or registered['worker_environment'] != runner.environment(os.environ)
                or os.environ.get('CUDA_VISIBLE_DEVICES') != runner.GPU
                or os.environ.get('CUDA_LOG_FILE') != 'stderr'
                or os.environ.get('CUDA_MODULE_LOADING') != 'LAZY'
                or os.environ.get('LD_PRELOAD')
                or os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK') == '1'):
            raise ValueError('frozen CUDA path/environment differs or NVML override present')
        for name in runner.NEW_FILES:
            path = HERE / name
            if sha(path) != registered['source_sha256'][str(path)]:
                raise ValueError('new source drift before runtime import')
        if (Path(sys.executable).resolve() != runner.PYTHON.resolve()
                or sha(sys.executable) != registered['source_sha256'][str(runner.PYTHON.resolve())]):
            raise ValueError('frozen interpreter drift')
        if resource.getrlimit(resource.RLIMIT_AS) != (AS_CAP, AS_CAP):
            raise ValueError('AS16GiB differs')
        record.update(cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=AS_CAP, log_file_cap_bytes=LOG_CAP,
            environment=runner.environment(os.environ), python_version=sys.version)
        event('before_torch_import')
        import torch
        record.update(torch_version=str(torch.__version__),
                      torch_cuda_version=str(torch.version.cuda))
        event('after_torch_import_before_availability')
        record['api_calls'] = 1
        record['cuda_available'] = bool(torch.cuda.is_available())
        record['api_returned'] = True
        event('after_availability_no_further_cuda_calls')
        if not record['cuda_available']:
            raise RuntimeError('CUDA unavailable; one call consumed, no retry or fallback')
    except BaseException as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        try:
            _, peak = tracemalloc.get_traced_memory()
            metadata = tracemalloc.get_tracemalloc_memory()
            growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
            record.update(final_memory=memory(), rss_highwater_growth_bytes=growth,
                traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
                summary_reserve_bytes=RESERVE, final_summary_reserve_bytes=RESERVE,
                host_observations_within_caps=(growth + RESERVE <= MEMORY_CAP
                    and peak + metadata + RESERVE <= MEMORY_CAP))
        except BaseException as exc:
            record['host_observations_within_caps'] = False
            record['memory_check_failure'] = str(exc)[:4096]
        record['wall_s'] = time.monotonic() - started
        if not record['host_observations_within_caps'] or record['wall_s'] > 240:
            record.setdefault('failure', dict(type='ResourceError', reason='unchanged host/time gate failed'))
        record['worker_exit'] = 0 if record['cuda_available'] is True and 'failure' not in record else 1
        payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
        if len(payload) > RESERVE:
            raise ValueError('worker summary exceeds its reserved boundary')
        with (RUN / 'worker.json').open('xb') as stream:
            stream.write(payload)
    return record['worker_exit']


if __name__ == '__main__':
    raise SystemExit(main())
