"""One same-cap CUDA availability observation, never a GPU computation test.

Only the frozen supervisor launches this opt-in worker.  The sole explicit CUDA
API call is torch.cuda.is_available(); its internals may contact the driver, but
no device-context, allocation, kernel, model, or solver qualification is inferred.
The supervisor owns the external 240-second deadline and syscall capture.
"""

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
RUN = HERE.parent.parent / 'results/d247_cuda_same_cap_trace_20261005_v1'
FILES = ('PREREG.md', 'run_diagnostic.py', 'collection_contract.py', 'init_worker.py')
SCHEMA = 'd247_cuda_same_cap_trace_v1'
GPU = 'GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc'
AS_CAP, MEMORY_CAP, RESERVE, LOG_CAP = 16 * 1024**3, 1024**3, 65536, 16 * 1024**2
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
STATUS_KEYS = ('Name', 'State', 'Pid', 'PPid', 'TracerPid', 'Threads',
               'Cpus_allowed_list', 'Mems_allowed_list', 'NoNewPrivs',
               'Seccomp', 'Seccomp_filters')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, cap):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('invalid bounded metadata: ' + str(path))
    return json.loads(path.read_text())


def telemetry():
    """Only these own-process fields are retained, never the full environment."""
    memory, status = {}, {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, separator, text = line.partition(':')
        if not separator:
            continue
        if key in ('VmSize', 'VmRSS', 'VmHWM'):
            value, unit = text.split()
            if unit != 'kB':
                raise ValueError('unexpected own-process memory unit')
            memory[key + '_bytes'] = int(value) * 1024
        elif key in STATUS_KEYS:
            text = text.strip()
            if len(text) > 1024:
                raise ValueError('oversized own-process status field')
            status[key] = text
    if len(memory) != 3 or any(key not in status for key in ('TracerPid', 'Threads', 'Cpus_allowed_list', 'Seccomp')):
        raise ValueError('incomplete own-process telemetry')
    return memory, status


def stamp():
    return dict(epoch_s=time.time(), monotonic_s=time.monotonic())


def event(name, timing=None):
    timing = stamp() if timing is None else timing
    memory, status = telemetry()
    print(json.dumps(dict(event=name, **timing, memory=memory, status=status),
                     sort_keys=True, allow_nan=False), flush=True)
    return timing


def authenticate():
    """Authenticate the new loader before loading even its pure functions."""
    freeze_path, runner_path = HERE / 'freeze.json', HERE / 'run_diagnostic.py'
    frozen = read_json(freeze_path, RESERVE)
    expected = {str(HERE/name) for name in FILES}
    if (frozen.get('schema') != SCHEMA or frozen.get('required_tests') != 4209
            or frozen.get('required_test_files') != 224
            or frozen.get('worker_api') != 'torch.cuda.is_available_once'
            or type(frozen.get('source_sha256')) is not dict
            or set(frozen['source_sha256']) != expected):
        raise ValueError('new four-source freeze or fixed population differs')
    for name in FILES:
        path = HERE/name
        digest = frozen['source_sha256'][str(path)]
        if (path.is_symlink() or not path.is_file() or type(digest) is not str
                or len(digest) != 64 or sha(path) != digest):
            raise ValueError('new frozen source identity drift before import')
    spec = importlib.util.spec_from_file_location('d247_readonly_runner', runner_path)
    if spec is None or spec.loader is None:
        raise ValueError('new authenticated runner cannot be loaded')
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    runner.check_freeze()
    registered = runner.read_json(RUN/'preregistered.json')
    if (registered.get('freeze_sha256') != sha(freeze_path)
            or registered.get('worker_api') != 'torch.cuda.is_available_once'
            or registered.get('worker_environment') != runner.environment(os.environ)
            or registered.get('cpu_affinity') != [0]
            or registered.get('required_tests') != 4209
            or registered.get('required_test_files') != 224):
        raise ValueError('registered source, population, environment or affinity differs')
    for name in FILES:
        path = HERE/name
        if (registered['source_sha256'].get(str(path)) != frozen['source_sha256'][str(path)]
                or sha(path) != frozen['source_sha256'][str(path)]):
            raise ValueError('new source drift against preregistered identity')
    interpreter = Path(sys.executable).resolve()
    if (interpreter != runner.PYTHON.resolve()
            or sha(interpreter) != registered['source_sha256'].get(str(interpreter))):
        raise ValueError('frozen interpreter differs')
    if 'torch' in sys.modules:
        raise ValueError('torch imported before its authenticated observation')
    torch_spec = importlib.util.find_spec('torch')
    if torch_spec is None or torch_spec.origin is None:
        raise ValueError('torch import origin unavailable')
    origin = Path(torch_spec.origin).resolve()
    if (not origin.is_file() or str(origin) not in registered['gpu_dependency_files']
            or sha(origin) != registered['source_sha256'].get(str(origin))):
        raise ValueError('torch import origin is not the frozen dependency')
    return runner, registered, origin


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
    resource.setrlimit(resource.RLIMIT_FSIZE, (LOG_CAP, LOG_CAP))
    sys.dont_write_bytecode = True
    started = time.monotonic()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tracemalloc.start()
    record = dict(schema='d247_cuda_availability_worker_v1', worker_pid=os.getpid(),
        api='torch.cuda.is_available', api_calls=0, api_returned=False,
        cuda_available=None, import_timing=None, availability_timing=None,
        source_identity_verified_before_torch=False,
        torch_origin_verified_after_import=False,
        initialization_succeeded=None, device_context_measured=None,
        observed_context_bytes=None, combined_physical_gate='unknown',
        gpu_computation_completed=False, source_census_completed=False,
        source_census_qualified=False, native_HZ_admitted=False,
        complete_physical_qualification=False, actual_model_binding_qualified=False,
        new_domain_qualified=False, new_capability_qualified=False,
        tracer_physical_memory_measured=False, numerical_fixture_executed=False,
        driver_cause_inferred=False, host_observations_within_caps=False,
        explicit_cuda_init_calls=0, explicit_device_count_calls=0,
        explicit_tensor_or_kernel_calls=0, diagnostic_solver_calls=0,
        model_forward_calls=0, formal_gain=0, independent_e0_gain=0,
        new_benchmark_solves=0)
    try:
        if RUN.is_symlink() or not RUN.is_dir() or (RUN/'worker.json').exists():
            raise ValueError('new exclusive worker destination unavailable')
        initial_memory, initial_status = telemetry()
        record.update(initial_memory=initial_memory, initial_status=initial_status)
        if (os.sched_getaffinity(0) != {0} or not __debug__
                or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0')):
            raise ValueError('CPU0 and enabled assertions required')
        if (any(os.environ.get(name) != '1' for name in THREAD_KEYS)
                or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU
                or os.environ.get('CUDA_MODULE_LOADING') != 'LAZY'
                or os.environ.get('CUDA_LOG_FILE') != 'stderr'
                or os.environ.get('LD_PRELOAD')
                or os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK') == '1'):
            raise ValueError('fixed one-thread CUDA runtime path differs')
        if (resource.getrlimit(resource.RLIMIT_AS) != (AS_CAP, AS_CAP)
                or resource.getrlimit(resource.RLIMIT_FSIZE) != (LOG_CAP, LOG_CAP)):
            raise ValueError('unchanged AS or file-size cap differs')
        runner, registered, origin = authenticate()
        record.update(source_identity_verified_before_torch=True,
            identity_scope='new four frozen sources, interpreter, torch import origin; supervisor verifies full dependency closure',
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            address_space_bytes=AS_CAP, log_file_cap_bytes=LOG_CAP,
            environment=runner.environment(os.environ), python_version=sys.version,
            torch_import_origin=str(origin),
            torch_import_origin_sha256=registered['source_sha256'][str(origin)])
        import_start = event('before_torch_import')
        try:
            import torch
        except BaseException:
            import_end = stamp()
            record['import_timing'] = dict(start=import_start, end=import_end)
            event('after_torch_import_exception', import_end)
            raise
        import_end = stamp()
        record['import_timing'] = dict(start=import_start, end=import_end)
        event('after_torch_import_before_availability', import_end)
        if (Path(torch.__file__).resolve() != origin
                or sha(origin) != registered['source_sha256'][str(origin)]):
            raise ValueError('loaded torch source differs from authenticated origin')
        record.update(torch_origin_verified_after_import=True,
                      torch_version=str(torch.__version__),
                      torch_cuda_version=str(torch.version.cuda))
        api_start = event('before_availability')
        record['api_calls'] = 1
        try:
            available = torch.cuda.is_available()
            record['api_returned'] = True
            if type(available) is not bool:
                raise TypeError('availability did not return a Boolean')
            record['cuda_available'] = available
        finally:
            api_end = stamp()
            record['availability_timing'] = dict(start=api_start, end=api_end)
            event('after_availability_no_further_cuda_calls' if record['api_returned']
                  else 'after_availability_exception', api_end)
        if not record['cuda_available']:
            raise RuntimeError('CUDA unavailable; sole call consumed, no retry or fallback')
    except BaseException as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        try:
            if not tracemalloc.is_tracing():
                raise ValueError('host allocation telemetry unavailable')
            _, peak = tracemalloc.get_traced_memory()
            metadata = tracemalloc.get_tracemalloc_memory()
            growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024-rss0)
            final_memory, final_status = telemetry()
            record.update(final_memory=final_memory, final_status=final_status,
                rss_highwater_growth_bytes=growth, traced_peak_bytes=peak,
                tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                final_summary_reserve_bytes=RESERVE,
                host_observations_within_caps=(growth+RESERVE <= MEMORY_CAP
                    and peak+metadata+RESERVE <= MEMORY_CAP))
        except BaseException as exc:
            record['host_observations_within_caps'] = False
            record['memory_check_failure'] = str(exc)[:4096]
        record['wall_s'] = time.monotonic()-started
        if not record['host_observations_within_caps'] or record['wall_s'] > 240:
            record.setdefault('failure', dict(type='ResourceError', reason='unchanged host/time gate failed'))
        record['worker_exit'] = 0 if record['cuda_available'] is True and 'failure' not in record else 1
        payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode()+b'\n'
        if len(payload) > RESERVE:
            raise ValueError('worker summary exceeds its reserved boundary')
        if RUN.is_symlink() or not RUN.is_dir():
            raise ValueError('worker summary destination is not the registered directory')
        with (RUN/'worker.json').open('xb') as stream:
            stream.write(payload)
    return record['worker_exit']


if __name__ == '__main__':
    raise SystemExit(main())
