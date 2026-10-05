"""Single-use same-cap CUDA initialization diagnostic; no kernels or fallback."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import tracemalloc
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
D017 = HERE.parent / 'd017_applicability_gpu_20260930'
PRIOR = EXP / 'results/d017_applicability_gpu_20260930_v1'
RUN = EXP / 'results/d020_gpu_init_trace_20260930_v1'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
STRACE = Path('/usr/bin/strace')
GPU = 'GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc'
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
TRACE_CAP = 16 * 1024**2
ANCHORS = {
    D017 / 'gpu_preflight.py': '902437443f6847482d21d0af227b7fc36234868444f11f4dfc580f2abbf89c01',
    D017 / 'PREREG.md': 'a5aa1e483f9180a7a4d6fdead682fc9ebda9ec12da71c6120a97ea90d4dda85d',
    D017 / 'RESULTS.md': '917bb9dfe62d6d262589ad929a7f76fecfc3c30621b551ca3808009c02483e93',
    PRIOR / 'preregistered.json': 'e57f934ca7e8b5161667734c6bd10eeed9b929bb4b3c63e9f8c98c69bfb83cac',
    PRIOR / 'inventory.json': 'ad99d5a92610fb3cd8415e7aeaf53856e0a1b2f7cf4d686493718cde3106415d',
    PRIOR / 'exit.json': 'e6e33752f227a81c12552cc856a9b41c1d40f7098cfa3022d3d5d8f7b5e16f8e',
    STRACE: '28f957c227012de0b18d1bd7fff2d396cb693ea60ed8013be68de071e84b5001',
}
NEW_FILES = ('run_trace.py', 'trace_summary.py', 'test_trace_summary.py', 'PREREG.md')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def trace_limits():
    limits()
    resource.setrlimit(resource.RLIMIT_FSIZE, (TRACE_CAP, TRACE_CAP))


def memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmSize:', 'VmRSS:', 'VmHWM:')):
            key, value, unit = line.split()
            if unit != 'kB':
                raise ValueError('unexpected proc memory unit')
            values[key[:-1] + '_bytes'] = int(value) * 1024
    return values


def event(name):
    print(json.dumps(dict(event=name, epoch_s=time.time(),
                          monotonic_s=time.monotonic(), memory=memory())), flush=True)


def worker():
    if sys.argv[1:] != ['--worker', '--enabled']:
        raise ValueError('explicit worker opt-in required')
    trace_limits()
    start = time.monotonic()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    record = dict(worker_pid=os.getpid(), initialization_succeeded=False,
                  gpu_computation_completed=False, source_census_completed=False,
                  native_HZ_admitted=False, complete_physical_qualification=False,
                  tracer_physical_memory_measured=False,
                  observed_context_bytes=None, combined_physical_gate='unknown',
                  formal_gain=0, initial_memory=memory())
    tracemalloc.start()
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
            raise ValueError('GPU identity differs')
        if len(os.sched_getaffinity(0)) != 1:
            raise ValueError('CPU1 required')
        registered = json.loads((RUN / 'preregistered.json').read_text())
        if any(sha(HERE / name) != registered['source_sha256'][str(HERE / name)]
               for name in NEW_FILES):
            raise ValueError('new source drift before runtime import')
        event('before_torch_import')
        import torch
        event('after_torch_import_before_availability')
        available = bool(torch.cuda.is_available())
        record['cuda_available'] = available
        event('after_availability')
        if not available:
            raise RuntimeError('CUDA unavailable; no retry or fallback')
        torch.cuda.init()
        event('after_cuda_init')
        record['initialization_succeeded'] = True
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
        host_ok = growth + RESERVE <= MEMORY_CAP and peak + metadata + RESERVE <= MEMORY_CAP
        record.update(wall_s=time.monotonic() - start, final_memory=memory(),
                      rss_highwater_growth_bytes=growth, traced_peak_bytes=peak,
                      tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                      host_observations_within_caps=host_ok)
        if not host_ok:
            record.setdefault('failure', dict(type='MemoryError', reason='host gate exceeded'))
        record['worker_exit'] = 0 if record['initialization_succeeded'] and 'failure' not in record else 1
        save('worker.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return record['worker_exit']


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit supervisor opt-in required')
    RUN.mkdir(exist_ok=False)
    start, test_start = time.monotonic(), None
    identities, inputs, helper, frozen = {}, {}, None, None
    record = dict(component_tests_passed=False, trace_launched=False,
                  trace_complete=False, initialization_succeeded=False,
                  gpu_computation_completed=False, source_census_completed=False,
                  native_HZ_admitted=False, complete_physical_qualification=False,
                  formal_gain=0)
    try:
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        if any(sha(path) != digest for path, digest in ANCHORS.items()):
            raise ValueError('frozen authority drift')
        inherited = load(D017 / 'gpu_preflight.py', 'd020_d017_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited.ANCHORS.items()):
            raise ValueError('older authority drift')
        helper = load(inherited.OLD, 'd020_old_readonly_helpers')
        prior = json.loads((PRIOR / 'preregistered.json').read_text())
        done = json.loads((PRIOR / 'exit.json').read_text())
        inventory = json.loads((PRIOR / 'inventory.json').read_text())
        if (done['component_tests_passed'] is not True or done['tests_exit'] != 0
                or done['tests_count'] != 3735 or done['gpu_preflight_passed'] is not False
                or done['source_census_completed'] is not False or not done.get('failure')
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or inventory['count'] != 3735 or inventory['files'] != 165
                or len(prior['tests']) != 165
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D017 population or failed initialization status differs')
        identities.update(prior['source_sha256'])
        inputs.update(prior['input_sha256'])
        for path, digest in ANCHORS.items():
            helper.bind(identities, path, digest)
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        for name in NEW_FILES:
            helper.bind(identities, HERE / name, sha(HERE / name))
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('interpreter drift')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('original source population changed')
        older = json.loads((inherited.PRIOR / 'preregistered.json').read_text())
        if helper.bind_decoder(identities) != older['decoder_dependency_files']:
            raise ValueError('decoder dependency population changed')
        if inherited.gpu_dependencies(helper, identities) != prior['gpu_dependency_files']:
            raise ValueError('GPU dependency population changed')
        frozen = helper.provenance()
        if frozen != prior['provenance'] or frozen['branch'] != 'redu-hz':
            raise ValueError('production provenance drift')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run identity drift')
        test_path = HERE / 'test_trace_summary.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name.startswith('test_')]
        if len(functions) != 4 or any(node.decorator_list or node.args.args for node in functions):
            raise ValueError('exact four plain parser tests required')
        relative = str(test_path.relative_to(ROOT))
        expected = [*inventory['nodeids'], *(relative + '::' + node.name for node in functions)]
        tests = [*prior['tests'], str(test_path)]
        if len(expected) != 3739 or len(set(expected)) != 3739 or len(tests) != 166:
            raise ValueError('complete test population differs')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                   CUDA_CACHE_PATH=str(RUN / 'cuda_cache'), TORCH_HOME=str(RUN / 'torch_home'),
                   XDG_CACHE_HOME=str(RUN / 'xdg_cache'), TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
                   TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=frozen, tests=tests, expected_nodeids=expected, selected_sources=selected,
            required_tests=3739, required_test_files=166, inherited_tests=3735,
            gpu_uuid=GPU, gpu_module_loading='LAZY', strace_sha256=ANCHORS[STRACE],
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, worker_wall_cap_s=240, trace_file_cap_bytes=TRACE_CAP,
            whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            host_memory_cap_bytes=MEMORY_CAP, retained_entry_cap=64_000_000,
            rational_bit_cap=512, no_original_model_decode=True,
            scope='full CPU test gate followed by one CUDA initialization syscall trace',
            prior_failure=done['failure'], caches_relocated_to_new_run=True, formal_gain=0))
        print(json.dumps(dict(event='frozen_before_import', tests=3739, files=166)), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_start = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3739:
            raise ValueError('complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=len(ids), files=166))
        remaining = 60 - (time.monotonic() - test_start)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined budget')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_start,
                      tests_exit=tested.returncode, tests_count=len(ids))
        test_start = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (tested.returncode or sorted(actual) != sorted(expected) or record['test_wall_s'] > 60
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('full component gate failed')
        record['component_tests_passed'] = True
        if helper.drift(identities) or helper.drift(inputs) or helper.provenance() != frozen:
            raise ValueError('identity drift before trace')
        gpu_env = dict(env, CUDA_VISIBLE_DEVICES=GPU, CUDA_MODULE_LOADING='LAZY')
        trace_path = RUN / 'syscalls.trace'
        with trace_path.open('x'):
            pass
        trace_command = [str(STRACE), '--kill-on-exit', '-q', '-f', '-ttt', '-T', '-s', '256',
                         '-e', 'trace=mmap,mremap,brk,ioctl', '-o', str(trace_path),
                         sys.executable, '-B', str(HERE / 'run_trace.py'), '--worker', '--enabled']
        with (RUN / 'worker.log').open('x') as stream:
            process = subprocess.Popen(trace_command, cwd=ROOT, env=gpu_env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                preexec_fn=trace_limits)
            record.update(trace_launched=True, tracer_pid=process.pid)
            try:
                record['tracer_exit'] = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                record['trace_timeout'] = True
                os.killpg(process.pid, signal.SIGKILL)
                record['tracer_exit'] = process.wait()
            finally:
                # Covers early tracer death as well as the normal closed session.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        worker_path = RUN / 'worker.json'
        report = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
        record['worker_record_present'] = bool(report)
        record['initialization_succeeded'] = report.get('initialization_succeeded') is True
        record['worker_failure'] = report.get('failure')
        record['host_observations_within_caps'] = report.get('host_observations_within_caps')
        record['combined_physical_gate'] = report.get('combined_physical_gate', 'unknown')
        parser = load(HERE / 'trace_summary.py', 'd020_trace_summary_runtime')
        trace_bytes = trace_path.stat().st_size
        log_error = (record.get('trace_timeout', False)
                     or 'strace:' in (RUN / 'worker.log').read_text())
        with trace_path.open() as stream:
            summary = parser.summarize(stream, worker_pid=report.get('worker_pid'),
                tracer_exit=record['tracer_exit'], worker_exit=report.get('worker_exit'),
                trace_bytes=trace_bytes, log_error=log_error)
        save('trace_summary.json', summary)
        record['trace_complete'] = summary['trace_complete']
        if not record['trace_complete']:
            raise ValueError('trace incomplete; no absence inference or retry')
        if not record['initialization_succeeded']:
            record['initialization_failure'] = report.get('failure', 'missing failure detail')
    except Exception as exc:
        if test_start is not None:
            record['test_wall_s'] = time.monotonic() - test_start
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        if helper is not None:
            record['source_drift'], record['input_drift'] = helper.drift(identities), helper.drift(inputs)
            try:
                record['provenance_drift'] = frozen is not None and helper.provenance() != frozen
            except Exception as exc:
                record['provenance_drift'] = True
                record['provenance_check_failure'] = str(exc)
            if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
                record['component_tests_passed'] = record['trace_complete'] = False
        record['wall_s'] = time.monotonic() - start
        record['artifacts'] = {str(path.relative_to(RUN)): sha(path)
                               for path in RUN.rglob('*') if path.is_file()}
        record['supervisor_exit'] = 0 if record['component_tests_passed'] and record['trace_complete'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(worker() if '--worker' in sys.argv else main())
