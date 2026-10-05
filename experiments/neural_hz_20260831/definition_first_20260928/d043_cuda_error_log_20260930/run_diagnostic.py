"""D043 single-use driver-log diagnostic; old helpers are read-only imports."""
import hashlib
import importlib.util
import json
import math
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
RUN = EXP / 'results/d043_cuda_error_log_20260930_v1'
PRIOR = EXP / 'results/d038_descendant_census_20260930_v1'
D038 = HERE.parent / 'd038_descendant_census_20260930'
D017 = HERE.parent / 'd017_applicability_gpu_20260930'
D015 = HERE.parent / 'd015_batch_binding_20260928_v2'
OLDER = EXP / 'results/d015_source_shielding_20260928_v2/preregistered.json'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FREEZE = HERE / 'freeze.json'
NEW_FILES = ('PREREG.md', 'run_diagnostic.py', 'init_worker.py')
GPU = 'GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc'
AS_CAP, MEMORY_CAP, RESERVE, LOG_CAP = 16 * 1024**3, 1024**3, 65536, 16 * 1024**2
INVENTORY_SHA = '62cba6b99355a0c72ab878a76f7500418f776f130204dd00ead264dbba4d4828'
ANCHORS = {
    PRIOR / 'preregistered.json': '06252697d664a213c7967a6e726d99fd6217dd5d7311e59ee202b38f4083ff04',
    PRIOR / 'inventory.json': INVENTORY_SHA,
    PRIOR / 'exit.json': 'b1f4c6555e538a52c84c0fd78a918fdc2c65677cf1d515ec9e771b97d8e700cd',
    D038 / 'freeze.json': '8f24ae5d3e7c674a66e80874e6f607733aeff9d46174434a8a68ce723cf8fc73',
    D038 / 'run_reference.py': 'f4abd1608c70aa308a590307b8f3369eff2ab11b217d54be597c99bb77c0efb5',
    D017 / 'gpu_preflight.py': '902437443f6847482d21d0af227b7fc36234868444f11f4dfc580f2abbf89c01',
    D015 / 'run_v2.py': '8ff9d296ce56b8dd9481d0c8367484b8ff37148db5dfa522062136d9d04e12ce',
    OLDER: '4697d2bbc1b86e7732e825e0e8a3b6d9c595fc275516688d746e738d2bba50cd',
}
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_LOG_FILE', 'CUDA_MODULE_LOADING',
    'CUDA_MODULE_DATA_LOADING', 'CUDA_FORCE_PRELOAD_LIBRARIES', 'CUDA_LAUNCH_BLOCKING',
    'CUDA_FORCE_PTX_JIT', 'CUDA_DISABLE_PTX_JIT', 'CUDA_CACHE_DISABLE',
    'PYTORCH_NVML_BASED_CUDA_CHECK', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
    'LD_PRELOAD', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'PYTHONDONTWRITEBYTECODE', 'PYTHONHASHSEED', 'CUDA_CACHE_PATH', 'TORCH_HOME',
    'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TMPDIR')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path, cap=8 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('missing, linked or oversized metadata: ' + str(path))
    return json.loads(path.read_text())


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


def limits(log_limit=False):
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
    if log_limit:
        resource.setrlimit(resource.RLIMIT_FSIZE, (LOG_CAP, LOG_CAP))


def environment(env):
    result = {key: env.get(key) for key in ENV_KEYS}
    if (any(value is not None and len(value) > 4096 for value in result.values())
            or sum(6 * len(value or '') + len(key) + 16 for key, value in result.items()) > 16384):
        raise ValueError('diagnostic environment value too large')
    return result


def check_freeze():
    frozen = read_json(FREEZE, RESERVE)
    paths = {str(HERE / name) for name in NEW_FILES}
    if (frozen.get('schema') != 'd043_frozen_v1'
            or frozen.get('required_tests') != 3759 or frozen.get('required_test_files') != 169
            or frozen.get('inventory_sha256') != INVENTORY_SHA
            or frozen.get('worker_api') != 'torch.cuda.is_available_once'
            or type(frozen.get('source_sha256')) is not dict
            or set(frozen['source_sha256']) != paths):
        raise ValueError('new three-file/fixed-population freeze differs')
    for path, digest in frozen['source_sha256'].items():
        if (type(digest) is not str or len(digest) != 64 or Path(path).is_symlink()
                or sha(path) != digest):
            raise ValueError('new source differs from pre-execution freeze')
    return frozen


def host_record(rss0):
    if not tracemalloc.is_tracing():
        raise ValueError('host allocation telemetry unavailable')
    _, peak = tracemalloc.get_traced_memory()
    metadata = tracemalloc.get_tracemalloc_memory()
    growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
    return dict(rss_highwater_growth_bytes=growth, traced_peak_bytes=peak,
                tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                host_observations_within_caps=(growth + RESERVE <= MEMORY_CAP
                    and peak + metadata + RESERVE <= MEMORY_CAP))


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    helper, shared, production, process = None, None, None, None
    identities, inputs = {}, {}
    record = dict(component_tests_passed=False, worker_launched=False,
        diagnostic_completed=False, availability_confirmed=False,
        host_observations_within_caps=False, complete_physical_qualification=False,
        gpu_computation_completed=False, native_HZ_admitted=False,
        source_census_completed=False, source_census_qualified=False,
        observed_context_bytes=None, combined_physical_gate='unknown', formal_gain=0)
    try:
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        tracemalloc.start()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        if os.environ.get('LD_PRELOAD') or os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK') == '1':
            raise ValueError('unregistered preload or NVML availability path; no execution')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
            CUDA_LOG_FILE='stderr',
            CUDA_CACHE_PATH=str(RUN / 'cuda_cache'), TORCH_HOME=str(RUN / 'torch_home'),
            XDG_CACHE_HOME=str(RUN / 'xdg_cache'), TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        prefreeze = check_freeze()
        identities.update(prefreeze['source_sha256'])
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in ANCHORS.items():
            if sha(path) != digest:
                raise ValueError('frozen old authority drift: ' + str(path))
            identities[str(path)] = digest
        shared = load(D038 / 'run_reference.py', 'd043_d038_readonly_helpers')
        record['initial_memory'] = shared.memory()
        helper = load(D015 / 'run_v2.py', 'd043_d015_readonly_helpers')
        gpu = load(D017 / 'gpu_preflight.py', 'd043_d017_readonly_helpers')
        prior, done = read_json(PRIOR / 'preregistered.json'), read_json(PRIOR / 'exit.json')
        inventory = read_json(PRIOR / 'inventory.json')
        tests, expected = prior['tests'], prior['expected_nodeids']
        if (done['component_tests_passed'] is not True or done['tests_exit'] != 0
                or done['tests_count'] != 3759 or done['all_stages_passed'] is not True
                or done['host_observations_within_caps'] is not True or done['formal_gain'] != 0
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or prior['required_tests'] != 3759 or prior['required_test_files'] != 169
                or len(tests) != 169 or len(set(tests)) != 169
                or len(expected) != 3759 or len(set(expected)) != 3759
                or inventory['count'] != 3759 or inventory['files'] != 169
                or sorted(inventory['nodeids']) != sorted(expected)):
            raise ValueError('D038 passed complete population differs')
        for path, digest in prior['source_sha256'].items():
            helper.bind(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        if any(path not in identities for path in tests):
            raise ValueError('test source identity absent')
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('frozen interpreter differs')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('three original read-only sources differ')
        decoder = helper.bind_decoder(identities)
        if decoder != read_json(OLDER)['decoder_dependency_files']:
            raise ValueError('frozen decoder population differs')
        dependencies = gpu.gpu_dependencies(helper, identities)
        if dependencies != prior['gpu_dependency_files']:
            raise ValueError('frozen CUDA dependency population differs')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run source/input drift')
        worker_env = dict(env, CUDA_VISIBLE_DEVICES=GPU, CUDA_MODULE_LOADING='LAZY',
                          CUDA_LOG_FILE='stderr')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            selected_sources=selected, decoder_dependency_files=decoder,
            gpu_dependency_files=dependencies, freeze_sha256=sha(FREEZE),
            required_tests=3759, required_test_files=169, inventory_sha256=INVENTORY_SHA,
            worker_api='torch.cuda.is_available_once', worker_environment=environment(worker_env),
            tests_environment=environment(env), environment_comparison_to_D017='not fully recorded',
            address_space_bytes=AS_CAP, host_memory_cap_bytes=MEMORY_CAP,
            cpu_affinity=list(os.sched_getaffinity(0)), tests_combined_wall_cap_s=60,
            worker_wall_cap_s=240, log_file_cap_bytes=LOG_CAP, summary_reserve_bytes=RESERVE,
            whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            evidence_prepaid_work=40_000_000, retained_entry_cap=64_000_000,
            rational_bit_cap=512, complete_physical_qualification=False, formal_gain=0))
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_started = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if (collected.returncode or sorted(ids) != sorted(expected)
                or len(set(ids)) != 3759 or len({node.split('::', 1)[0] for node in ids}) != 169):
            raise ValueError('exact collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=3759, files=169))
        remaining = 60 - (time.monotonic() - test_started)
        if remaining <= 0:
            raise TimeoutError('collection exhausted the combined 60 seconds')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_started,
                      tests_exit=tested.returncode, tests_count=3759)
        test_started = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (tested.returncode or record['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('full inherited component gate failed')
        record['component_tests_passed'] = True
        if helper.drift(identities) or helper.drift(inputs) or helper.provenance() != production:
            raise ValueError('identity drift before the sole worker')
        worker_started = time.monotonic()
        with (RUN / 'worker.stdout.log').open('xb') as out, (RUN / 'worker.stderr.log').open('xb') as err:
            process = subprocess.Popen([sys.executable, '-B', str(HERE / 'init_worker.py'), '--enabled'],
                cwd=ROOT, env=worker_env, stdout=out, stderr=err, start_new_session=True,
                preexec_fn=lambda: limits(True))
            record['worker_launched'] = True
            record['worker_exit'] = process.wait(timeout=max(0, 240 - (time.monotonic() - worker_started)))
        record['worker_wall_s'] = time.monotonic() - worker_started
        if record['worker_wall_s'] > 240:
            raise TimeoutError('worker exhausted its unchanged 240 seconds')
        for name in ('worker.stdout.log', 'worker.stderr.log'):
            if (RUN / name).stat().st_size >= LOG_CAP:
                raise ValueError('log cap reached; capture is incomplete')
        report = read_json(RUN / 'worker.json', RESERVE)
        record['worker_report'] = report
        for name in ('complete_physical_qualification', 'gpu_computation_completed',
                     'native_HZ_admitted', 'source_census_completed'):
            if report.get(name) is not False:
                raise ValueError('worker claimed an unregistered qualification')
        if (report.get('formal_gain') != 0 or report.get('observed_context_bytes') is not None
                or report.get('combined_physical_gate') != 'unknown'
                or report.get('api_calls') != 1 or report.get('api_returned') is not True):
            raise ValueError('sole availability call incomplete; no inferred CUDA cause')
        record['diagnostic_completed'] = True
        record['availability_confirmed'] = report.get('cuda_available') is True
        metrics = ('rss_highwater_growth_bytes', 'traced_peak_bytes',
                   'tracer_metadata_bytes', 'summary_reserve_bytes', 'final_summary_reserve_bytes')
        if (any(type(report.get(key)) is not int or report[key] < 0 for key in metrics)
                or report['summary_reserve_bytes'] != RESERVE
                or report['final_summary_reserve_bytes'] != RESERVE
                or report['rss_highwater_growth_bytes'] + RESERVE > MEMORY_CAP
                or report['traced_peak_bytes'] + report['tracer_metadata_bytes'] + RESERVE > MEMORY_CAP
                or type(report.get('wall_s')) not in (int, float)
                or not math.isfinite(report['wall_s']) or not 0 <= report['wall_s'] <= 240
                or report.get('address_space_bytes') != AS_CAP
                or report.get('cpu_affinity') != list(os.sched_getaffinity(0))
                or report.get('environment') != environment(worker_env)
                or report.get('worker_exit') != record['worker_exit']):
            raise ValueError('worker resource/environment evidence failed independent checks')
        if (record['worker_exit'] != 0 or not record['availability_confirmed']
                or report.get('host_observations_within_caps') is not True):
            raise ValueError('availability or worker host gate failed; evidence retained')
    except BaseException as exc:
        if test_started is not None:
            record['test_wall_s'] = time.monotonic() - test_started
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as exc:
                record['worker_cleanup_failure'] = str(exc)[:4096]
            record['availability_confirmed'] = False
        try:
            record['source_drift'] = [p for p, d in identities.items() if sha(p) != d]
            record['input_drift'] = [p for p, d in inputs.items() if sha(p) != d]
            record['provenance_drift'] = (production is not None
                and (helper is None or helper.provenance() != production))
            if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
                raise ValueError('final source/input/provenance drift')
        except BaseException as exc:
            record['component_tests_passed'] = record['availability_confirmed'] = False
            record['final_identity_check_failure'] = str(exc)[:4096]
            record.setdefault('failure', dict(type=type(exc).__name__, reason='final identity check failed'))
        record['artifacts'] = {}
        try:
            for path in RUN.rglob('*'):
                if path.is_file():
                    record['artifacts'][str(path.relative_to(RUN))] = sha(path)
        except BaseException as exc:
            record['failure'] = dict(type=type(exc).__name__, reason='artifact sealing incomplete')
        try:
            record.update(host_record(rss0))
            if shared is not None:
                record['final_memory'] = shared.memory()
        except BaseException as exc:
            record['host_observations_within_caps'] = False
            record['memory_check_failure'] = str(exc)[:4096]
        record.update(wall_s=time.monotonic() - started,
            worker_record_present=(RUN / 'worker.json').is_file(),
            memory_scope='supervisor and worker separately; no combined physical qualification')
        record['all_stages_passed'] = (record['component_tests_passed']
            and record['availability_confirmed'] and record['host_observations_within_caps']
            and 'failure' not in record)
        record['supervisor_exit'] = 0 if record['all_stages_passed'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
