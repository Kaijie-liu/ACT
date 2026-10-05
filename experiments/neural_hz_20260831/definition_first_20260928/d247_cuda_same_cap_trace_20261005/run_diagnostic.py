"""D247 once-only same-cap CUDA syscall diagnostic; no GPU numerical fixture."""
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
RUN = EXP / 'results/d247_cuda_same_cap_trace_20261005_v1'
PRIOR = EXP / 'results/d245_mixed_source_component_20261005_v1'
BASE = HERE.parent / 'd245_mixed_source_component_20261005'
DEPENDENCIES = HERE.parent / 'd214_parametric_relation_consumption_20261005/run_math.py'
OLD = HERE.parent / 'd130_import_isolation_20261002'
PARSER = HERE.parent / 'd020_gpu_init_trace_20260930/trace_summary.py'
STRACE = Path('/usr/bin/strace')
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
GPU = 'GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd247_cuda_same_cap_trace_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd247_cuda_same_cap_trace_20261005.collection_contract')
FILES = ('PREREG.md', 'run_diagnostic.py', 'collection_contract.py', 'init_worker.py')
NEW_FILES = FILES
NAMES = ()
NEW_RECORD_FILES = ()
ANCHORS = {
    PRIOR / 'preregistered.json': 'a6aee5aa1feb1ea175cd08f522540af8ba54dabb90e176409a8f00f55758ce82',
    PRIOR / 'inventory.json': 'bfdad3342f98966306bf0d772ae3936cb298221cdd24ce15675f91f7859530e1',
    PRIOR / 'exit.json': '22fdb218cbdd61c671e810117144ca0aa92d712d416311cf718a633fad5a78ed',
    PRIOR / 'summary.json': '6506f09df564c5d6a17c2d48c8a49cb2eadf576ec0478ed2fd369779b44981d4',
    BASE / 'freeze.json': '9ea7e0c6f642688127ee8d6e2abf3dc6c29c0beaf614dc8a5f8596b08682cc95',
    BASE / 'ARCHIVE.sha256': '4186878dabecf25edee40e1cc0d03b2d88290f0e5b4ad227d4df55a81946ecd6',
    BASE / 'CONTRACT.md': '70746b75f8e92e039d17e8f6dc7524d546926cd89d7b8efe0b194fb5d1551697',
    BASE / 'PREREG.md': 'feca4d346ab65ccde9bb2fe90447c5ad5de10b5bf2fe6d2bf95c7e3cf7ea3f1f',
    BASE / 'mixed_relation.py': '53e7b4a8072775d309f9241da46e38a2ce81c371d675698b39b66096278bcbca',
    BASE / 'test_mixed_relation.py': '9b9ade26bb01be96649131132d3a66d9c3006973dfc09d582e5921a1001f92f8',
    BASE / 'run_math.py': '4141cb51d44292e358e7c18662612622659efa6bb357a8739d7238432b836502',
    BASE / 'collection_contract.py': '895808ba303d7baaa7a467d9df02acfcebedce8cac6d53dd9536b7c62a79aef3',
    OLD / 'run_math.py': 'cf7f9f1b767468430bcc3dfac861105f88625e74a4d1a7f0e6beb4f4744bc33d',
    DEPENDENCIES: 'f0f25fe16df7392ed8f407e612af3e51e965177549a832bf4daab01bc4041f42',
    PARSER: '5170d0ad419fa9bdcd7784f65e3ae0aad0e7c412ab3d8cbbbf354dd2d2d9b15c',
    STRACE: '28f957c227012de0b18d1bd7fff2d396cb693ea60ed8013be68de071e84b5001',
}
AS_CAP, MEMORY_CAP, RESERVE, LOG_CAP = 16 * 1024**3, 1024**3, 65536, 16 * 1024**2
TRUE_FLAGS = ('diagnostic_only', 'diagnostic_solver_free',
              'fixed_component_lp_controls_registered', 'worker_stage_registered')
FALSE_FLAGS = ('mathematical_stage_only', 'domain_definition_changed',
    'new_component_solver_free', 'worker_launched', 'source_component_qualified',
    'source_census_completed', 'source_census_qualified', 'actual_model_binding_qualified',
    'actual_phase_column_binding_verified', 'native_HZ_admitted', 'gpu_computation_completed',
    'complete_physical_qualification', 'candidate_physical_gate_evaluated',
    'production_snapshot_imported', 'solver_rescue_registered', 'negative_audit_only',
    'new_set_class', 'new_domain_qualified', 'new_capability_qualified',
    'capability_improvement_claimed')
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_LOG_FILE', 'CUDA_MODULE_LOADING',
    'CUDA_MODULE_DATA_LOADING', 'CUDA_FORCE_PRELOAD_LIBRARIES', 'CUDA_LAUNCH_BLOCKING',
    'CUDA_FORCE_PTX_JIT', 'CUDA_DISABLE_PTX_JIT', 'CUDA_CACHE_DISABLE',
    'PYTORCH_NVML_BASED_CUDA_CHECK', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
    'LD_PRELOAD', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'PYTHONDONTWRITEBYTECODE', 'PYTHONHASHSEED', 'CUDA_CACHE_PATH', 'TORCH_HOME',
    'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TMPDIR')


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked identity: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, cap=8 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('invalid JSON identity: ' + str(path))
    return json.loads(path.read_text())


def read(path):
    return read_json(path)


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load_checked(path, name, identities):
    if sha(path) != identities.get(str(path)):
        raise ValueError('unauthenticated helper: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def trace_limits():
    limits()
    resource.setrlimit(resource.RLIMIT_FSIZE, (LOG_CAP, LOG_CAP))


def environment(env):
    values = {key: env.get(key) for key in ENV_KEYS}
    if (any(value is not None and len(value) > 4096 for value in values.values())
            or sum(6 * len(value or '') + len(key) + 16
                   for key, value in values.items()) > 16384):
        raise ValueError('diagnostic environment exceeds the existing record bound')
    return values


def memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmSize:', 'VmRSS:', 'VmHWM:')):
            key, value, unit = line.split()
            if unit != 'kB':
                raise ValueError('unexpected own-process memory unit')
            values[key[:-1] + '_bytes'] = int(value) * 1024
    if len(values) != 3:
        raise ValueError('incomplete supervisor process-memory telemetry')
    return values


def check_freeze():
    frozen = read_json(FREEZE, RESERVE)
    if (frozen.get('schema') != SCHEMA
            or frozen.get('required_tests') != 4209
            or frozen.get('required_test_files') != 224
            or frozen.get('new_test_names') != []
            or frozen.get('new_evidence_files') != []
            or frozen.get('worker_api') != 'torch.cuda.is_available_once'
            or any(frozen.get(key) is not True for key in TRUE_FLAGS)
            or any(frozen.get(key) is not False for key in FALSE_FLAGS)
            or type(frozen.get('source_sha256')) is not dict
            or set(frozen['source_sha256']) != {str(HERE / name) for name in FILES}):
        raise ValueError('frozen four-source D247 diagnostic contract differs')
    for path, digest in frozen['source_sha256'].items():
        if type(digest) is not str or len(digest) != 64 or sha(path) != digest:
            raise ValueError('new source differs from its pre-execution freeze')
    return frozen


def trace_once(identities, worker_env, result):
    """One owned child tree; retain a conservative summary even when CUDA fails."""
    trace_path = RUN / 'syscalls.trace'
    with trace_path.open('x'):
        pass
    command = [str(STRACE), '--kill-on-exit', '-q', '-f', '-ttt', '-T', '-s', '256',
        '-e', 'trace=mmap,mremap,brk,ioctl', '-o', str(trace_path),
        sys.executable, '-B', str(HERE / 'init_worker.py'), '--enabled']
    process = None
    worker_started = time.monotonic()
    try:
        with (RUN / 'worker.stdout.log').open('xb') as out, (RUN / 'worker.stderr.log').open('xb') as err:
            process = subprocess.Popen(command, cwd=ROOT, env=worker_env,
                stdout=out, stderr=err, start_new_session=True, preexec_fn=trace_limits)
            result.update(worker_launched=True, trace_launched=True, tracer_pid=process.pid)
            try:
                remaining = 240 - (time.monotonic() - worker_started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 240)
                result['tracer_exit'] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                result['trace_timeout'] = True
                os.killpg(process.pid, signal.SIGKILL)
                result['tracer_exit'] = process.wait(timeout=5)
    finally:
        # Never touch an unrelated PID: this group was created by this Popen.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait(timeout=5)
        result['worker_wall_s'] = time.monotonic() - worker_started
    report_path = RUN / 'worker.json'
    result['worker_record_present'] = report_path.is_file()
    report = read_json(report_path, RESERVE) if report_path.is_file() else {}
    result['worker_report'] = report
    result['worker_exit'] = report.get('worker_exit')
    trace_bytes = trace_path.stat().st_size
    incomplete_log = any((RUN / name).stat().st_size >= LOG_CAP
                         for name in ('worker.stdout.log', 'worker.stderr.log'))
    # Presence of a tracer warning prevents absence-based interpretation.
    tracer_warning = 'strace:' in (RUN / 'worker.stderr.log').read_text(errors='replace')
    parser = load_checked(PARSER, '_d247_authenticated_d020_trace_parser', identities)
    with trace_path.open() as stream:
        summary = parser.summarize(stream, worker_pid=report.get('worker_pid'),
            tracer_exit=result.get('tracer_exit'), trace_bytes=trace_bytes,
            worker_exit=report.get('worker_exit'),
            log_error=(incomplete_log or tracer_warning or result.get('trace_timeout', False)
                       or result['worker_wall_s'] > 240))
    save('trace_summary.json', summary)
    result['trace_complete'] = summary['trace_complete']
    if not result['trace_complete']:
        raise ValueError('trace incomplete; positive observations retained, no retry or absence inference')
    for name in ('complete_physical_qualification', 'gpu_computation_completed',
                 'native_HZ_admitted', 'source_census_completed',
                 'source_census_qualified', 'actual_model_binding_qualified',
                 'new_domain_qualified', 'new_capability_qualified',
                 'tracer_physical_memory_measured',
                 'numerical_fixture_executed', 'driver_cause_inferred'):
        if report.get(name) is not False:
            raise ValueError('worker claimed an unregistered qualification')
    if (report.get('schema') != 'd247_cuda_availability_worker_v1'
            or report.get('formal_gain') != 0 or report.get('new_benchmark_solves') != 0
            or report.get('independent_e0_gain') != 0
            or report.get('diagnostic_solver_calls') != 0 or report.get('model_forward_calls') != 0
            or any(report.get(key) != 0 for key in ('explicit_cuda_init_calls',
                   'explicit_device_count_calls', 'explicit_tensor_or_kernel_calls'))
            or report.get('source_identity_verified_before_torch') is not True
            or report.get('torch_origin_verified_after_import') is not True
            or report.get('initialization_succeeded') is not None
            or report.get('observed_context_bytes') is not None
            or report.get('device_context_measured') is not None
            or report.get('combined_physical_gate') != 'unknown'
            or report.get('api') != 'torch.cuda.is_available'
            or report.get('api_calls') != 1 or report.get('api_returned') is not True
            or type(report.get('cuda_available')) is not bool):
        raise ValueError('sole availability call incomplete or diagnostic boundary changed')
    metrics = ('rss_highwater_growth_bytes', 'traced_peak_bytes',
               'tracer_metadata_bytes', 'summary_reserve_bytes', 'final_summary_reserve_bytes')
    if (any(type(report.get(key)) is not int or report[key] < 0 for key in metrics)
            or report['summary_reserve_bytes'] != RESERVE
            or report['final_summary_reserve_bytes'] != RESERVE
            or report['rss_highwater_growth_bytes'] + RESERVE > MEMORY_CAP
            or report['traced_peak_bytes'] + report['tracer_metadata_bytes'] + RESERVE > MEMORY_CAP
            or report.get('host_observations_within_caps') is not True
            or type(report.get('wall_s')) not in (int, float)
            or not math.isfinite(report['wall_s']) or not 0 <= report['wall_s'] <= 240
            or report.get('address_space_bytes') != AS_CAP
            or report.get('log_file_cap_bytes') != LOG_CAP
            or report.get('cpu_affinity') != [0]
            or report.get('environment') != environment(worker_env)
            or type(report.get('worker_pid')) is not int or report['worker_pid'] <= 0
            or report.get('worker_exit') != result.get('tracer_exit')
            or report.get('worker_exit') not in (0, 1)):
        raise ValueError('worker resource/environment receipt failed independent checks')
    result['availability_confirmed'] = (report['cuda_available'] is True
        and report['worker_exit'] == 0 and 'failure' not in report)
    if report['cuda_available'] and not result['availability_confirmed']:
        raise ValueError('available observation was not a normally completed worker')
    if not report['cuda_available'] and (report['worker_exit'] != 1
            or report.get('failure') != dict(type='RuntimeError',
                reason='CUDA unavailable; sole call consumed, no retry or fallback')):
        raise ValueError('unavailable observation has an additional worker failure')
    # Only a provisional diagnostic result until post-run identities/host checks.
    result['diagnostic_completed'] = True


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs = {}, {}
    old = inherited = dependency_helper = helper = gpu = production = closure = prior = None
    manifest_digest = None
    result = dict(schema=SCHEMA, new_tests=0, formal_gain=0, independent_e0_gain=0, new_benchmark_solves=0,
        component_tests_passed=False, mathematical_component_gate_passed=False,
        inventory_validated_before_execution=False, diagnostic_completed=False,
        trace_launched=False, trace_complete=False, availability_confirmed=False,
        observed_context_bytes=None, device_context_measured=None, combined_physical_gate='unknown')
    result.update({key: True for key in TRUE_FLAGS})
    result.update({key: False for key in FALSE_FLAGS})
    try:
        limits()
        if 0 not in os.sched_getaffinity(0):
            raise ValueError('required CPU 0 is unavailable')
        os.sched_setaffinity(0, {0})
        sys.dont_write_bytecode = True
        tracemalloc.start()
        result['initial_memory'] = memory()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        if os.environ.get('LD_PRELOAD') or os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK') == '1':
            raise ValueError('unregistered preload or NVML availability path')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='', CUDA_LOG_FILE='stderr', CUDA_CACHE_PATH=str(RUN / 'cuda_cache'),
            TORCH_HOME=str(RUN / 'torch_home'), XDG_CACHE_HOME=str(RUN / 'xdg_cache'),
            TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        sys.path.insert(0, str(ROOT))
        frozen = check_freeze()
        identities.update(frozen['source_sha256'])
        identities.update({str(path): digest for path, digest in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d247_authenticated_d130', identities)
        inherited = load_checked(BASE / 'run_math.py', '_d247_authenticated_d245', identities)
        dependency_helper = load_checked(DEPENDENCIES, '_d247_authenticated_dependencies', identities)
        prior, done, inventory = (read(PRIOR / name) for name in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        prior_freeze = read(BASE / 'freeze.json')
        if (prior.get('schema') != 'd245_mixed_source_component_v1'
                or prior.get('required_tests') != 4209 or prior.get('required_test_files') != 224
                or len(prior.get('source_sha256', {})) != 7752
                or len(prior.get('input_sha256', {})) != 14
                or prior.get('cpu_affinity') != [0]
                or any(prior.get(key) is not True for key in inherited.TRUE_FLAGS)
                or any(prior.get(key) is not False for key in inherited.FALSE_FLAGS)
                or any(done.get(key) is not True for key in ('component_tests_passed',
                    'mathematical_component_gate_passed', 'host_observations_within_caps',
                    'inventory_validated_before_execution', 'all_registered_stages_passed'))
                or any(done.get(key) is not False for key in inherited.FALSE_FLAGS)
                or done.get('negative_audit_only') is not False
                or done.get('formal_gain') != 0 or done.get('new_benchmark_solves') != 0
                or done.get('tests_exit') != 0 or done.get('supervisor_exit') != 0
                or done.get('tests_count') != 4209 or done.get('test_files') != 224
                or not 0 <= done.get('test_wall_s', 61) <= 60 or 'failure' in done
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 4209 or inventory.get('files') != 224
                or inventory.get('manifest_sha256') != ANCHORS[PRIOR / 'preregistered.json']
                or inventory.get('validated_before_execution') is not True
                or prior_freeze.get('source_sha256') != {
                    str(BASE / name): ANCHORS[BASE / name] for name in inherited.FILES}):
            raise ValueError('D245 complete mathematical receipt differs')
        for path, digest in prior['source_sha256'].items():
            old.merge_identity(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = PRIOR / name
            if path.resolve() != path or not path.is_relative_to(PRIOR):
                raise ValueError('historical artifact escapes original directory')
            old.merge_identity(identities, str(path), digest)
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('inherited identity drift: ' + path)
        if (Path(sys.executable).resolve() != old.PYTHON.resolve()
                or sha(Path(sys.executable).resolve()) != identities[str(old.PYTHON.resolve())]):
            raise ValueError('interpreter differs')
        closure = old.project_closure(identities, prior.get('project_import_closure'))
        helper = load_checked(old.HELPER, '_d247_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d247_authenticated_d017_dependencies', identities)
        dependency_helper.dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance'] or list(os.sched_getaffinity(0)) != [0]:
            raise ValueError('production provenance or CPU affinity differs')
        tests, expected = list(prior['tests']), list(prior['expected_nodeids'])
        if (len(tests) != 224 or len(set(tests)) != 224
                or len(expected) != 4209 or len(set(expected)) != 4209):
            raise ValueError('inherited ordered population differs')
        if (len(set(tests)) != 224 or len(set(expected)) != 4209
                or any(path not in identities for path in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        relocations = {}
        for key in ('inherited_component_evidence_relocation', 'inherited_D207_evidence_relocation',
                    'inherited_D208_evidence_relocation', 'inherited_D209_evidence_relocation',
                    'inherited_D214_evidence_relocation', 'inherited_D228_evidence_relocation',
                    'inherited_D229_evidence_relocation', 'inherited_D230_evidence_relocation',
                    'inherited_D231_evidence_relocation', 'inherited_D240_evidence_relocation',
                    'inherited_D243_evidence_relocation'):
            relocation = dict(prior[key])
            relocation['relocated_run'] = str(RUN / Path(relocation['relocated_run']).name)
            relocations[key] = relocation
        relocations['inherited_D245_evidence_relocation'] = dict(
            module_path=str(BASE / 'test_mixed_relation.py'), source_sha256=ANCHORS[BASE / 'test_mixed_relation.py'],
            function_name='_record_file', function_firstlineno=56,
            original_run=str(PRIOR), relocated_run=str(RUN / 'inherited_d245_controls'),
            allowed_filenames=['summary.json'], mechanism='module_local_record_function_only')
        worker_env = dict(env, CUDA_VISIBLE_DEVICES=GPU, CUDA_MODULE_LOADING='LAZY',
                          CUDA_LOG_FILE='stderr')
        # Historical candidate definitions remain provenance, not current claims.
        manifest = dict(prior)
        manifest.update(schema=SCHEMA, source_sha256=identities, input_sha256=inputs,
            provenance=production, project_import_closure=closure, tests=tests,
            expected_nodeids=expected, required_tests=4209, required_test_files=224,
            inherited_tests=4209, inherited_test_files=224, new_test_files=0,
            new_test_names=[], new_evidence_files=[],
            inherited_test_population_unchanged=True,
            inherited_D245_receipt=dict(path=str(PRIOR),
                manifest_sha256=ANCHORS[PRIOR / 'preregistered.json'],
                inventory_sha256=ANCHORS[PRIOR / 'inventory.json'],
                exit_sha256=ANCHORS[PRIOR / 'exit.json'],
                mathematical_component_gate_passed=True, qualification_transferred=False),
            preserved_D245_operator_definition=prior['operator_definition'],
            preserved_D245_candidate_semantic_definition=prior['candidate_semantic_definition'],
            last_successful_candidate_semantic_definition=prior['candidate_semantic_definition'],
            candidate_semantic_definition=None, operator_definition=None,
            freeze_sha256=sha(FREEZE), collection_plugin=PLUGIN, pytest_import_mode='importlib',
            pytest_plugin_autoload=False, same_process_collection_gate=True,
            single_pytest_process=True, cpu_affinity=[0], address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, cuda_visible_devices='', formal_gain=0,
            independent_e0_gain=0, new_benchmark_solves=0,
            worker_api='torch.cuda.is_available_once', worker_wall_cap_s=240,
            worker_environment=environment(worker_env), tests_environment=environment(env),
            gpu_uuid=GPU, gpu_module_loading='LAZY', log_file_cap_bytes=LOG_CAP,
            trace_file_cap_bytes=LOG_CAP, strace_sha256=ANCHORS[STRACE],
            trace_parser_sha256=ANCHORS[PARSER], no_original_model_decode=True,
            whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            retained_entry_cap=64_000_000, rational_bit_cap=512,
            scope='unchanged full mathematical replay followed by one CUDA availability syscall trace',
            observed_context_bytes=None, device_context_measured=None,
            combined_physical_gate='unknown', **relocations)
        manifest.update({key: True for key in TRUE_FLAGS})
        manifest.update({key: False for key in FALSE_FLAGS})
        save('preregistered.json', manifest)
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D247_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
            '--tb=short', '-p', 'no:cacheprovider', '-p', PLUGIN,
            '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic() - test_started,
                      tests_count=4209, test_files=224)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 4209
                or checked.get('files') != 224 or checked.get('manifest_sha256') != manifest_digest
                or checked.get('validated_before_execution') is not True
                or any(checked.get(key) != value for key, value in relocations.items())
                or sha(RUN / 'preregistered.json') != manifest_digest):
            raise ValueError('pre-execution inventory contract differs')
        result['inventory_validated_before_execution'] = True
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (process.returncode != 0 or result['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('complete mathematical test gate failed')
        if any((RUN / name).is_symlink() or not (RUN / name).is_file() for name in NEW_RECORD_FILES):
            raise ValueError('complete inherited mathematical evidence missing')
        result['component_tests_passed'] = True
        result['mathematical_component_gate_passed'] = True
        if helper.drift(identities) or helper.drift(inputs) or helper.provenance() != production:
            raise ValueError('identity drift before the sole traced worker')
        trace_once(identities, worker_env, result)
    except BaseException as exc:
        if test_started is not None:
            result['test_wall_s'] = time.monotonic() - test_started
        result['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096], stage=('diagnostic' if result['component_tests_passed'] else 'mathematical'))
    finally:
        try:
            if closure is not None and old.project_closure(identities, closure) != closure:
                raise ValueError('post-execution project closure differs')
            if helper is not None and gpu is not None and prior is not None:
                dependency_helper.dependencies(helper, gpu, identities, inputs, prior)
                if list(os.sched_getaffinity(0)) != prior['cpu_affinity']:
                    raise ValueError('post-execution CPU affinity differs')
            result['source_drift'] = [path for path, digest in identities.items() if sha(path) != digest]
            result['input_drift'] = [path for path, digest in inputs.items() if sha(path) != digest]
            result['provenance_drift'] = production is not None and helper.provenance() != production
            if (result['source_drift'] or result['input_drift'] or result['provenance_drift']
                    or (manifest_digest is not None and sha(RUN / 'preregistered.json') != manifest_digest)):
                raise ValueError('post-execution identity drift')
        except BaseException as exc:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        try:
            result['artifacts'] = {str(path.relative_to(RUN)): sha(path)
                                   for path in RUN.rglob('*') if path.is_file()}
        except BaseException as exc:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type=type(exc).__name__,
                reason='artifact sealing incomplete: ' + str(exc)[:4096]))
        try:
            if old is None:
                raise ValueError('authenticated telemetry helper unavailable')
            old.host_observations(result, rss0)
            result['final_memory'] = memory()
        except BaseException as exc:
            result['host_observations_within_caps'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        if not result['host_observations_within_caps']:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type='MemoryError', reason='supervisor memory gate failed'))
        if 'failure' in result or not result['host_observations_within_caps']:
            result['diagnostic_completed'] = False
        passed = (result['mathematical_component_gate_passed']
                  and result['diagnostic_completed'] and 'failure' not in result)
        result.update(all_registered_stages_passed=passed, supervisor_exit=0 if passed else 1,
            wall_s=time.monotonic() - started,
            memory_scope='supervisor and worker separately observed; pytest AS/CPU/time only; '
                'tracer/context combined memory unmeasured, no full physical qualification')
        save('exit.json', result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
