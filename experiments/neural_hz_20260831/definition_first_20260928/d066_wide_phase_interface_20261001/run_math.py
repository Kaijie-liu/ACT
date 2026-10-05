"""Single-use D066 relational-interface mathematics gate.

The four new tests exercise the default-off shared-phase relation interface.
D064 native predicate mathematics remains an inherited, authenticated success,
not a new native-model claim. D057/D047 failed source attempts remain unchanged.
No worker, archived-source evaluation, model or GPU computation is registered.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d066_wide_phase_interface_20261001_v1'
PRIOR = EXP / 'results/d064_native_predicate_binding_20261001_v1'
FAILED_DIRECT = EXP / 'results/d057_triangle_source_census_20260930_v1'
FAILED_CENSUS = EXP / 'results/d047_multiphase_source_census_20260930_v1'
D064 = HERE.parent / 'd064_native_predicate_binding_20261001'
D038 = HERE.parent / 'd038_descendant_census_20260930'
D017 = HERE.parent / 'd017_applicability_gpu_20260930'
D015 = HERE.parent / 'd015_batch_binding_20260928_v2'
OLDER = EXP / 'results/d015_source_shielding_20260928_v2/preregistered.json'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FREEZE = HERE / 'freeze.json'
NEW_FILES = ('PREREG.md', 'THEORY.md', 'phase_interface.py',
             'test_phase_interface.py', 'run_math.py')
TEST_NAMES = ('test_shared_phase_forward_control',
              'test_wide_signed_cached_support',
              'test_interval_actual_phase_and_quantization',
              'test_identity_default_off_and_limits')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    PRIOR / 'preregistered.json': '959ecd1875cb51dee735d7508d0a87c2b40baf63a45d587c386a4e841d31d8be',
    PRIOR / 'inventory.json': 'd66721faa6bcf0f3089143f3af0b73def368e3dbca363cde6ac72bcf95212b6a',
    PRIOR / 'exit.json': '8deebfb4efad74ecb1a183c6734c8fecabe14bf520e68f68bbed6bb1df33a97f',
    D064 / 'freeze.json': '65511f07dd2969377e5c87ab58c83047e92e919dfa690ba5f25eba233c088f75',
    D064 / 'run_math.py': 'd63d2d8a8f4d031e50332f17ad50d0406d92f22728fb0d58675b77f1a426b5d4',
    D064 / 'RESULTS.md': 'aaa5e366e1587a8db12b5826f84cfe35012ef986e849adf4649895fd3d845025',
    D038 / 'freeze.json': '8f24ae5d3e7c674a66e80874e6f607733aeff9d46174434a8a68ce723cf8fc73',
    D038 / 'run_reference.py': 'f4abd1608c70aa308a590307b8f3369eff2ab11b217d54be597c99bb77c0efb5',
    D017 / 'gpu_preflight.py': '902437443f6847482d21d0af227b7fc36234868444f11f4dfc580f2abbf89c01',
    D015 / 'run_v2.py': '8ff9d296ce56b8dd9481d0c8367484b8ff37148db5dfa522062136d9d04e12ce',
    OLDER: '4697d2bbc1b86e7732e825e0e8a3b6d9c595fc275516688d746e738d2bba50cd',
    HERE.parent / 'd065_shared_phase_tables_20261001/THEORY.md':
        '2c6c4eda5908ecdd2e95d7397939b385fa92e05a6ce09791e503ee9edec269a1',
    EXP / 'literature_mechanism_test_20261001/AUDIT.md':
        'e70fdb66223458a5db2c66a06a763db76e22cd7241e25f97b27de43733fcf6ba',
}
SCOPE = ('complete inherited D064 mathematical population plus four relational interface fixtures; '
         'D064 native predicate mathematical success and unchanged D057/D047 source failures preserved; '
         'new relational interface fixture exercised only on mathematical pass; '
         'actual model binding unqualified; no worker, archive evaluation, model, GPU, LP, '
         'source census, shadow, native HZ admission, complete physical qualification or formal gain')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, cap=8 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('missing, linked or oversized metadata: ' + str(path))
    return json.loads(path.read_text())


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def check_freeze():
    frozen = read_json(FREEZE, RESERVE)
    sources = frozen.get('source_sha256')
    if (frozen.get('schema') != 'd066_frozen_v1'
            or frozen.get('required_tests') != 3809 or frozen.get('required_test_files') != 179
            or type(sources) is not dict or set(sources) != {str(HERE / n) for n in NEW_FILES}
            or frozen.get('new_test_names') != list(TEST_NAMES)):
        raise ValueError('frozen five-file/four-test contract differs')
    for path, digest in sources.items():
        if (type(digest) is not str or len(digest) != 64 or Path(path).is_symlink()
                or sha(path) != digest):
            raise ValueError('source differs from pre-execution freeze: ' + path)
    return frozen


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    helper, shared, production, log = None, None, None, None
    identities, inputs = {}, {}
    record = dict(scope=SCOPE, component_tests_passed=False, mathematical_component_gate_passed=False,
        relational_interface_fixture_exercised=False, actual_model_binding_qualified=False,
        worker_stage_registered=False, worker_launched=False, worker_exit=None,
        worker_wall_cap_s=240, host_observations_within_caps=False,
        complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
        native_HZ_admitted=False, gpu_computation_completed=False,
        source_census_completed=False, source_census_qualified=False,
        actual_phase_column_binding_verified=False, formal_gain=0)

    def emit(value):
        line = json.dumps(value, sort_keys=True, allow_nan=False)
        if log is not None:
            log.write(line + '\n')
            log.flush()
        print(line, flush=True)

    try:
        log = (RUN / 'supervisor.log').open('x')
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        tracemalloc.start()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
            CUDA_LOG_FILE='stderr', CUDA_CACHE_PATH=str(RUN / 'cuda_cache'),
            TORCH_HOME=str(RUN / 'torch_home'), XDG_CACHE_HOME=str(RUN / 'xdg_cache'),
            TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        frozen = check_freeze()
        identities.update(frozen['source_sha256'])
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in ANCHORS.items():
            if sha(path) != digest:
                raise ValueError('frozen prior/helper drift: ' + str(path))
            identities[str(path)] = digest
        # Authenticated stdlib-only helpers, without invoking old mains/workers/writers.
        shared = load(D038 / 'run_reference.py', 'd066_d038_readonly_helpers')
        helper = load(D015 / 'run_v2.py', 'd066_d015_readonly_helpers')
        gpu = load(D017 / 'gpu_preflight.py', 'd066_d017_readonly_helpers')
        record['initial_memory'] = shared.memory()
        prior, done = read_json(PRIOR / 'preregistered.json'), read_json(PRIOR / 'exit.json')
        inventory = read_json(PRIOR / 'inventory.json')
        if (done['component_tests_passed'] is not True
                or done['mathematical_component_gate_passed'] is not True
                or done['native_predicate_fixture_exercised'] is not True
                or done['actual_model_binding_qualified'] is not False
                or prior['native_predicate_fixture_registered'] is not True
                or prior['actual_model_binding_qualified'] is not False
                or done['tests_exit'] != 0 or done['tests_count'] != 3805
                or done['test_wall_s'] > 60 or done['all_stages_passed'] is not True
                or done['worker_stage_registered'] is not False or done['worker_launched'] is not False
                or done['worker_exit'] is not None or done['supervisor_exit'] != 0
                or done['source_census_completed'] is not False or done['source_census_qualified'] is not False
                or done['host_observations_within_caps'] is not True or done['formal_gain'] != 0
                or done['native_HZ_admitted'] is not False or done['gpu_computation_completed'] is not False
                or done['actual_phase_column_binding_verified'] is not False
                or done['complete_physical_qualification'] is not False
                or done['candidate_physical_gate_evaluated'] is not False
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or 'failure' in done
                or prior['required_tests'] != 3805 or prior['required_test_files'] != 178
                or len(prior['tests']) != 178 or len(set(prior['tests'])) != 178
                or len(prior['expected_nodeids']) != 3805 or len(set(prior['expected_nodeids'])) != 3805
                or inventory['count'] != 3805 or inventory['files'] != 178
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D064 complete mathematical/native-fixture success contract differs')
        prior_attempt = prior['prior_attempt']
        if (type(prior_attempt) is not dict or prior_attempt != done['prior_attempt']
                or prior.get('prior_census_failure_preserved') is not True
                or done.get('prior_census_failure_preserved') is not True
                or prior_attempt.get('path') != str(FAILED_CENSUS)
                or prior_attempt.get('tests_count') != 3781 or prior_attempt.get('test_files') != 172
                or prior_attempt.get('mathematical_component_gate_passed') is not True
                or prior_attempt.get('all_stages_passed') is not False
                or prior_attempt.get('source_census_completed') is not False
                or prior_attempt.get('source_census_qualified') is not False
                or prior_attempt.get('worker_exit') != 1 or prior_attempt.get('formal_gain') != 0
                or prior_attempt.get('failure') != dict(type='ValueError',
                    reason='source census worker failed; partial evidence retained')
                or prior_attempt.get('worker_failure') != dict(type='BudgetExceeded',
                    reason='whole-work limit exceeded before operation')):
            raise ValueError('nested D047 failed census record differs')
        direct_attempt = prior['direct_prior_attempt']
        if (type(direct_attempt) is not dict or direct_attempt != done['direct_prior_attempt']
                or prior.get('direct_prior_failure_preserved') is not True
                or done.get('direct_prior_failure_preserved') is not True
                or direct_attempt.get('path') != str(FAILED_DIRECT)
                or direct_attempt.get('exit_sha256') != '6c1521af25dfb446e920774cfcbadd5e5ceb1aed4cad831a85bd46a4600d1b9c'
                or direct_attempt.get('tests_count') != 3797 or direct_attempt.get('test_files') != 176
                or direct_attempt.get('mathematical_component_gate_passed') is not True
                or direct_attempt.get('all_stages_passed') is not False
                or direct_attempt.get('source_census_completed') is not False
                or direct_attempt.get('source_census_qualified') is not False
                or direct_attempt.get('worker_exit') != 1 or direct_attempt.get('formal_gain') != 0
                or direct_attempt.get('failure') != dict(type='ValueError',
                    reason='source census worker failed; partial evidence retained')
                or direct_attempt.get('worker_failure') != dict(type='ValueError',
                    reason='evidence budget exhausted before operation')):
            raise ValueError('unchanged D057 failed census record differs')
        inherited_gate = dict(path=str(PRIOR), exit_sha256=sha(PRIOR / 'exit.json'),
            tests_count=3805, test_files=178, mathematical_component_gate_passed=True,
            native_predicate_fixture_exercised=done['native_predicate_fixture_exercised'],
            actual_model_binding_qualified=done['actual_model_binding_qualified'],
            all_stages_passed=True, worker_stage_registered=False, worker_launched=False,
            source_census_qualified=False, native_HZ_admitted=False,
            gpu_computation_completed=False, formal_gain=0)
        record.update(prior_attempt=prior_attempt, prior_census_failure_preserved=True,
            direct_prior_attempt=direct_attempt, direct_prior_failure_preserved=True,
            inherited_mathematical_gate=inherited_gate)
        for path, digest in prior['source_sha256'].items():
            helper.bind(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        if any(path not in identities for path in prior['tests']):
            raise ValueError('inherited test file lacks frozen identity')
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('frozen interpreter differs')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('three original source identities differ')
        decoder = helper.bind_decoder(identities)
        if decoder != read_json(OLDER)['decoder_dependency_files']:
            raise ValueError('frozen decoder population differs')
        dependencies = gpu.gpu_dependencies(helper, identities)
        if dependencies != prior['gpu_dependency_files']:
            raise ValueError('frozen GPU dependencies differ')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run source/input drift')
        test_path = HERE / 'test_phase_interface.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 4 or len({node.name for node in functions}) != 4
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)
                or [node.name for node in functions] != list(TEST_NAMES)):
            raise ValueError('exact four frozen plain new tests required')
        relative = str(test_path.relative_to(ROOT))
        tests = [*prior['tests'], str(test_path)]
        expected = [*inventory['nodeids'], *(relative + '::' + n.name for n in functions)]
        if len(tests) != 179 or len(set(tests)) != 179 or len(expected) != 3809 or len(set(expected)) != 3809:
            raise ValueError('complete 3809/179 population differs')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            selected_sources=selected, decoder_dependency_files=decoder,
            gpu_dependency_files=dependencies, freeze_path=str(FREEZE), freeze_sha256=sha(FREEZE),
            required_tests=3809, required_test_files=179, inherited_tests=3805, inherited_test_files=178,
            new_test_names=list(TEST_NAMES), prior_attempt=prior_attempt, prior_census_failure_preserved=True,
            direct_prior_attempt=direct_attempt, direct_prior_failure_preserved=True,
            inherited_mathematical_gate=inherited_gate, cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=AS_CAP, tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            worker_stage_registered=False, scope=SCOPE, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            evidence_prepaid_work=40_000_000, retained_entry_cap=64_000_000, rational_bit_cap=512,
            caches_relocated_to_new_run=True, cuda_visible_devices='', complete_physical_qualification=False,
            candidate_physical_gate_evaluated=False, source_census_completed=False, source_census_qualified=False,
            relational_interface_fixture_registered=True, actual_model_binding_qualified=False,
            actual_phase_column_binding_verified=False, native_HZ_admitted=False,
            gpu_computation_completed=False, formal_gain=0))
        emit(dict(event='frozen_before_candidate_import', tests=3809, files=179, scope=SCOPE))
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', *tests]
        test_started = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if (collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3809
                or len({n.split('::', 1)[0] for n in ids}) != 179):
            raise ValueError('exact complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=3809, files=179))
        remaining = 60 - (time.monotonic() - test_started)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined 60-second budget')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_started, tests_exit=tested.returncode, tests_count=3809)
        test_started = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if (tested.returncode or record['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))):
            raise ValueError('complete inherited and mathematical component gate failed')
        record['component_tests_passed'] = True
        emit(dict(event='complete_component_pass', tests=3809, files=179, wall_s=record['test_wall_s']))
    except BaseException as exc:
        if test_started is not None:
            record['test_wall_s'] = time.monotonic() - test_started
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
        emit(dict(event='failed', failure=record['failure']))
    finally:
        try:
            record['source_drift'] = [p for p, d in identities.items() if sha(p) != d]
            record['input_drift'] = [p for p, d in inputs.items() if sha(p) != d]
            record['provenance_drift'] = (production is not None
                and (helper is None or helper.provenance() != production))
            if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
                raise ValueError('final source/input/provenance drift')
        except BaseException as exc:
            record['component_tests_passed'] = False
            record['final_identity_check_failure'] = str(exc)[:4096]
            record.setdefault('failure', dict(type=type(exc).__name__, reason='final identity check failed'))
        if log is not None:
            log.close()
            log = None
        record['artifacts'] = {}
        try:
            for path in RUN.rglob('*'):
                if path.is_file():
                    record['artifacts'][str(path.relative_to(RUN))] = sha(path)
        except BaseException as exc:
            record['artifact_sealing_failure'] = str(exc)[:4096]
            record.setdefault('failure', dict(type=type(exc).__name__, reason='artifact sealing incomplete'))
        try:
            if not tracemalloc.is_tracing() or shared is None:
                raise ValueError('supervisor host telemetry unavailable')
            _, peak = tracemalloc.get_traced_memory()
            metadata = tracemalloc.get_tracemalloc_memory()
            growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
            record.update(final_memory=shared.memory(), rss_highwater_growth_bytes=growth,
                traced_peak_bytes=peak, tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                host_observations_within_caps=(growth + RESERVE <= MEMORY_CAP
                    and peak + metadata + RESERVE <= MEMORY_CAP))
        except BaseException as exc:
            record['host_observations_within_caps'] = False
            record['memory_check_failure'] = str(exc)[:4096]
        if not record['host_observations_within_caps']:
            record.setdefault('failure', dict(type='MemoryError', reason='supervisor host gate failed'))
        record.update(wall_s=time.monotonic() - started,
            memory_scope='supervisor only; pytest has AS/CPU/time limits, not full physical qualification')
        record['mathematical_component_gate_passed'] = (record['component_tests_passed']
            and record['host_observations_within_caps'] and 'failure' not in record)
        record['relational_interface_fixture_exercised'] = record['mathematical_component_gate_passed']
        record['all_stages_passed'] = record['mathematical_component_gate_passed']
        record['supervisor_exit'] = 0 if record['all_stages_passed'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())

