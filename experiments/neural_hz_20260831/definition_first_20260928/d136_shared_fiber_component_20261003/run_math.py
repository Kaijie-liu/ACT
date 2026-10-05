"""Once-only D136 mathematical gate; no source worker or benchmark verdicts."""
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
RUN = EXP / 'results/d136_shared_fiber_component_20261003_v1'
PRIOR = EXP / 'results/d130_import_isolation_20261002_v1'
OLD = HERE.parent / 'd130_import_isolation_20261002'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd136_shared_fiber_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd136_shared_fiber_component_20261003.collection_contract')
FILES = ('CONTRACT.md', 'PREREG.md', 'fiber.py', 'test_fiber.py',
         'run_math.py', 'collection_contract.py')
NAMES = (
    'test_opt_in_and_bad_bounds_fail_closed',
    'test_affine_composition_is_exact',
    'test_add_cancellation_and_shared_consumers',
    'test_concat_preserves_one_joint_amplitude',
    'test_misaligned_frames_and_unjoined_branches_rejected',
    'test_relu_zero_keeps_both_phase_labels',
    'test_relu_graph_is_nonconvex',
    'test_stable_relu_keeps_original_phase_identity',
    'test_source_predicates_and_decoder_preserved',
    'test_exact_fiber_support_recurrence',
    'test_global_control_and_stable_successor',
    'test_old_triangle_feasible_point_is_only_comparison',
)
ANCHORS = {
    HERE.parent / 'd135_causal_fiber_candidate_20261003/DEFINITION.md':
        '3dc8e91ddd5cc317871a49d10d286d33434ce471d9994837dea58ffa36e1d8fc',
    PRIOR / 'preregistered.json': 'be10d0155dbabb1621ac1039bd98dcafef8748d23bdeabfb120fbed044b8be2f',
    PRIOR / 'inventory.json': '49584d326b6f5bbc69a469fa435fdf9cd6867370f50663e0f606ba4e8ed95582',
    PRIOR / 'exit.json': '39d9029d448bffe6b25b34246b5da20756df78a80c6c151993f48a1ed49b0514',
    OLD / 'freeze.json': '01a893685f9994a879315e69aecf59d5e4e75f20c0bb8eb7dd582e1201c6bb01',
    OLD / 'run_math.py': 'cf7f9f1b767468430bcc3dfac861105f88625e74a4d1a7f0e6beb4f4744bc33d',
    OLD / 'collection_contract.py': '843985d7c6fa2c73ec5034b95a4b286466c48811ecfdcdb63147fa5b5da95653',
}
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
INHERITED_MODULE = HERE.parent / 'd112_shared_endpoint_forward_20261002/test_endpoint_forward.py'
EVIDENCE_RELOCATION = dict(module_path=str(INHERITED_MODULE),
    original_run=str(EXP / 'results/d112_shared_endpoint_forward_20261002_v1'),
    relocated_run=str(RUN / 'inherited_d112_controls'))


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked identity: ' + str(path))
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 8 * 1024**2:
        raise ValueError('invalid JSON identity: ' + str(path))
    return json.loads(path.read_text())


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


def dependencies(helper, gpu, identities, inputs, prior):
    """Read and hash the inherited package/source populations; never initialize CUDA."""
    sources_copy, inputs_copy = dict(identities), dict(inputs)
    if (helper.select_sources(sources_copy, inputs_copy) != prior['selected_sources']
            or helper.bind_decoder(sources_copy) != prior['decoder_dependency_files']
            or gpu.gpu_dependencies(helper, sources_copy) != prior['gpu_dependency_files']
            or sources_copy != identities or inputs_copy != inputs):
        raise ValueError('inherited dependency or original-input inventory drift')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs = {}, {}
    old = helper = gpu = production = closure = prior = None
    manifest_digest = None
    result = dict(schema=SCHEMA, mathematical_stage_only=True,
        worker_stage_registered=False, worker_launched=False,
        component_tests_passed=False, mathematical_component_gate_passed=False,
        inventory_validated_before_execution=False,
        source_component_qualified=False, source_census_completed=False,
        source_census_qualified=False, actual_model_binding_qualified=False,
        actual_phase_column_binding_verified=False, native_HZ_admitted=False,
        gpu_computation_completed=False, complete_physical_qualification=False,
        candidate_physical_gate_evaluated=False, production_snapshot_imported=False,
        fixed_component_lp_controls_registered=True, solver_rescue_registered=False,
        new_tests=12, formal_gain=0, new_benchmark_solves=0)
    try:
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        tracemalloc.start()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
            CUDA_LOG_FILE='stderr', CUDA_CACHE_PATH=str(RUN / 'cuda_cache'),
            TORCH_HOME=str(RUN / 'torch_home'), XDG_CACHE_HOME=str(RUN / 'xdg_cache'),
            TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        sys.path.insert(0, str(ROOT))
        frozen = read(FREEZE)
        if (frozen.get('schema') != SCHEMA
                or frozen.get('required_tests') != 3965
                or frozen.get('required_test_files') != 208
                or frozen.get('new_test_names') != list(NAMES)
                or frozen.get('mathematical_stage_only') is not True
                or frozen.get('worker_stage_registered') is not False
                or frozen.get('fixed_component_lp_controls_registered') is not True
                or frozen.get('solver_rescue_registered') is not False
                or type(frozen.get('source_sha256')) is not dict
                or set(frozen['source_sha256']) != {str(HERE / n) for n in FILES}):
            raise ValueError('frozen D136 contract differs')
        identities.update(frozen['source_sha256'])
        identities.update({str(p): d for p, d in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d136_authenticated_d130', identities)
        prior, done, inventory = (read(PRIOR / n) for n in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        if (prior.get('schema') != 'd130_import_isolation_v1'
                or prior.get('required_tests') != 3953 or prior.get('required_test_files') != 207
                or len(prior.get('source_sha256', {})) != 7184
                or len(prior.get('input_sha256', {})) != 14
                or any(done.get(k) is not True for k in ('component_tests_passed',
                    'mathematical_component_gate_passed', 'host_observations_within_caps',
                    'inventory_validated_before_execution'))
                or done.get('tests_exit') != 0 or done.get('tests_count') != 3953
                or done.get('test_files') != 207 or not 0 <= done.get('test_wall_s', 61) <= 60
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or done.get('all_stages_passed') is not False
                or done.get('source_component_qualified') is not False
                or done.get('source_census_qualified') is not False
                or done.get('failure') != dict(type='ValueError', stage='source',
                                               reason='complete source population gate failed')
                or done.get('source_diagnostic_failure') != dict(type='TimeoutError',
                    reason='source internal 235 second stop for terminal evidence')
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 3953 or inventory.get('files') != 207
                or inventory.get('manifest_sha256') != sha(PRIOR / 'preregistered.json')
                or inventory.get('validated_before_execution') is not True):
            raise ValueError('D130 mathematical receipt or preserved source failure differs')
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
        helper = load_checked(old.HELPER, '_d136_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d136_authenticated_d017_dependencies', identities)
        if (type(prior.get('cpu_affinity')) is not list or len(prior['cpu_affinity']) != 1
                or list(os.sched_getaffinity(0)) != prior['cpu_affinity']):
            raise ValueError('inherited CPU affinity differs')
        dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance']:
            raise ValueError('production provenance differs')
        tests = list(prior['tests'])
        expected = list(prior['expected_nodeids'])
        if (len(tests) != 207 or len(set(tests)) != 207
                or len(expected) != 3953 or len(set(expected)) != 3953):
            raise ValueError('inherited test population differs')
        test_path = HERE / 'test_fiber.py'
        functions = [n for n in ast.parse(test_path.read_text()).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith('test_')]
        if ([n.name for n in functions] != list(NAMES)
                or any(not isinstance(n, ast.FunctionDef) or n.decorator_list
                    or n.args.args or n.args.posonlyargs or n.args.kwonlyargs
                    or n.args.vararg or n.args.kwarg for n in functions)):
            raise ValueError('twelve registered plain test functions required')
        tests.append(str(test_path))
        expected.extend(str(test_path.relative_to(ROOT)) + '::' + n for n in NAMES)
        if (len(set(tests)) != 208 or len(set(expected)) != 3965
                or any(p not in identities for p in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        save('preregistered.json', dict(schema=SCHEMA, source_sha256=identities,
            input_sha256=inputs, provenance=production, project_import_closure=closure,
            tests=tests, expected_nodeids=expected, required_tests=3965,
            required_test_files=208, inherited_tests=3953, inherited_test_files=207,
            new_test_files=1, new_test_names=NAMES, inherited_test_population_unchanged=True,
            inherited_D130_receipt=dict(path=str(PRIOR),
                manifest_sha256=sha(PRIOR / 'preregistered.json'),
                exit_sha256=sha(PRIOR / 'exit.json'), inventory_sha256=sha(PRIOR / 'inventory.json'),
                mathematical_component_gate_passed=True, source_component_qualified=False,
                source_failure=done['source_diagnostic_failure'], qualification_transferred=False),
            semantic_definition=dict(path=str(HERE.parent / 'd135_causal_fiber_candidate_20261003/DEFINITION.md'),
                sha256=ANCHORS[HERE.parent / 'd135_causal_fiber_candidate_20261003/DEFINITION.md']),
            historical_contracts=prior['historical_contracts'],
            gpu_dependency_files=prior['gpu_dependency_files'],
            decoder_dependency_files=prior['decoder_dependency_files'],
            selected_sources=prior['selected_sources'],
            inherited_component_evidence_relocation=EVIDENCE_RELOCATION,
            freeze_sha256=sha(FREEZE), pytest_import_mode='importlib',
            pytest_plugin_autoload=False,
            same_process_collection_gate=True, single_pytest_process=True,
            collection_plugin=PLUGIN, cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=AS_CAP, tests_combined_wall_cap_s=60,
            host_memory_cap_bytes=MEMORY_CAP, summary_reserve_bytes=RESERVE,
            mathematical_stage_only=True, worker_stage_registered=False,
            source_component_qualified=False, source_census_qualified=False,
            native_HZ_admitted=False, gpu_computation_completed=False,
            complete_physical_qualification=False, fixed_component_lp_controls_registered=True,
            solver_rescue_registered=False, cuda_visible_devices='', formal_gain=0,
            new_benchmark_solves=0))
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D136_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
            '--tb=short', '-p', 'no:cacheprovider', '-p', PLUGIN,
            '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic() - test_started,
                      tests_count=3965, test_files=208)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 3965
                or checked.get('files') != 208 or checked.get('manifest_sha256') != manifest_digest
                or checked.get('validated_before_execution') is not True
                or checked.get('inherited_component_evidence_relocation') != EVIDENCE_RELOCATION
                or sha(RUN / 'preregistered.json') != manifest_digest):
            raise ValueError('pre-execution inventory contract differs')
        result['inventory_validated_before_execution'] = True
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if (process.returncode != 0 or result['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))):
            raise ValueError('complete mathematical test gate failed')
        result['component_tests_passed'] = True
        result['mathematical_component_gate_passed'] = True
    except BaseException as exc:
        if test_started is not None:
            result['test_wall_s'] = time.monotonic() - test_started
        result['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096], stage='mathematical')
    finally:
        try:
            if closure is not None and old.project_closure(identities, closure) != closure:
                raise ValueError('post-execution project closure differs')
            if helper is not None and gpu is not None and prior is not None:
                dependencies(helper, gpu, identities, inputs, prior)
                if list(os.sched_getaffinity(0)) != prior['cpu_affinity']:
                    raise ValueError('post-execution CPU affinity differs')
            result['source_drift'] = [p for p, d in identities.items() if sha(p) != d]
            result['input_drift'] = [p for p, d in inputs.items() if sha(p) != d]
            result['provenance_drift'] = production is not None and helper.provenance() != production
            if (result['source_drift'] or result['input_drift'] or result['provenance_drift']
                    or (manifest_digest is not None and sha(RUN / 'preregistered.json') != manifest_digest)):
                raise ValueError('post-execution identity drift')
        except BaseException as exc:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        try:
            result['artifacts'] = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*') if p.is_file()}
        except BaseException as exc:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason='artifact sealing incomplete: ' + str(exc)[:4096]))
        try:
            if old is None:
                raise ValueError('authenticated telemetry helper unavailable')
            old.host_observations(result, rss0)
        except BaseException as exc:
            result['host_observations_within_caps'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        if not result['host_observations_within_caps']:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type='MemoryError', reason='supervisor memory gate failed'))
        passed = result['mathematical_component_gate_passed'] and 'failure' not in result
        result.update(all_registered_stages_passed=passed, supervisor_exit=0 if passed else 1,
            wall_s=time.monotonic() - started,
            memory_scope='supervisor observed; pytest AS/CPU/time only; no GPU or full physical qualification')
        save('exit.json', result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
