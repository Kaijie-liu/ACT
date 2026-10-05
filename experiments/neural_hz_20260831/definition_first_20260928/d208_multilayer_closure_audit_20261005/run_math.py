"""Once-only D208 closure counterexample audit; no new domain or capability."""
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
RUN = EXP / 'results/d208_multilayer_closure_audit_20261005_v1'
PRIOR = EXP / 'results/d207_owned_source_phase_20261005_v1'
BASE = HERE.parent / 'd207_owned_source_phase_20261005'
OLD = HERE.parent / 'd130_import_isolation_20261002'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd208_multilayer_closure_audit_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd208_multilayer_closure_audit_20261005.collection_contract')
FILES = ('CONTRACT.md', 'PREREG.md', 'test_closure.py',
         'run_math.py', 'collection_contract.py')
NAMES = (
    'test_two_layer_counterexample_retains_full_interface',
    'test_same_domain_proves_the_lost_preactivation_relation',
    'test_fixed_queries_lose_the_relation_at_next_relu',
    'test_direction_family_and_phase_preservation',
)
ANCHORS = {
    PRIOR / 'preregistered.json': '04eea48c758e5a163fc5ed5889930d459e27ef6507ea6a293acf2c8ed5dce42a',
    PRIOR / 'inventory.json': 'a689dbf08e640a9be2db0b635d0becf8184634fef0711cde878c95df5111a7a9',
    PRIOR / 'exit.json': '0f0b8ba612481bcfaf5a7af6c8ac92a1ba7e202be4f796b51dcc46f9786d724d',
    PRIOR / 'source_phase_positive_control.json': '931ac276a6729e4df849683d2d47d7b84a4c5cb0e1a428f2524f4bb3274303ad',
    PRIOR / 'exact_alias_control.json': 'e915041eb241928859c3ce85349a48986a153d9190cd35fa662188bedadf2182',
    PRIOR / 'complete_200_gate_control.json': '2b43b81b3f981bba0f5b008567cc5ff043cfe5434021df87ca6dae54db04321c',
    BASE / 'freeze.json': '5422d502b74929f9454ef3b88d15c685ed51dd7dcc34da888ab0ca706db15f60',
    BASE / 'run_math.py': 'cd6f302fe79cef1a16f9e5493897a39614d4b3ab7f16d45bcb75c37ea44fb237',
    BASE / 'collection_contract.py': 'f17e3caa7bed0181c8ddbfdc09380b71ed6fa9f9efcf0ef0ab65716094f61130',
    BASE / 'fiber.py': '1e33d626749abecba4236a932151edd3411ccfacc1418049c5bd1fd8c6d5f1ef',
    BASE / 'test_fiber.py': '1c23e38befc50be621e7dacf863f20bddc9ff107b4ef2c3bc3742ee8d6627e02',
    OLD / 'run_math.py': 'cf7f9f1b767468430bcc3dfac861105f88625e74a4d1a7f0e6beb4f4744bc33d',
}
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
INHERITED_MODULE = HERE.parent / 'd112_shared_endpoint_forward_20261002/test_endpoint_forward.py'
EVIDENCE_RELOCATION = dict(module_path=str(INHERITED_MODULE),
    original_run=str(EXP / 'results/d112_shared_endpoint_forward_20261002_v1'),
    relocated_run=str(RUN / 'inherited_d112_controls'))

D207_MODULE = HERE.parent / 'd207_owned_source_phase_20261005/test_fiber.py'
D207_RECORD_FILES = ('source_phase_positive_control.json', 'exact_alias_control.json',
                     'complete_200_gate_control.json')
D207_EVIDENCE_RELOCATION = dict(module_path=str(D207_MODULE),
    source_sha256='1c23e38befc50be621e7dacf863f20bddc9ff107b4ef2c3bc3742ee8d6627e02',
    function_name='_record', function_firstlineno=44,
    original_run=str(EXP / 'results/d207_owned_source_phase_20261005_v1'),
    relocated_run=str(RUN / 'inherited_d207_controls'),
    allowed_filenames=list(D207_RECORD_FILES), mechanism='module_local_record_function_only')



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
    """Hash complete inherited dependency populations; never initialize CUDA."""
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
        new_tests=4, formal_gain=0, new_benchmark_solves=0,
        negative_audit_only=True, domain_definition_changed=False,
        new_domain_qualified=False, new_capability_qualified=False,
        capability_improvement_claimed=False, negative_audit_tests_passed=False)
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
                or frozen.get('required_tests') != 4084
                or frozen.get('required_test_files') != 215
                or frozen.get('new_test_names') != list(NAMES)
                or frozen.get('mathematical_stage_only') is not True
                or frozen.get('worker_stage_registered') is not False
                or frozen.get('fixed_component_lp_controls_registered') is not True
                or frozen.get('solver_rescue_registered') is not False
                or frozen.get('negative_audit_only') is not True
                or frozen.get('domain_definition_changed') is not False
                or type(frozen.get('source_sha256')) is not dict
                or set(frozen['source_sha256']) != {str(HERE / n) for n in FILES}):
            raise ValueError('frozen D208 contract differs')
        identities.update(frozen['source_sha256'])
        identities.update({str(p): d for p, d in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d208_authenticated_d130_helpers', identities)
        prior, done, inventory = (read(PRIOR / n) for n in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        if (prior.get('schema') != 'd207_owned_source_phase_v1'
                or prior.get('required_tests') != 4080 or prior.get('required_test_files') != 214
                or len(prior.get('source_sha256', {})) != 7377
                or len(prior.get('input_sha256', {})) != 14
                or prior.get('mathematical_stage_only') is not True
                or prior.get('worker_stage_registered') is not False
                or any(done.get(k) is not True for k in ('component_tests_passed',
                    'mathematical_component_gate_passed', 'host_observations_within_caps',
                    'inventory_validated_before_execution', 'all_registered_stages_passed'))
                or done.get('tests_exit') != 0 or done.get('supervisor_exit') != 0
                or done.get('tests_count') != 4080 or done.get('test_files') != 214
                or not 0 <= done.get('test_wall_s', 61) <= 60 or 'failure' in done
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or any(done.get(k) is not False for k in ('worker_stage_registered',
                    'worker_launched', 'source_component_qualified', 'source_census_qualified',
                    'actual_model_binding_qualified', 'actual_phase_column_binding_verified',
                    'native_HZ_admitted', 'gpu_computation_completed', 'complete_physical_qualification'))
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 4080 or inventory.get('files') != 214
                or inventory.get('manifest_sha256') != sha(PRIOR / 'preregistered.json')
                or inventory.get('validated_before_execution') is not True):
            raise ValueError('D207 complete mathematical receipt differs')
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
        helper = load_checked(old.HELPER, '_d208_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d208_authenticated_d017_dependencies', identities)
        if (type(prior.get('cpu_affinity')) is not list or len(prior['cpu_affinity']) != 1
                or list(os.sched_getaffinity(0)) != prior['cpu_affinity']):
            raise ValueError('inherited CPU affinity differs')
        dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance']:
            raise ValueError('production provenance differs')
        tests = list(prior['tests'])
        expected = list(prior['expected_nodeids'])
        if (len(tests) != 214 or len(set(tests)) != 214
                or len(expected) != 4080 or len(set(expected)) != 4080):
            raise ValueError('inherited test population differs')
        test_path = HERE / 'test_closure.py'
        functions = [n for n in ast.parse(test_path.read_text()).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith('test_')]
        if ([n.name for n in functions] != list(NAMES)
                or any(not isinstance(n, ast.FunctionDef) or n.decorator_list
                    or n.args.args or n.args.posonlyargs or n.args.kwonlyargs
                    or n.args.vararg or n.args.kwarg for n in functions)):
            raise ValueError('four registered plain test functions required')
        tests.append(str(test_path))
        expected.extend(str(test_path.relative_to(ROOT)) + '::' + n for n in NAMES)
        if (len(set(tests)) != 215 or len(set(expected)) != 4084
                or any(p not in identities for p in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        save('preregistered.json', dict(schema=SCHEMA, source_sha256=identities,
            input_sha256=inputs, provenance=production, project_import_closure=closure,
            tests=tests, expected_nodeids=expected, required_tests=4084,
            required_test_files=215, inherited_tests=4080, inherited_test_files=214,
            new_test_files=1, new_test_names=NAMES, inherited_test_population_unchanged=True,
            inherited_D207_receipt=dict(path=str(PRIOR),
                manifest_sha256=sha(PRIOR / 'preregistered.json'),
                exit_sha256=sha(PRIOR / 'exit.json'), inventory_sha256=sha(PRIOR / 'inventory.json'),
                mathematical_component_gate_passed=True, source_component_qualified=False,
                qualification_transferred=False),
            preserved_D180_receipt=prior['inherited_D180_receipt'],
            preserved_D158_receipt=prior['preserved_D158_receipt'],
            preserved_D157_receipt=prior['preserved_D157_receipt'],
            preserved_D150_receipt=prior['preserved_D150_receipt'],
            preserved_D149_receipt=prior['preserved_D149_receipt'],
            preserved_D136_receipt=prior['preserved_D136_receipt'],
            preserved_D130_receipt=prior['preserved_D130_receipt'],
            inherited_semantic_definition=prior['inherited_semantic_definition'],
            previous_candidate_semantic_definition=prior['candidate_semantic_definition'],
            preserved_D207_operator_definition=prior['operator_definition'],
            preserved_D180_operator_definition=prior['preserved_D180_operator_definition'],
            audited_semantic_definition=prior['candidate_semantic_definition'],
            exact_alias_definition=prior['exact_alias_definition'],
            formula_anchors=prior['formula_anchors'],
            preserved_structural_diagnostic_anchors=prior['preserved_structural_diagnostic_anchors'],
            operator_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                kind='multilayer_closure_negative_audit', domain_definition_changed=False),
            historical_contracts=prior['historical_contracts'],
            gpu_dependency_files=prior['gpu_dependency_files'],
            decoder_dependency_files=prior['decoder_dependency_files'],
            selected_sources=prior['selected_sources'],
            inherited_component_evidence_relocation=EVIDENCE_RELOCATION,
            inherited_D207_evidence_relocation=D207_EVIDENCE_RELOCATION,
            freeze_sha256=sha(FREEZE), pytest_import_mode='importlib',
            pytest_plugin_autoload=False,
            same_process_collection_gate=True, single_pytest_process=True,
            collection_plugin=PLUGIN, cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=AS_CAP, tests_combined_wall_cap_s=60,
            host_memory_cap_bytes=MEMORY_CAP, summary_reserve_bytes=RESERVE,
            mathematical_stage_only=True, worker_stage_registered=False, worker_launched=False,
            source_component_qualified=False, source_census_completed=False, source_census_qualified=False,
            actual_model_binding_qualified=False, actual_phase_column_binding_verified=False,
            native_HZ_admitted=False, gpu_computation_completed=False,
            complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
            production_snapshot_imported=False, fixed_component_lp_controls_registered=True,
            solver_rescue_registered=False, cuda_visible_devices='', formal_gain=0,
            new_benchmark_solves=0, negative_audit_only=True,
            domain_definition_changed=False, new_domain_qualified=False,
            new_capability_qualified=False, capability_improvement_claimed=False))
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D208_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
            '--tb=short', '-p', 'no:cacheprovider', '-p', PLUGIN,
            '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic() - test_started,
                      tests_count=4084, test_files=215)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 4084
                or checked.get('files') != 215 or checked.get('manifest_sha256') != manifest_digest
                or checked.get('validated_before_execution') is not True
                or checked.get('inherited_component_evidence_relocation') != EVIDENCE_RELOCATION
                or checked.get('inherited_D207_evidence_relocation') != D207_EVIDENCE_RELOCATION
                or sha(RUN / 'preregistered.json') != manifest_digest):
            raise ValueError('pre-execution inventory contract differs')
        result['inventory_validated_before_execution'] = True
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if (process.returncode != 0 or result['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))):
            raise ValueError('complete mathematical test gate failed')
        result['negative_audit_tests_passed'] = True
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
