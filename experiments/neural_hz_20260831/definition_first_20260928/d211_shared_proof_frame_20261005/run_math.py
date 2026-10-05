"""Once-only D211 mathematical gate; no native, GPU, or formal qualification."""
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
RUN = EXP / 'results/d211_shared_proof_frame_20261005_v1'
PRIOR = EXP / 'results/d209_owned_slack_closure_20261005_v1'
BASE = HERE.parent / 'd209_owned_slack_closure_20261005'
REFERENCE = HERE.parent / 'd210_full_prefix_admission_20261005'
REFERENCE_RUN = EXP / 'results/d210_full_prefix_admission_20261005_v1'
OLD = HERE.parent / 'd130_import_isolation_20261002'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd211_shared_proof_frame_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd211_shared_proof_frame_20261005.collection_contract')
FILES = ('CONTRACT.md', 'PREREG.md', 'fiber.py', 'test_fiber.py',
         'run_math.py', 'collection_contract.py')
NAMES = (
    'test_default_off_and_hz_embedding',
    'test_all_four_bounds_match_d209',
    'test_positive_cap_pair_equivalence',
    'test_negative_cap_closure_equivalence',
    'test_shared_owner_preserves_original_identity',
    'test_foreign_and_sibling_proofs_rejected',
    'test_modified_and_missing_proofs_rejected',
    'test_ancestor_proof_keeps_birth_bounds',
    'test_exact_alias_equivalence_and_signed_eq',
    'test_sparse_affine_matches_dense',
    'test_sparse_affine_rejects_invalid_rows',
    'test_full_interfaces_bits_and_decoder',
    'test_shared_frame_cost_strictly_decreases',
    'test_wide_source_complete_two_hundred_gates',
    'test_cumulative_budgets_fail_closed',
    'test_stable_zero_and_unshared_boundaries',
)
ANCHORS = {
    PRIOR / 'preregistered.json': '06bf3b6dfce47c0d83571f459e3269acc064001e1dfa2b252979e2185ac846b6',
    PRIOR / 'inventory.json': 'd610e43a02e43d7b61562e31a6d7fed9a8e9e5bcc933d84866311f8c023a1d93',
    PRIOR / 'exit.json': 'd2220587a746d95f3e1db62a7d215aafb9384c24a78f83fe83787da24f706f2e',
    PRIOR / 'closure_repair.json': '414c390b36e45f5f2f45d7c7213cbb25c5364655ef45db8f90e06b072a2c8759',
    PRIOR / 'positive_slack_control.json': 'bc1586e1e07b6f980f83db8137acbe62364b296ee89104437bafadd8e5717edb',
    PRIOR / 'complete_200_gate_control.json': '9f4c637b792845f8c2e5c8b8651404938804b4492f624c81f9faf9fc10bd2de0',
    BASE / 'freeze.json': '50d375250735c53029c59ec8258b54ab0af617a13791b85e26d293fff4c36129',
    BASE / 'run_math.py': 'd9f30c4866fe6bc9c0530329537f73034741b501728e81e2e43a678b3b4ce041',
    BASE / 'collection_contract.py': '2d95ac6fb3890c74a59426024cc46079c8ce767e60eb6c8e23420f59e71c0034',
    BASE / 'test_fiber.py': 'f150be4bb9e9fdcf4fe4117038ceacebf9410d39085ab86022dacaca0f72af77',
    BASE / 'CONTRACT.md': '44f2c3b8df2f56c454780b798ad4f59c70a61b3fc22885689a2d110180d3b51b',
    BASE / 'fiber.py': 'ef8bafaa09512d36c268a72caf8725af5325c73edcc9b721b291ad6210d04430',
    REFERENCE / 'THEORY.md': '2020691220a2fa0350cfe1efa5a14d817d9e62e87a2ac4b6edc0f74ff96124c6',
    REFERENCE / 'RESULTS.md': '0954761405c4fb1dcaa127340012895fab364d53b25005f2b28a2dba9dafece9',
    REFERENCE_RUN / 'admission.json': '36be38c278b1327b1ec9d8b0797d4c7e78f59012b97261b1d7c809ebf39a06cb',
    REFERENCE_RUN / 'exit.json': 'd0612270e90b2d41bf4a598ce2e19aa2fc89f312ebd9f79ce4b678efb0e6afbd',
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


D208_MODULE = HERE.parent / 'd208_multilayer_closure_audit_20261005/test_closure.py'
D208_RECORD_FILES = ('closure_counterexample.json', 'direction_family.json')
D208_EVIDENCE_RELOCATION = dict(module_path=str(D208_MODULE),
    source_sha256='4825d15388db02c02ab4918a915baa0501c41849e784b5556a10a949043007e8',
    function_name='_record', function_firstlineno=39,
    original_run=str(EXP / 'results/d208_multilayer_closure_audit_20261005_v1'),
    relocated_run=str(RUN / 'inherited_d208_controls'),
    allowed_filenames=list(D208_RECORD_FILES), mechanism='module_local_record_function_only')


D209_MODULE = BASE / 'test_fiber.py'
D209_RECORD_FILES = ('closure_repair.json', 'positive_slack_control.json',
                     'complete_200_gate_control.json')
D209_EVIDENCE_RELOCATION = dict(module_path=str(D209_MODULE),
    source_sha256='f150be4bb9e9fdcf4fe4117038ceacebf9410d39085ab86022dacaca0f72af77',
    function_name='_record', function_firstlineno=35,
    original_run=str(PRIOR), relocated_run=str(RUN / 'inherited_d209_controls'),
    allowed_filenames=list(D209_RECORD_FILES), mechanism='module_local_record_function_only')


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
        new_tests=16, formal_gain=0, new_benchmark_solves=0,
        negative_audit_only=False, domain_definition_changed=False,
        new_domain_qualified=False, new_capability_qualified=False,
        capability_improvement_claimed=False)
    try:
        limits()
        if 0 not in os.sched_getaffinity(0):
            raise ValueError('required CPU 0 is unavailable')
        os.sched_setaffinity(0, {0})
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
                or frozen.get('required_tests') != 4116
                or frozen.get('required_test_files') != 217
                or frozen.get('new_test_names') != list(NAMES)
                or frozen.get('mathematical_stage_only') is not True
                or frozen.get('worker_stage_registered') is not False
                or frozen.get('fixed_component_lp_controls_registered') is not True
                or frozen.get('solver_rescue_registered') is not False
                or frozen.get('negative_audit_only') is not False
                or frozen.get('domain_definition_changed') is not False
                or type(frozen.get('source_sha256')) is not dict
                or set(frozen['source_sha256']) != {str(HERE / n) for n in FILES}):
            raise ValueError('frozen D211 contract differs')
        identities.update(frozen['source_sha256'])
        identities.update({str(p): d for p, d in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d211_authenticated_d130_helpers', identities)
        prior, done, inventory = (read(PRIOR / n) for n in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        if (prior.get('schema') != 'd209_owned_slack_closure_v1'
                or prior.get('required_tests') != 4100 or prior.get('required_test_files') != 216
                or len(prior.get('source_sha256', {})) != 7425
                or len(prior.get('input_sha256', {})) != 14
                or prior.get('mathematical_stage_only') is not True
                or prior.get('worker_stage_registered') is not False
                or prior.get('negative_audit_only') is not False
                or prior.get('domain_definition_changed') is not True
                or done.get('negative_audit_only') is not False
                or done.get('domain_definition_changed') is not True
                or done.get('new_domain_qualified') is not False
                or done.get('new_capability_qualified') is not False
                or done.get('capability_improvement_claimed') is not False
                or done.get('formal_gain') != 0 or done.get('new_benchmark_solves') != 0
                or any(done.get(k) is not True for k in ('component_tests_passed',
                    'mathematical_component_gate_passed', 'host_observations_within_caps',
                    'inventory_validated_before_execution', 'all_registered_stages_passed'))
                or done.get('tests_exit') != 0 or done.get('supervisor_exit') != 0
                or done.get('tests_count') != 4100 or done.get('test_files') != 216
                or not 0 <= done.get('test_wall_s', 61) <= 60 or 'failure' in done
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or any(done.get(k) is not False for k in ('worker_stage_registered',
                    'worker_launched', 'source_component_qualified', 'source_census_qualified',
                    'actual_model_binding_qualified', 'actual_phase_column_binding_verified',
                    'native_HZ_admitted', 'gpu_computation_completed', 'complete_physical_qualification'))
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 4100 or inventory.get('files') != 216
                or inventory.get('manifest_sha256') != sha(PRIOR / 'preregistered.json')
                or inventory.get('validated_before_execution') is not True):
            raise ValueError('D209 complete mathematical receipt differs')
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
        helper = load_checked(old.HELPER, '_d211_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d211_authenticated_d017_dependencies', identities)
        if (type(prior.get('cpu_affinity')) is not list or len(prior['cpu_affinity']) != 1
                or list(os.sched_getaffinity(0)) != prior['cpu_affinity']):
            raise ValueError('inherited CPU affinity differs')
        dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance']:
            raise ValueError('production provenance differs')
        tests = list(prior['tests'])
        expected = list(prior['expected_nodeids'])
        if (len(tests) != 216 or len(set(tests)) != 216
                or len(expected) != 4100 or len(set(expected)) != 4100):
            raise ValueError('inherited test population differs')
        test_path = HERE / 'test_fiber.py'
        functions = [n for n in ast.parse(test_path.read_text()).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith('test_')]
        if ([n.name for n in functions] != list(NAMES)
                or any(not isinstance(n, ast.FunctionDef) or n.decorator_list
                    or n.args.args or n.args.posonlyargs or n.args.kwonlyargs
                    or n.args.vararg or n.args.kwarg for n in functions)):
            raise ValueError('sixteen registered plain test functions required')
        tests.append(str(test_path))
        expected.extend(str(test_path.relative_to(ROOT)) + '::' + n for n in NAMES)
        if (len(set(tests)) != 217 or len(set(expected)) != 4116
                or any(p not in identities for p in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        save('preregistered.json', dict(schema=SCHEMA, source_sha256=identities,
            input_sha256=inputs, provenance=production, project_import_closure=closure,
            tests=tests, expected_nodeids=expected, required_tests=4116,
            required_test_files=217, inherited_tests=4100, inherited_test_files=216,
            new_test_files=1, new_test_names=NAMES, inherited_test_population_unchanged=True,
            inherited_D209_receipt=dict(path=str(PRIOR),
                manifest_sha256=sha(PRIOR / 'preregistered.json'),
                exit_sha256=sha(PRIOR / 'exit.json'), inventory_sha256=sha(PRIOR / 'inventory.json'),
                mathematical_component_gate_passed=True, source_component_qualified=False,
                negative_audit_only=False, domain_definition_changed=True,
                new_domain_qualified=False, qualification_transferred=False),
            preserved_D208_receipt=prior['inherited_D208_receipt'],
            preserved_D207_receipt=prior['preserved_D207_receipt'],
            preserved_D180_receipt=prior['preserved_D180_receipt'],
            preserved_D158_receipt=prior['preserved_D158_receipt'],
            preserved_D157_receipt=prior['preserved_D157_receipt'],
            preserved_D150_receipt=prior['preserved_D150_receipt'],
            preserved_D149_receipt=prior['preserved_D149_receipt'],
            preserved_D136_receipt=prior['preserved_D136_receipt'],
            preserved_D130_receipt=prior['preserved_D130_receipt'],
            inherited_semantic_definition=prior['inherited_semantic_definition'],
            previous_candidate_semantic_definition=prior['candidate_semantic_definition'],
            candidate_semantic_definition=prior['candidate_semantic_definition'],
            implementation_contract=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                qualification_transferred=False),
            preserved_D209_operator_definition=prior['operator_definition'],
            preserved_D208_operator_definition=prior['preserved_D208_operator_definition'],
            preserved_D207_operator_definition=prior['preserved_D207_operator_definition'],
            preserved_D180_operator_definition=prior['preserved_D180_operator_definition'],
            audited_semantic_definition=prior['audited_semantic_definition'],
            exact_alias_definition=prior['exact_alias_definition'],
            formula_anchors=prior['formula_anchors'],
            preserved_structural_diagnostic_anchors=prior['preserved_structural_diagnostic_anchors'],
            operator_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                kind='exact_shared_proof_frame', domain_definition_changed=False),
            preserved_D210_admission_reference=dict(
                theory=dict(path=str(REFERENCE / 'THEORY.md'), sha256=ANCHORS[REFERENCE / 'THEORY.md']),
                results=dict(path=str(REFERENCE / 'RESULTS.md'), sha256=ANCHORS[REFERENCE / 'RESULTS.md']),
                admission=dict(path=str(REFERENCE_RUN / 'admission.json'),
                    sha256=ANCHORS[REFERENCE_RUN / 'admission.json']),
                exit=dict(path=str(REFERENCE_RUN / 'exit.json'),
                    sha256=ANCHORS[REFERENCE_RUN / 'exit.json']),
                metadata_only=True, mathematical_component_gate_passed=False,
                qualification_transferred=False),
            historical_contracts=prior['historical_contracts'],
            gpu_dependency_files=prior['gpu_dependency_files'],
            decoder_dependency_files=prior['decoder_dependency_files'],
            selected_sources=prior['selected_sources'],
            inherited_component_evidence_relocation=EVIDENCE_RELOCATION,
            inherited_D207_evidence_relocation=D207_EVIDENCE_RELOCATION,
            inherited_D208_evidence_relocation=D208_EVIDENCE_RELOCATION,
            inherited_D209_evidence_relocation=D209_EVIDENCE_RELOCATION,
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
            new_benchmark_solves=0, negative_audit_only=False,
            domain_definition_changed=False, new_domain_qualified=False,
            new_capability_qualified=False, capability_improvement_claimed=False))
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D211_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
            '--tb=short', '-p', 'no:cacheprovider', '-p', PLUGIN,
            '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic() - test_started,
                      tests_count=4116, test_files=217)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 4116
                or checked.get('files') != 217 or checked.get('manifest_sha256') != manifest_digest
                or checked.get('validated_before_execution') is not True
                or checked.get('inherited_component_evidence_relocation') != EVIDENCE_RELOCATION
                or checked.get('inherited_D207_evidence_relocation') != D207_EVIDENCE_RELOCATION
                or checked.get('inherited_D208_evidence_relocation') != D208_EVIDENCE_RELOCATION
                or checked.get('inherited_D209_evidence_relocation') != D209_EVIDENCE_RELOCATION
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

