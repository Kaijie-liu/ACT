"""One frozen D243 common-source operator relation; mathematical stage only."""
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
RUN = EXP / 'results/d243_operator_source_relation_20261005_v1'
PRIOR = EXP / 'results/d240_phase_supported_component_20261005_v1'
BASE = HERE.parent / 'd240_phase_supported_component_20261005'
DEPENDENCIES = HERE.parent / 'd214_parametric_relation_consumption_20261005/run_math.py'
THEORY = HERE.parent / 'd242_common_source_operator_20261005'
SOURCE_BASE = HERE.parent / 'd241_residual_source_binding_20261005'
SOURCE_RUN = EXP / 'results/d241_residual_source_binding_20261005_v1'
OLD = HERE.parent / 'd130_import_isolation_20261002'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd243_operator_source_relation_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd243_operator_source_relation_20261005.collection_contract')
FILES = ('CONTRACT.md', 'PREREG.md', 'source_relation.py', 'test_source_relation.py',
         'run_math.py', 'collection_contract.py')
NAMES = (
    'test_01_default_off_and_frame_identity',
    'test_02_append_seal_and_lineage',
    'test_03_affine_add_and_full_input_decoder',
    'test_04_conv_complete_population_and_padding',
    'test_05_two_layer_padding_boundary',
    'test_06_raw_bn_enclosure_and_shared_error',
    'test_07_same_h_symmetric_certificate',
    'test_08_same_h_asymmetric_certificate',
    'test_09_strong_reference_witnesses',
    'test_10_complete_residual_and_derived_tau',
    'test_11_same_range_distinct_source_identity',
    'test_12_whole_h_integer_extensions',
    'test_13_original_zero_phase_labels',
    'test_14_binary_source_and_retained_predicates',
    'test_15_difference_only_alias',
    'test_16_unsupported_contract_rejected',
    'test_17_exact_arithmetic_and_mutation_rejection',
    'test_18_sticky_shared_budget',
    'test_19_complete_rows_and_cost',
    'test_20_immutable_state_and_summary',
)
NEW_RECORD_FILES = ('summary.json',)
ANCHORS = {
    PRIOR / 'preregistered.json': '7a7c7663bfd2496a840bac459ea136acb76a49393c8211978dec1669a49106ff',
    PRIOR / 'inventory.json': 'f499a8db427c58f945cedf2e508aec204072f2c8406088a1d4efb79c129bdc68',
    PRIOR / 'exit.json': 'e0418f0b687ef18d1fb2d579011cc1517691e2393d766e97e45398ed9649f664',
    PRIOR / 'summary.json': 'b7312330eb45265e4a7288e9c32233f8c334e0421a39c212d055cc0d18388cba',
    BASE / 'freeze.json': '332eead93b3e37aa07c65984059134512700a7db8b12a343955ef54fa8f718ec',
    BASE / 'ARCHIVE.sha256': '81fb9219fc5025fa00d1d28539712fc2317fd496e08da2909cc43268a3115f61',
    BASE / 'CONTRACT.md': 'ba45c0c655a5edc2adc8acdf957b6f03d525142908cf36a365833e87a92938f4',
    BASE / 'PREREG.md': '73379bf638cf0e9141f19abd9e7d694d8c2bf1efe4d80de57a3debdca5e5f73a',
    BASE / 'factor.py': '80505c78877b744a94c284955f6d2d3bc6ce57a2266f3357d9a952e7bbef77df',
    BASE / 'test_factor.py': '0c22812c9123708b291d32b71e3dc761bb75e0aaa46fcaff8df2eeb1d8306829',
    BASE / 'run_math.py': '5bc9ba8ab78a94fc2582711f68b7794c560e10245833b47f3881c4eb9c98e29b',
    BASE / 'collection_contract.py': '50023abe60a48841886bcbeaa1f76c8bfa3df99ca7856edd9551e9e16e699adf',
    THEORY / 'ARCHIVE.sha256': '2a00214a4ac7862fa2125bb711878051c1ef8c1ffcfa9af7e8e12b87f177b679',
    THEORY / 'THEORY.md': '3c0b39083da2648bc34fb088bb20a4e800577f8928c1420830f560d62e3c439d',
    THEORY / 'SOURCE_COSTS.md': '4ad298b2960a59f173e6b323e298b39025e736905a993decfab16c52fb15c2b3',
    THEORY / 'RESEARCH_RECORD.md': 'd72d640ae72f07b90d75223345cff5f02f27e79cfb527ceb9f6917b5ad4a47b9',
    SOURCE_BASE / 'ARCHIVE.sha256': '3caa96f73117a8d78a494b1aec82b755ca4281b845dc335dcb78171474bad86d',
    SOURCE_BASE / 'freeze.json': '5409ca5968df7f6f1c79120f4714a225efb79e0fe87d381972c1736d08274515',
    SOURCE_BASE / 'THEORY.md': 'b512a445ea0ef69e8158d6832f341ffe1aefe58baa5d06d610141fff07373b28',
    SOURCE_BASE / 'PREREG.md': '14d65823e2062f3e3b450964bc3bb1a4614748d4424e42a571315d574b71ac5f',
    SOURCE_RUN / 'preregistered.json': '7600699bc665b0103088e1a9e7ba46d36c53add6cf4588fc12c7858084e2ad29',
    SOURCE_RUN / 'report.json': '638a520ec58d64fb1d28e0725df458c7880ffca037ed86c266c88c63cf204b02',
    SOURCE_RUN / 'exit.json': '9cb62fb6f392a993fa61c24acffa448101ce5ce5434fb5f82864f9c4014ad1bd',
    SOURCE_RUN / 'source_0.json': '9494cb57122809d61f62a7c75f7ee4533c8b41e9683cda86a668321fbef20dc5',
    SOURCE_RUN / 'source_1.json': '5374d47e99db2a34e2099603b469dcb50d828050e536900e1d832267bf8222e0',
    SOURCE_RUN / 'source_2.json': 'ccac731af636a70c1b6210fb8ea6bdd6108fd72ab0949547263fee1bd48b7bed',
    OLD / 'run_math.py': 'cf7f9f1b767468430bcc3dfac861105f88625e74a4d1a7f0e6beb4f4744bc33d',
    DEPENDENCIES: 'f0f25fe16df7392ed8f407e612af3e51e965177549a832bf4daab01bc4041f42',
}
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
TRUE_FLAGS = ('mathematical_stage_only', 'fixed_component_lp_controls_registered',
              'domain_definition_changed', 'new_component_solver_free')
FALSE_FLAGS = ('worker_stage_registered', 'worker_launched', 'source_component_qualified',
    'source_census_completed', 'source_census_qualified', 'actual_model_binding_qualified',
    'actual_phase_column_binding_verified', 'native_HZ_admitted', 'gpu_computation_completed',
    'complete_physical_qualification', 'candidate_physical_gate_evaluated',
    'production_snapshot_imported', 'solver_rescue_registered', 'negative_audit_only',
    'new_set_class', 'new_domain_qualified', 'new_capability_qualified',
    'capability_improvement_claimed')


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked identity: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


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


def source_reference(old, identities, inputs, prior):
    """Authenticate frozen D241 data only; never import or rerun its worker."""
    registration, report, done = (read(SOURCE_RUN / name) for name in
                                 ('preregistered.json', 'report.json', 'exit.json'))
    if (registration.get('schema') != 'd241_residual_source_binding_v1'
            or registration.get('input_sha256') != inputs
            or registration.get('selected_sources') != prior['selected_sources']
            or registration.get('inherited_ordered_nodeids') != prior['expected_nodeids']
            or registration.get('inherited_test_paths') != prior['tests']
            or len(registration.get('source_sha256', {})) != 7691
            or any(registration['source_sha256'].get(path) != digest
                   for path, digest in prior['source_sha256'].items())
            or done.get('status') != 0 or done.get('diagnostic_complete') is not True
            or report.get('diagnostic_complete') is not True or report.get('exit_status') != 0
            or 'failure' in done or 'failure' in report
            or any(report.get(key) is not True for key in ('source_precheck_complete',
                'input_precheck_complete', 'source_postcheck_complete', 'input_postcheck_complete',
                'prior_math_receipt_validated', 'host_observations_within_caps'))
            or report.get('provenance_before') != report.get('provenance_after')
            or registration.get('provenance') != report.get('provenance_before')
            or {key: registration['provenance'].get(key) for key in prior['provenance']} != prior['provenance']
            or report.get('historical_math_tests') != 4169
            or report.get('historical_math_files') != 222
            or report.get('historical_math_population_reduced') is not False
            or registration.get('freeze_sha256') != ANCHORS[SOURCE_BASE / 'freeze.json']
            or done.get('artifact_sha256') != {
                name: ANCHORS[SOURCE_RUN / name] for name in
                ('preregistered.json', 'report.json', 'source_0.json', 'source_1.json', 'source_2.json')}
            or any(done.get(key) is not False for key in ('candidate_executed',
                'model_forward_executed', 'solver_executed', 'mathematical_tests_executed',
                'mathematical_component_gate_passed', 'actual_model_binding_qualified',
                'actual_phase_column_binding_verified', 'native_HZ_admitted',
                'complete_physical_qualification', 'source_component_qualified',
                'gpu_computation_completed', 'new_domain_qualified', 'new_capability_qualified'))
            or any(done.get(key) != 0 for key in ('formal_gain',
                'independent_e0_gain', 'new_benchmark_solves'))
            or len(report.get('models', ())) != 3
            or not all(item.get('complete') is True for item in report['models'])
            or [item.get('source') for item in report['models']] != prior['selected_sources']):
        raise ValueError('D241 source-only receipt differs; no qualification transfers')
    for path, digest in registration['source_sha256'].items():
        old.merge_identity(identities, path, digest)
    return dict(path=str(SOURCE_RUN), manifest_sha256=ANCHORS[SOURCE_RUN / 'preregistered.json'],
        report_sha256=ANCHORS[SOURCE_RUN / 'report.json'], exit_sha256=ANCHORS[SOURCE_RUN / 'exit.json'],
        source_files={str(SOURCE_RUN / ('source_' + str(index) + '.json')):
            ANCHORS[SOURCE_RUN / ('source_' + str(index) + '.json')] for index in range(3)},
        diagnostic_complete=True, audit_reexecuted=False, qualification_transferred=False,
        actual_H_binding_transferred=False, scope='frozen raw source provenance only')


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs = {}, {}
    old = inherited = dependency_helper = helper = gpu = production = closure = prior = None
    manifest_digest = None
    result = dict(schema=SCHEMA, new_tests=20, formal_gain=0, independent_e0_gain=0, new_benchmark_solves=0,
        component_tests_passed=False, mathematical_component_gate_passed=False,
        inventory_validated_before_execution=False)
    result.update({key: True for key in TRUE_FLAGS})
    result.update({key: False for key in FALSE_FLAGS})
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
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='', CUDA_LOG_FILE='stderr', CUDA_CACHE_PATH=str(RUN / 'cuda_cache'),
            TORCH_HOME=str(RUN / 'torch_home'), XDG_CACHE_HOME=str(RUN / 'xdg_cache'),
            TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        sys.path.insert(0, str(ROOT))
        frozen = read(FREEZE)
        if (frozen.get('schema') != SCHEMA or frozen.get('required_tests') != 4189
                or frozen.get('required_test_files') != 223
                or frozen.get('new_test_names') != list(NAMES)
                or frozen.get('new_evidence_files') != list(NEW_RECORD_FILES)
                or any(frozen.get(key) is not True for key in TRUE_FLAGS)
                or any(frozen.get(key) is not False for key in
                       ('worker_stage_registered', 'solver_rescue_registered',
                        'negative_audit_only', 'new_set_class'))
                or type(frozen.get('source_sha256')) is not dict
                or set(frozen['source_sha256']) != {str(HERE / name) for name in FILES}):
            raise ValueError('frozen D243 contract differs')
        identities.update(frozen['source_sha256'])
        identities.update({str(path): digest for path, digest in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d243_authenticated_d130', identities)
        inherited = load_checked(BASE / 'run_math.py', '_d243_authenticated_d240', identities)
        dependency_helper = load_checked(DEPENDENCIES, '_d243_authenticated_dependencies', identities)
        prior, done, inventory = (read(PRIOR / name) for name in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        prior_freeze = read(BASE / 'freeze.json')
        if (prior.get('schema') != 'd240_phase_supported_component_v1'
                or prior.get('required_tests') != 4169 or prior.get('required_test_files') != 222
                or len(prior.get('source_sha256', {})) != 7655
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
                or done.get('tests_count') != 4169 or done.get('test_files') != 222
                or not 0 <= done.get('test_wall_s', 61) <= 60 or 'failure' in done
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 4169 or inventory.get('files') != 222
                or inventory.get('manifest_sha256') != ANCHORS[PRIOR / 'preregistered.json']
                or inventory.get('validated_before_execution') is not True
                or prior_freeze.get('source_sha256') != {
                    str(BASE / name): ANCHORS[BASE / name] for name in inherited.FILES}):
            raise ValueError('D240 complete mathematical receipt differs')
        for path, digest in prior['source_sha256'].items():
            old.merge_identity(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = PRIOR / name
            if path.resolve() != path or not path.is_relative_to(PRIOR):
                raise ValueError('historical artifact escapes original directory')
            old.merge_identity(identities, str(path), digest)
        source_record = source_reference(old, identities, inputs, prior)
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('inherited identity drift: ' + path)
        if (Path(sys.executable).resolve() != old.PYTHON.resolve()
                or sha(Path(sys.executable).resolve()) != identities[str(old.PYTHON.resolve())]):
            raise ValueError('interpreter differs')
        closure = old.project_closure(identities, prior.get('project_import_closure'))
        helper = load_checked(old.HELPER, '_d243_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d243_authenticated_d017_dependencies', identities)
        dependency_helper.dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance'] or list(os.sched_getaffinity(0)) != [0]:
            raise ValueError('production provenance or CPU affinity differs')
        tests, expected = list(prior['tests']), list(prior['expected_nodeids'])
        if (len(tests) != 222 or len(set(tests)) != 222
                or len(expected) != 4169 or len(set(expected)) != 4169):
            raise ValueError('inherited ordered population differs')
        test_path = HERE / 'test_source_relation.py'
        functions = [node for node in ast.parse(test_path.read_text()).body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if ([node.name for node in functions] != list(NAMES)
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                    or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                    or node.args.vararg or node.args.kwarg for node in functions)):
            raise ValueError('twenty registered plain test functions required')
        tests.append(str(test_path))
        expected.extend(str(test_path.relative_to(ROOT)) + '::' + name for name in NAMES)
        if (len(set(tests)) != 223 or len(set(expected)) != 4189
                or any(path not in identities for path in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        relocations = {}
        for key in ('inherited_component_evidence_relocation', 'inherited_D207_evidence_relocation',
                    'inherited_D208_evidence_relocation', 'inherited_D209_evidence_relocation',
                    'inherited_D214_evidence_relocation', 'inherited_D228_evidence_relocation',
                    'inherited_D229_evidence_relocation', 'inherited_D230_evidence_relocation',
                    'inherited_D231_evidence_relocation'):
            relocation = dict(prior[key])
            relocation['relocated_run'] = str(RUN / Path(relocation['relocated_run']).name)
            relocations[key] = relocation
        relocations['inherited_D240_evidence_relocation'] = dict(
            module_path=str(BASE / 'test_factor.py'), source_sha256=ANCHORS[BASE / 'test_factor.py'],
            function_name='_record_file', function_firstlineno=44,
            original_run=str(PRIOR), relocated_run=str(RUN / 'inherited_d240_controls'),
            allowed_filenames=['summary.json'], mechanism='module_local_record_function_only')
        # Preserve all authenticated historical definitions/receipts, but never
        # inherit a qualification or the previous candidate's execution identity.
        manifest = dict(prior)
        manifest.update(schema=SCHEMA, source_sha256=identities, input_sha256=inputs,
            provenance=production, project_import_closure=closure, tests=tests,
            expected_nodeids=expected, required_tests=4189, required_test_files=223,
            inherited_tests=4169, inherited_test_files=222, new_test_files=1,
            new_test_names=list(NAMES), new_evidence_files=list(NEW_RECORD_FILES),
            inherited_test_population_unchanged=True,
            inherited_D240_receipt=dict(path=str(PRIOR),
                manifest_sha256=ANCHORS[PRIOR / 'preregistered.json'],
                inventory_sha256=ANCHORS[PRIOR / 'inventory.json'],
                exit_sha256=ANCHORS[PRIOR / 'exit.json'],
                mathematical_component_gate_passed=True, qualification_transferred=False),
            preserved_D240_operator_definition=prior['operator_definition'],
            preserved_D240_candidate_semantic_definition=prior['candidate_semantic_definition'],
            last_successful_candidate_semantic_definition=prior['candidate_semantic_definition'],
            candidate_semantic_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                qualification_transferred=False),
            operator_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                kind='common_source_operator_phase_supported_relation',
                domain_definition_changed=True, new_set_class=False),
            D241_source_reference=source_record,
            D242_theorem_reference={str(path): digest for path, digest in ANCHORS.items()
                                   if path.is_relative_to(THEORY)},
            freeze_sha256=sha(FREEZE), collection_plugin=PLUGIN, pytest_import_mode='importlib',
            pytest_plugin_autoload=False, same_process_collection_gate=True,
            single_pytest_process=True, cpu_affinity=[0], address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, cuda_visible_devices='', formal_gain=0,
            independent_e0_gain=0, new_benchmark_solves=0, **relocations)
        manifest.update({key: True for key in TRUE_FLAGS})
        manifest.update({key: False for key in FALSE_FLAGS})
        save('preregistered.json', manifest)
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D243_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
            '--tb=short', '-p', 'no:cacheprovider', '-p', PLUGIN,
            '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic() - test_started,
                      tests_count=4189, test_files=223)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 4189
                or checked.get('files') != 223 or checked.get('manifest_sha256') != manifest_digest
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
            raise ValueError('complete D243 mathematical evidence missing')
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
