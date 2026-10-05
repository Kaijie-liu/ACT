"""One frozen D245 mixed-source phase capacity relation; mathematical stage only."""
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
RUN = EXP / 'results/d245_mixed_source_component_20261005_v1'
PRIOR = EXP / 'results/d243_operator_source_relation_20261005_v1'
BASE = HERE.parent / 'd243_operator_source_relation_20261005'
DEPENDENCIES = HERE.parent / 'd214_parametric_relation_consumption_20261005/run_math.py'
THEORY = HERE.parent / 'd244_mixed_source_capacity_20261005'
SOURCE_BASE = HERE.parent / 'd241_residual_source_binding_20261005'
SOURCE_RUN = EXP / 'results/d241_residual_source_binding_20261005_v1'
OLD = HERE.parent / 'd130_import_isolation_20261002'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd245_mixed_source_component_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd245_mixed_source_component_20261005.collection_contract')
FILES = ('CONTRACT.md', 'PREREG.md', 'mixed_relation.py', 'test_mixed_relation.py',
         'run_math.py', 'collection_contract.py')
NAMES = (
    'test_01_default_off_and_frame_identity',
    'test_02_shared_source_recovery_without_direct_parent_edges',
    'test_03_stored_row_physical_projection_certificate',
    'test_04_complete_single_child_prefix_witnesses',
    'test_05_joint_children_over_convexified_parents',
    'test_06_all_qualified_old_tau_certificates',
    'test_07_fixed_integer_extensions_and_decoder',
    'test_08_original_zero_phase_labels',
    'test_09_complete_nonzero_residual_and_derived_tau',
    'test_10_same_range_distinct_source_identity',
    'test_11_asymmetric_parent_normalization',
    'test_12_bn_error_and_shared_consumers',
    'test_13_binary_source_and_retained_predicates',
    'test_14_difference_only_tail_alias',
    'test_15_rank_and_dependent_parent_rejection',
    'test_16_guard_and_invalid_input_rejection',
    'test_17_sticky_shared_budget',
    'test_18_complete_materialized_cost',
    'test_19_original_state_immutability',
    'test_20_summary_and_evidence_boundary',
)
NEW_RECORD_FILES = ('summary.json',)
ANCHORS = {
    PRIOR / 'preregistered.json': '07d3ab4c5792c470935e65aabc080acb75f38c6fdfc23b2623d7ae6a8a55162f',
    PRIOR / 'inventory.json': 'c5f27c281a9d20bd038e7f9d7c48055faab73e63b5054d4403629b66f4a6b2d7',
    PRIOR / 'exit.json': 'ddfab992b32ba48f1eedb09a59e5349b7e0c849ad233909e05e97a7f2e0e134a',
    PRIOR / 'summary.json': '595dec2222a5c75c478f500cc36d4f8249f03d3aea5137bc189ee4562e2383cd',
    BASE / 'freeze.json': '397fb28da73fcc75acb9d4981ce62d5ef774c37e49f2b227f154438f79533f51',
    BASE / 'ARCHIVE.sha256': '377eb818d7dbe5dc61663cca0ef04cc129d1582243964b848df016b641ba507a',
    BASE / 'CONTRACT.md': '832628efcbbcf443f85f6c2a9fea21b3bb7d83b3458b02025ef4268038d2ce19',
    BASE / 'PREREG.md': '134eabd8c3ca2769a49714ac76d2718d6d2b758943c9a458f0906059c12632c0',
    BASE / 'source_relation.py': '8ec6e75d10ddfb47ad16c3968f1335fdcec0a7862d97a47443a61c56ae8e1e04',
    BASE / 'test_source_relation.py': '1954a87da60a9da4601e9bbb292ae9bbeb3dac75976724304fb48d48ccd9a026',
    BASE / 'run_math.py': 'd29050101ea55f9ccd314d2ba3dfad73bc581cf7b0365498b0b86628b9bbf8a0',
    BASE / 'collection_contract.py': 'cbbf505c369de9d8cbd27488b905558a8c5f7b0abe1143253a1e298752e23e7e',
    THEORY / 'ARCHIVE.sha256': 'c67a55e65172b3969c8c85396d76a8ef7e4f12226017437424bad7ec159abda4',
    THEORY / 'THEORY.md': '6ad33a74d3319d6a430d16cdecf5deaa12cf70467f40f590edde21836c95f2f5',
    THEORY / 'STRUCTURAL_AUDIT.md': '65c4f5dba78f714ec477e09753bcbecbc9ec212e5af4c8b55ce599217b0b2e17',
    THEORY / 'RESEARCH_RECORD.md': '7d40fd9383d90b94ca61564b77a18437612d4500c8ac71aa4badf8cb8d392579',
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
        if (frozen.get('schema') != SCHEMA or frozen.get('required_tests') != 4209
                or frozen.get('required_test_files') != 224
                or frozen.get('new_test_names') != list(NAMES)
                or frozen.get('new_evidence_files') != list(NEW_RECORD_FILES)
                or any(frozen.get(key) is not True for key in TRUE_FLAGS)
                or any(frozen.get(key) is not False for key in
                       ('worker_stage_registered', 'solver_rescue_registered',
                        'negative_audit_only', 'new_set_class'))
                or type(frozen.get('source_sha256')) is not dict
                or set(frozen['source_sha256']) != {str(HERE / name) for name in FILES}):
            raise ValueError('frozen D245 contract differs')
        identities.update(frozen['source_sha256'])
        identities.update({str(path): digest for path, digest in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d245_authenticated_d130', identities)
        inherited = load_checked(BASE / 'run_math.py', '_d245_authenticated_d243', identities)
        dependency_helper = load_checked(DEPENDENCIES, '_d245_authenticated_dependencies', identities)
        prior, done, inventory = (read(PRIOR / name) for name in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        prior_freeze = read(BASE / 'freeze.json')
        if (prior.get('schema') != 'd243_operator_source_relation_v1'
                or prior.get('required_tests') != 4189 or prior.get('required_test_files') != 223
                or len(prior.get('source_sha256', {})) != 7709
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
                or done.get('tests_count') != 4189 or done.get('test_files') != 223
                or not 0 <= done.get('test_wall_s', 61) <= 60 or 'failure' in done
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 4189 or inventory.get('files') != 223
                or inventory.get('manifest_sha256') != ANCHORS[PRIOR / 'preregistered.json']
                or inventory.get('validated_before_execution') is not True
                or prior_freeze.get('source_sha256') != {
                    str(BASE / name): ANCHORS[BASE / name] for name in inherited.FILES}):
            raise ValueError('D243 complete mathematical receipt differs')
        for path, digest in prior['source_sha256'].items():
            old.merge_identity(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = PRIOR / name
            if path.resolve() != path or not path.is_relative_to(PRIOR):
                raise ValueError('historical artifact escapes original directory')
            old.merge_identity(identities, str(path), digest)
        source_record = inherited.source_reference(
            old, identities, inputs, read(inherited.PRIOR / 'preregistered.json'))
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('inherited identity drift: ' + path)
        if (Path(sys.executable).resolve() != old.PYTHON.resolve()
                or sha(Path(sys.executable).resolve()) != identities[str(old.PYTHON.resolve())]):
            raise ValueError('interpreter differs')
        closure = old.project_closure(identities, prior.get('project_import_closure'))
        helper = load_checked(old.HELPER, '_d245_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d245_authenticated_d017_dependencies', identities)
        dependency_helper.dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance'] or list(os.sched_getaffinity(0)) != [0]:
            raise ValueError('production provenance or CPU affinity differs')
        tests, expected = list(prior['tests']), list(prior['expected_nodeids'])
        if (len(tests) != 223 or len(set(tests)) != 223
                or len(expected) != 4189 or len(set(expected)) != 4189):
            raise ValueError('inherited ordered population differs')
        test_path = HERE / 'test_mixed_relation.py'
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
        if (len(set(tests)) != 224 or len(set(expected)) != 4209
                or any(path not in identities for path in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        relocations = {}
        for key in ('inherited_component_evidence_relocation', 'inherited_D207_evidence_relocation',
                    'inherited_D208_evidence_relocation', 'inherited_D209_evidence_relocation',
                    'inherited_D214_evidence_relocation', 'inherited_D228_evidence_relocation',
                    'inherited_D229_evidence_relocation', 'inherited_D230_evidence_relocation',
                    'inherited_D231_evidence_relocation', 'inherited_D240_evidence_relocation'):
            relocation = dict(prior[key])
            relocation['relocated_run'] = str(RUN / Path(relocation['relocated_run']).name)
            relocations[key] = relocation
        relocations['inherited_D243_evidence_relocation'] = dict(
            module_path=str(BASE / 'test_source_relation.py'), source_sha256=ANCHORS[BASE / 'test_source_relation.py'],
            function_name='_record_file', function_firstlineno=46,
            original_run=str(PRIOR), relocated_run=str(RUN / 'inherited_d243_controls'),
            allowed_filenames=['summary.json'], mechanism='module_local_record_function_only')
        # Preserve all authenticated historical definitions/receipts, but never
        # inherit a qualification or the previous candidate's execution identity.
        manifest = dict(prior)
        manifest.update(schema=SCHEMA, source_sha256=identities, input_sha256=inputs,
            provenance=production, project_import_closure=closure, tests=tests,
            expected_nodeids=expected, required_tests=4209, required_test_files=224,
            inherited_tests=4189, inherited_test_files=223, new_test_files=1,
            new_test_names=list(NAMES), new_evidence_files=list(NEW_RECORD_FILES),
            inherited_test_population_unchanged=True,
            inherited_D243_receipt=dict(path=str(PRIOR),
                manifest_sha256=ANCHORS[PRIOR / 'preregistered.json'],
                inventory_sha256=ANCHORS[PRIOR / 'inventory.json'],
                exit_sha256=ANCHORS[PRIOR / 'exit.json'],
                mathematical_component_gate_passed=True, qualification_transferred=False),
            preserved_D243_operator_definition=prior['operator_definition'],
            preserved_D243_candidate_semantic_definition=prior['candidate_semantic_definition'],
            last_successful_candidate_semantic_definition=prior['candidate_semantic_definition'],
            candidate_semantic_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                qualification_transferred=False),
            operator_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                kind='same_H_mixed_source_phase_capacity',
                domain_definition_changed=True, new_set_class=False),
            D241_source_reference=source_record,
            D244_theorem_reference={str(path): digest for path, digest in ANCHORS.items()
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
        env['NEURAL_HZ_D245_MANIFEST_SHA256'] = manifest_digest
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
            raise ValueError('complete D245 mathematical evidence missing')
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
