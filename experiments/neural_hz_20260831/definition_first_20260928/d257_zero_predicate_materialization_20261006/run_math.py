"""One frozen D257 zero-predicate materialization; mathematical stage only."""
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
RUN = EXP / 'results/d257_zero_predicate_materialization_20261006_v1'
PRIOR = EXP / 'results/d255_rebase_relation_transport_20261006_v1'
BASE = HERE.parent / 'd255_rebase_relation_transport_20261006'
DEPENDENCIES = HERE.parent / 'd214_parametric_relation_consumption_20261005/run_math.py'
THEORY = HERE.parent / 'd252_quantified_guard_capacity_20261006'
TRANSPORT_THEORY = HERE.parent / 'd256_zero_readout_predicate_transport_20261006'
SOURCE_BASE = HERE.parent / 'd241_residual_source_binding_20261005'
SOURCE_RUN = EXP / 'results/d241_residual_source_binding_20261005_v1'
SOURCE_AUTHORITY = HERE.parent / 'd243_operator_source_relation_20261005/run_math.py'
OLD = HERE.parent / 'd130_import_isolation_20261002'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd257_zero_predicate_materialization_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd257_zero_predicate_materialization_20261006.collection_contract')
FILES = ('CONTRACT.md', 'PREREG.md', 'predicate_transport.py', 'test_transport.py',
         'run_math.py', 'collection_contract.py')
NAMES = (
    'test_01_default_off_and_private_namespace',
    'test_02_reachable_source_and_fresh_widths',
    'test_03_joint_sources_skip_and_bias',
    'test_04_zero_chain_normalization_and_checkpoint',
    'test_05_masked_rows_retain_predicates',
    'test_06_implicit_operator_and_invalid_inputs',
    'test_07_full_materialized_terminal_relation',
    'test_08_integer_extensions_and_decoder',
    'test_09_internal_selective_materialization',
    'test_10_source_drift_and_atomic_rejection',
    'test_11_shared_budget_and_no_production_mutation',
    'test_12_summary_and_qualification_boundary',
)
NEW_RECORD_FILES = ('summary.json',)
ANCHORS = {
    HERE.parent / 'd253_quantified_native_component_20261006/freeze.json': '9ed1df8d814383de358efa05a9eb6bcbc862a4b77e67112ff069cdd71e0317ed',
    HERE.parent / 'd253_quantified_native_component_20261006/ARCHIVE.sha256': '730a4c2be3fa1e5a611d18a4bbaabc50fc8ab9bedc87070463b91423d888a1ca',
    EXP / 'results/d253_quantified_native_component_20261006_v1/exit.json': 'd3e44ae1463bcc921964560e2c1dbe748f81102093e77d3ddd4a2d1ee2cd5285',
    PRIOR / 'preregistered.json': '49a4af4dcea963fa2bf719761a1e97f86db31585ffdbbde9e5ccfa3f5b153714',
    PRIOR / 'inventory.json': '117593a550a3268a103c53b64dadcae8425b46e9ed0dad07cda0b81ab176a08b',
    PRIOR / 'exit.json': 'd1e86c7c166d867b89b60a4c1021253bddfcc6f7853f938599f3b0c58b2dfd27',
    PRIOR / 'summary.json': '4d3408a64297469f53dba851af4526a7f517f856040dbb7d8f97fd26b37bcc68',
    BASE / 'freeze.json': 'd92f7bc30be05bceb1a8efbb061616fc95f7c0938f65f98a5617839def7f5f69',
    BASE / 'ARCHIVE.sha256': '06e1b0a38fb11f0287106e1174d11f7acf1b8f0f96de38eeaf28f6c68c4a8328',
    BASE / 'CONTRACT.md': '15ce3c92e2dcc8c4124153246596a2d26f6d355610a9e678d14d0163cbb623ab',
    BASE / 'PREREG.md': 'e94c06d1b18ba56f77fe73848e802ae5697fab2501a4a849bd116b9984e43d4d',
    BASE / 'native_rebased.py': 'e6f2c3581203d00a8fffa19c4faa2e05d7c64f9b4fa6f3439f7f7500918154bd',
    BASE / 'test_rebased.py': 'c04c3a775aa1cf09d16e3a865a8fb211b8b758a911bf2799df4803ac4a5b4639',
    BASE / 'run_math.py': 'da94752d515fb054b4093ee856449820305718697f98138cfe9028392dd2edde',
    BASE / 'collection_contract.py': 'b0056f1616b0c4b2dcfcf04704628e115da6c696bab06dc0dddc17834b4273bd',
    BASE / 'SOURCE_ENTRY.md': 'a6725cceb5ea591b14f452d50396dbc3961dce2c5e9fb3dabf4414093f9f8fd4',
    BASE / 'RESULTS.md': '6a9b25e71dcd59ee8df225f554f4a717a781079d414f7282c51993933b1841fe',
    BASE / 'RESEARCH_RECORD.md': 'a6f635e6f45d9adc131e044eba175b045f009bbbd2ce8872526ecbb530fe5755',
    THEORY / 'ARCHIVE.sha256': 'f6e5a0c8a3341a342b2c7a898477ccc370be2471953ac976165888d729f8c198',
    THEORY / 'THEORY.md': '58a809e63dd1153bcf3055a18820dc27e25cc9be506e695e8d41c152b907fd1d',
    THEORY / 'CONTROL.md': '1970f7ac468119e541e1b17bbb9934bb84550a92d49e38f4d469fedf05434e1a',
    THEORY / 'CLIP_BOUNDARY.md': 'a158f8c12f9b11aac2a02f309d63120d89f10bed3e9df7e47600c9177a6c8764',
    THEORY / 'RESEARCH_RECORD.md': 'f65c606efce5349ca226a9d64d7498e00c0ed7ffb90c9dc42d024eee25286823',
    TRANSPORT_THEORY / 'THEORY.md': 'd3e6a14ac2b616e78c3f3ca8ef664f08c181b01389aac02f207eec9a7dc36148',
    TRANSPORT_THEORY / 'SOURCE_AUDIT.md': 'ae5219e5c32424d9737cc51769425cef510352f33c1b6591e396beffc2c0c579',
    TRANSPORT_THEORY / 'RESEARCH_RECORD.md': '23f84cf5e54cf2bd7f279614e1a455114e242bef0b5b59b37d51d4f1c74565b6',
    TRANSPORT_THEORY / 'ARCHIVE.sha256': '43ef0b3c06b4b26a72e1d293423e3369cf8eea2445a659bdeac955e60fd53486',
    OLD / 'run_math.py': 'cf7f9f1b767468430bcc3dfac861105f88625e74a4d1a7f0e6beb4f4744bc33d',
    DEPENDENCIES: 'f0f25fe16df7392ed8f407e612af3e51e965177549a832bf4daab01bc4041f42',
    SOURCE_AUTHORITY: 'd29050101ea55f9ccd314d2ba3dfad73bc581cf7b0365498b0b86628b9bbf8a0',
}
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
TRUE_FLAGS = ('mathematical_stage_only', 'fixed_component_lp_controls_registered',
              'new_component_solver_free')
FALSE_FLAGS = ('worker_stage_registered', 'worker_launched', 'source_component_qualified',
    'source_census_completed', 'source_census_qualified', 'actual_model_binding_qualified',
    'actual_phase_column_binding_verified', 'native_HZ_admitted', 'gpu_computation_completed',
    'complete_physical_qualification', 'candidate_physical_gate_evaluated',
    'production_snapshot_imported', 'solver_rescue_registered', 'negative_audit_only',
    'new_set_class', 'new_domain_qualified', 'new_capability_qualified',
    'capability_improvement_claimed', 'domain_definition_changed',
    'native_runtime_installation_qualified', 'online_lifecycle_qualified',
    'native_mathematical_transport_passed', 'quantified_native_transport_passed',
    'rebase_native_transport_passed')


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
    old = inherited = source_authority = dependency_helper = helper = gpu = production = closure = prior = None
    manifest_digest = None
    result = dict(schema=SCHEMA, new_tests=12, formal_gain=0, independent_e0_gain=0, new_benchmark_solves=0,
        component_tests_passed=False, mathematical_component_gate_passed=False,
        zero_predicate_materialization_passed=False,
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
        if (frozen.get('schema') != SCHEMA or frozen.get('required_tests') != 4269
                or frozen.get('required_test_files') != 228
                or frozen.get('new_test_names') != list(NAMES)
                or frozen.get('new_evidence_files') != list(NEW_RECORD_FILES)
                or any(frozen.get(key) is not True for key in TRUE_FLAGS)
                or any(frozen.get(key) is not False for key in
                       ('worker_stage_registered', 'solver_rescue_registered',
                        'negative_audit_only', 'new_set_class', 'domain_definition_changed',
                        'zero_predicate_materialization_passed', 'quantified_native_transport_passed',
                        'native_mathematical_transport_passed',
                        'native_runtime_installation_qualified', 'online_lifecycle_qualified',
                        'rebase_native_transport_passed'))
                or type(frozen.get('source_sha256')) is not dict
                or set(frozen['source_sha256']) != {str(HERE / name) for name in FILES}):
            raise ValueError('frozen D257 contract differs')
        identities.update(frozen['source_sha256'])
        identities.update({str(path): digest for path, digest in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d257_authenticated_d130', identities)
        inherited = load_checked(BASE / 'run_math.py', '_d257_authenticated_d255', identities)
        dependency_helper = load_checked(DEPENDENCIES, '_d257_authenticated_dependencies', identities)
        prior, done, inventory = (read(PRIOR / name) for name in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        prior_freeze = read(BASE / 'freeze.json')
        if (prior.get('schema') != 'd255_rebase_relation_v1'
                or prior.get('required_tests') != 4257 or prior.get('required_test_files') != 227
                or len(prior.get('source_sha256', {})) != 7886
                or len(prior.get('input_sha256', {})) != 14
                or prior.get('cpu_affinity') != [0]
                or any(prior.get(key) is not True for key in inherited.TRUE_FLAGS)
                or any(prior.get(key) is not False for key in inherited.FALSE_FLAGS)
                or any(done.get(key) is not True for key in ('component_tests_passed',
                    'mathematical_component_gate_passed', 'host_observations_within_caps',
                    'inventory_validated_before_execution', 'all_registered_stages_passed',
                    'rebase_native_transport_passed'))
                or any(done.get(key) is not False for key in inherited.FALSE_FLAGS)
                or done.get('negative_audit_only') is not False
                or done.get('formal_gain') != 0 or done.get('new_benchmark_solves') != 0
                or done.get('tests_exit') != 0 or done.get('supervisor_exit') != 0
                or done.get('tests_count') != 4257 or done.get('test_files') != 227
                or not 0 <= done.get('test_wall_s', 61) <= 60 or 'failure' in done
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 4257 or inventory.get('files') != 227
                or inventory.get('manifest_sha256') != ANCHORS[PRIOR / 'preregistered.json']
                or inventory.get('validated_before_execution') is not True
                or prior_freeze.get('source_sha256') != {
                    str(BASE / name): ANCHORS[BASE / name] for name in inherited.FILES}):
            raise ValueError('D255 complete mathematical receipt differs')
        for path, digest in prior['source_sha256'].items():
            old.merge_identity(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = PRIOR / name
            if path.resolve() != path or not path.is_relative_to(PRIOR):
                raise ValueError('historical artifact escapes original directory')
            old.merge_identity(identities, str(path), digest)
        source_authority = load_checked(SOURCE_AUTHORITY, '_d257_authenticated_d243_source_reference', identities)
        source_record = source_authority.source_reference(
            old, identities, inputs, read(source_authority.PRIOR / 'preregistered.json'))
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('inherited identity drift: ' + path)
        if (Path(sys.executable).resolve() != old.PYTHON.resolve()
                or sha(Path(sys.executable).resolve()) != identities[str(old.PYTHON.resolve())]):
            raise ValueError('interpreter differs')
        closure = old.project_closure(identities, prior.get('project_import_closure'))
        helper = load_checked(old.HELPER, '_d257_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d257_authenticated_d017_dependencies', identities)
        dependency_helper.dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance'] or list(os.sched_getaffinity(0)) != [0]:
            raise ValueError('production provenance or CPU affinity differs')
        tests, expected = list(prior['tests']), list(prior['expected_nodeids'])
        if (len(tests) != 227 or len(set(tests)) != 227
                or len(expected) != 4257 or len(set(expected)) != 4257):
            raise ValueError('inherited ordered population differs')
        test_path = HERE / 'test_transport.py'
        functions = [node for node in ast.parse(test_path.read_text()).body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if ([node.name for node in functions] != list(NAMES)
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                    or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                    or node.args.vararg or node.args.kwarg for node in functions)):
            raise ValueError('twelve registered plain test functions required')
        tests.append(str(test_path))
        expected.extend(str(test_path.relative_to(ROOT)) + '::' + name for name in NAMES)
        if (len(set(tests)) != 228 or len(set(expected)) != 4269
                or any(path not in identities for path in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        relocations = {}
        for key in ('inherited_component_evidence_relocation', 'inherited_D207_evidence_relocation',
                    'inherited_D208_evidence_relocation', 'inherited_D209_evidence_relocation',
                    'inherited_D214_evidence_relocation', 'inherited_D228_evidence_relocation',
                    'inherited_D229_evidence_relocation', 'inherited_D230_evidence_relocation',
                    'inherited_D231_evidence_relocation', 'inherited_D240_evidence_relocation',
                    'inherited_D243_evidence_relocation', 'inherited_D245_evidence_relocation',
                    'inherited_D249_evidence_relocation', 'inherited_D254_evidence_relocation'):
            relocation = dict(prior[key])
            relocation['relocated_run'] = str(RUN / Path(relocation['relocated_run']).name)
            relocations[key] = relocation
        relocations['inherited_D255_evidence_relocation'] = dict(
            module_path=str(BASE / 'test_rebased.py'), source_sha256=ANCHORS[BASE / 'test_rebased.py'],
            function_name='_record_file', function_firstlineno=54,
            original_run=str(PRIOR), relocated_run=str(RUN / 'inherited_d255_controls'),
            allowed_filenames=['summary.json'], mechanism='module_local_record_function_only')
        # Preserve all authenticated historical definitions/receipts, but never
        # inherit a qualification or the previous candidate's execution identity.
        manifest = dict(prior)
        manifest.update(schema=SCHEMA, source_sha256=identities, input_sha256=inputs,
            provenance=production, project_import_closure=closure, tests=tests,
            expected_nodeids=expected, required_tests=4269, required_test_files=228,
            inherited_tests=4257, inherited_test_files=227, new_test_files=1,
            new_test_names=list(NAMES), new_evidence_files=list(NEW_RECORD_FILES),
            inherited_test_population_unchanged=True,
            inherited_D255_receipt=dict(path=str(PRIOR),
                manifest_sha256=ANCHORS[PRIOR / 'preregistered.json'],
                inventory_sha256=ANCHORS[PRIOR / 'inventory.json'],
                exit_sha256=ANCHORS[PRIOR / 'exit.json'],
                mathematical_component_gate_passed=True, qualification_transferred=False),
            preserved_D255_operator_definition=prior['operator_definition'],
            preserved_D255_candidate_semantic_definition=prior['candidate_semantic_definition'],
            last_successful_candidate_semantic_definition=prior['candidate_semantic_definition'],
            candidate_semantic_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                qualification_transferred=False),
            operator_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                kind='exact_zero_predicate_materialization',
                domain_definition_changed=False, new_set_class=False),
            D241_source_reference=source_record,
            D255_source_entry_reference=dict(path=str(BASE / 'SOURCE_ENTRY.md'),
                sha256=ANCHORS[BASE / 'SOURCE_ENTRY.md'], qualification_transferred=False),
            D256_transport_theory_reference=dict(path=str(TRANSPORT_THEORY),
                source_sha256={str(path): digest for path, digest in ANCHORS.items()
                               if path.is_relative_to(TRANSPORT_THEORY)},
                qualification_transferred=False),
            D252_theorem_reference={str(path): digest for path, digest in ANCHORS.items()
                                   if path.is_relative_to(THEORY)},
            freeze_sha256=sha(FREEZE), collection_plugin=PLUGIN, pytest_import_mode='importlib',
            pytest_plugin_autoload=False, same_process_collection_gate=True,
            single_pytest_process=True, cpu_affinity=[0], address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, cuda_visible_devices='', formal_gain=0,
            independent_e0_gain=0, new_benchmark_solves=0,
            zero_predicate_materialization_passed=False, **relocations)
        manifest.update({key: True for key in TRUE_FLAGS})
        manifest.update({key: False for key in FALSE_FLAGS})
        save('preregistered.json', manifest)
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D257_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
            '--tb=short', '-p', 'no:cacheprovider', '-p', PLUGIN,
            '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic() - test_started,
                      tests_count=4269, test_files=228)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 4269
                or checked.get('files') != 228 or checked.get('manifest_sha256') != manifest_digest
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
            raise ValueError('complete D257 mathematical evidence missing')
        result['component_tests_passed'] = True
        result['mathematical_component_gate_passed'] = True
        result['zero_predicate_materialization_passed'] = True
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
        result['zero_predicate_materialization_passed'] = passed
        result.update(all_registered_stages_passed=passed, supervisor_exit=0 if passed else 1,
            wall_s=time.monotonic() - started,
            memory_scope='supervisor observed; pytest AS/CPU/time only; no GPU or full physical qualification')
        save('exit.json', result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
