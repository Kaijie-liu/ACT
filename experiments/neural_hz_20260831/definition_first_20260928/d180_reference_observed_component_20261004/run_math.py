"""Once-only D180 mathematical gate; no source worker or benchmark verdicts."""
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
RUN = EXP / 'results/d180_reference_observed_component_20261004_v1'
PRIOR = EXP / 'results/d158_joint_forward_support_20261004_v1'
BASE = HERE.parent / 'd158_joint_forward_support_20261004'
OLD = HERE.parent / 'd130_import_isolation_20261002'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd180_reference_observed_component_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd180_reference_observed_component_20261004.collection_contract')
FILES = ('CONTRACT.md', 'PREREG.md', 'reference_fiber.py', 'test_reference_fiber.py',
         'run_math.py', 'collection_contract.py')
NAMES = (
    'test_default_off_and_rational_inputs',
    'test_hz_embedding_preserves_predicates_and_decoder',
    'test_single_gate_nonconvexity_and_zero_labels',
    'test_complete_carrier_and_mass_deduplication',
    'test_original_identities_survive_extension',
    'test_shared_skip_add_and_concat',
    'test_independent_sibling_readouts_fail_closed',
    'test_actual_extensions_cover_mixed_two_gate_phases',
    'test_reference_chart_rejects_d157_false_packet',
    'test_complementary_reference_chart_is_exact',
    'test_nonreference_native_loss_remains_visible',
    'test_finite_rows_do_not_replace_common_witness',
    'test_all_active_and_inactive_packets_are_exact',
    'test_canonical_readout_retains_source_coefficients',
    'test_shifted_energy_rows_accept_true_extensions',
    'test_range_rows_keep_negative_phase_coefficients',
    'test_flip_energy_is_zero_at_reference_phases',
    'test_two_layer_reference_slice_and_shared_source',
    'test_abstract_parent_extension_is_sound',
    'test_every_inherited_bank_is_checked',
    'test_inherited_joint_positive_control_is_preserved',
    'test_all_fixed_support_certificates_are_sound',
    'test_complete_row_and_logical_cost_accounting',
    'test_unsupported_values_and_resource_limits_fail_closed',
)
THEORY = HERE.parent / 'd178_reference_observed_fiber_20261004'
STRUCTURE = HERE.parent / 'd179_preterminal_domain_20261004'
STRUCTURE_RUN = EXP / 'results/d179_preterminal_domain_20261004_v1'
ANCHORS = {
    PRIOR / 'preregistered.json': '64cd535e4ae10bfaeb8204d21841f991781aa981e2aeddb359a87ca8050103d9',
    PRIOR / 'inventory.json': '7526f53a18410f3108e384c78b5c9226a430262d7bce4cb8cfe24c361e4573bb',
    PRIOR / 'exit.json': '1001c78199c0df081576d2630630ca1d6c36fb2e43515c33d20cd6fdd5e92b23',
    BASE / 'freeze.json': 'b053962da9b59a48d0cdff6587ecc02c5f77300bdf362f315c1cfae63a6525a3',
    BASE / 'run_math.py': '620d7dbb68262e1487401568882b7facbb974ac624b22ba1fbd884a4215fa8b9',
    BASE / 'collection_contract.py': '41ed8f991ccc1ebaa85197e8a126cf598cf2a07de49e4e27c82ddac0fbeb1d38',
    OLD / 'run_math.py': 'cf7f9f1b767468430bcc3dfac861105f88625e74a4d1a7f0e6beb4f4744bc33d',
    THEORY / 'THEORY.md': 'd2737d4b5db496a064bec467691154d2213e23b813a52515d12b9f588e31a78f',
    STRUCTURE / 'DEFINITION_TEST.md': 'c0f59b528647ba74a1c4e080f5cba4bc59edfd6fec0c9c4d3f3d40d22ed2e3b0',
    STRUCTURE / 'RESULTS.md': 'b58bc2536d12687df4e2a94120f0c24253bc134ade9087dd6c5b60b6e4e66461',
    STRUCTURE_RUN / 'model_0.json': '5c1fbd033e91c5fe044a409a782f7f7aa39bc98b47a42719e615b95cc6733da1',
    STRUCTURE_RUN / 'model_1.json': 'f93c16ae170e62d688a76a4f630dfadfb8dcaf75a394b0e59eaf87390405aadb',
    STRUCTURE_RUN / 'model_2.json': 'd6e049295ba60748fee67a9bf9a8a7bc7716d07964ccff413aac2cda3132f4e7',
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
        new_tests=24, formal_gain=0, new_benchmark_solves=0)
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
                or frozen.get('required_tests') != 4056
                or frozen.get('required_test_files') != 213
                or frozen.get('new_test_names') != list(NAMES)
                or frozen.get('mathematical_stage_only') is not True
                or frozen.get('worker_stage_registered') is not False
                or frozen.get('fixed_component_lp_controls_registered') is not True
                or frozen.get('solver_rescue_registered') is not False
                or type(frozen.get('source_sha256')) is not dict
                or set(frozen['source_sha256']) != {str(HERE / n) for n in FILES}):
            raise ValueError('frozen D180 contract differs')
        identities.update(frozen['source_sha256'])
        identities.update({str(p): d for p, d in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d180_authenticated_d130_helpers', identities)
        prior, done, inventory = (read(PRIOR / n) for n in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        if (prior.get('schema') != 'd158_joint_forward_support_v1'
                or prior.get('required_tests') != 4032 or prior.get('required_test_files') != 212
                or len(prior.get('source_sha256', {})) != 7327
                or len(prior.get('input_sha256', {})) != 14
                or prior.get('mathematical_stage_only') is not True
                or prior.get('worker_stage_registered') is not False
                or any(done.get(k) is not True for k in ('component_tests_passed',
                    'mathematical_component_gate_passed', 'host_observations_within_caps',
                    'inventory_validated_before_execution', 'all_registered_stages_passed'))
                or done.get('tests_exit') != 0 or done.get('supervisor_exit') != 0
                or done.get('tests_count') != 4032 or done.get('test_files') != 212
                or not 0 <= done.get('test_wall_s', 61) <= 60 or 'failure' in done
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or any(done.get(k) is not False for k in ('worker_stage_registered',
                    'worker_launched', 'source_component_qualified', 'source_census_qualified',
                    'actual_model_binding_qualified', 'actual_phase_column_binding_verified',
                    'native_HZ_admitted', 'gpu_computation_completed', 'complete_physical_qualification'))
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 4032 or inventory.get('files') != 212
                or inventory.get('manifest_sha256') != sha(PRIOR / 'preregistered.json')
                or inventory.get('validated_before_execution') is not True):
            raise ValueError('D158 complete mathematical receipt differs')
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
        helper = load_checked(old.HELPER, '_d180_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d180_authenticated_d017_dependencies', identities)
        if (type(prior.get('cpu_affinity')) is not list or len(prior['cpu_affinity']) != 1
                or list(os.sched_getaffinity(0)) != prior['cpu_affinity']):
            raise ValueError('inherited CPU affinity differs')
        dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance']:
            raise ValueError('production provenance differs')
        tests = list(prior['tests'])
        expected = list(prior['expected_nodeids'])
        if (len(tests) != 212 or len(set(tests)) != 212
                or len(expected) != 4032 or len(set(expected)) != 4032):
            raise ValueError('inherited test population differs')
        test_path = HERE / 'test_reference_fiber.py'
        functions = [n for n in ast.parse(test_path.read_text()).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith('test_')]
        if ([n.name for n in functions] != list(NAMES)
                or any(not isinstance(n, ast.FunctionDef) or n.decorator_list
                    or n.args.args or n.args.posonlyargs or n.args.kwonlyargs
                    or n.args.vararg or n.args.kwarg for n in functions)):
            raise ValueError('twenty-four registered plain test functions required')
        tests.append(str(test_path))
        expected.extend(str(test_path.relative_to(ROOT)) + '::' + n for n in NAMES)
        if (len(set(tests)) != 213 or len(set(expected)) != 4056
                or any(p not in identities for p in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        save('preregistered.json', dict(schema=SCHEMA, source_sha256=identities,
            input_sha256=inputs, provenance=production, project_import_closure=closure,
            tests=tests, expected_nodeids=expected, required_tests=4056,
            required_test_files=213, inherited_tests=4032, inherited_test_files=212,
            new_test_files=1, new_test_names=NAMES, inherited_test_population_unchanged=True,
            inherited_D158_receipt=dict(path=str(PRIOR),
                manifest_sha256=sha(PRIOR / 'preregistered.json'),
                exit_sha256=sha(PRIOR / 'exit.json'), inventory_sha256=sha(PRIOR / 'inventory.json'),
                mathematical_component_gate_passed=True, source_component_qualified=False,
                qualification_transferred=False),
            preserved_D157_receipt=prior['inherited_D157_receipt'],
            preserved_D150_receipt=prior['preserved_D150_receipt'],
            preserved_D149_receipt=prior['preserved_D149_receipt'],
            preserved_D136_receipt=prior['preserved_D136_receipt'],
            preserved_D130_receipt=prior['preserved_D130_receipt'],
            inherited_semantic_definition=prior['inherited_semantic_definition'],
            previous_candidate_semantic_definition=prior['candidate_semantic_definition'],
            candidate_semantic_definition=dict(path=str(THEORY / 'THEORY.md'),
                sha256=ANCHORS[THEORY / 'THEORY.md'], qualification_transferred=False),
            structural_diagnostic_anchors=dict(
                definition_path=str(STRUCTURE / 'DEFINITION_TEST.md'),
                results_path=str(STRUCTURE / 'RESULTS.md'),
                model_reports=[str(STRUCTURE_RUN / ('model_' + str(i) + '.json')) for i in range(3)],
                mathematical_qualification_transferred=False, model_binding_qualified=False),
            operator_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                kind='reference_observed_consumer_fiber', domain_definition_changed=True),
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
        env['NEURAL_HZ_D180_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
            '--tb=short', '-p', 'no:cacheprovider', '-p', PLUGIN,
            '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic() - test_started,
                      tests_count=4056, test_files=213)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 4056
                or checked.get('files') != 213 or checked.get('manifest_sha256') != manifest_digest
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
