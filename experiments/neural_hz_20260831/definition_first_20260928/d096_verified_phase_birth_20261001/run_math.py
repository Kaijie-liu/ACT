"""Single-use verified phase-birth math qualification; all D093 tests preserved.

No old main or old module globals are reused. This supervisor adds exactly
four tests, authenticates the inherited complete project-source closure, and
never launches an archive worker, model, solver, or GPU computation.
D095 is an unexecuted source reference, not a successful predecessor.
A mathematical pass does not establish actual model or physical qualification.
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
RUN = EXP / 'results/d096_verified_phase_birth_20261001_v1'
PRIOR = EXP / 'results/d093_sparse_native_relay_20261001_v1'
ARCHIVE_PRIOR = EXP / 'results/d090_bound_native_discovery_20261001_v1'
HELPER = HERE.parent / 'd015_batch_binding_20260928_v2/run_v2.py'
GPU = HERE.parent / 'd017_applicability_gpu_20260930/gpu_preflight.py'
FREEZE = HERE / 'freeze.json'
CLOSURE = HERE.parent / 'd090_bound_native_discovery_20261001/project_import_closure.json'
ACT_ROOT = ROOT / 'act'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd096_verified_phase_birth_20261001.collection_contract')
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FILES = ('CONTRACT.md', 'PREREG.md', 'phase_birth.py', 'test_phase_birth.py',
         'run_math.py', 'collection_contract.py')
NAMES = ('test_exact_birth_and_zero_phase_labels',
         'test_reused_carriers_and_transported_gate_certificates',
         'test_birth_bank_population_and_literal_equations',
         'test_birth_binding_and_resource_fail_closed')
ANCHORS = {
    PRIOR / 'preregistered.json': '409fa3776ab626079f17b991965f0af88f037b87f17f178e4db6ad35239e5bdd',
    PRIOR / 'exit.json': '21a46dadfe8ed9cbf4900dec7e8b26e5546a19e3598b5439294eab805070c7b6',
    PRIOR / 'inventory.json': '3fdae651f4c4b82cd32030b03cfe5ac6ad54027ef367024f757afa86c439a47d',
    HERE.parent / 'd093_sparse_native_relay_20261001/freeze.json':
        '0fdc6b1415eeb8a5cb58b4a86368c3fb743eab301e1ae919f7dc98814d79e540',
    HERE.parent / 'd093_sparse_native_relay_20261001/run_math.py':
        '3e6c22ac2b85d285eb383df54fbd9379ecae4bd8e44fe4b367142bce635215d4',
    HERE.parent / 'd093_sparse_native_relay_20261001/collection_contract.py':
        '2a7b95081a614d49ffcc1977d5d44eb8ecfaa7d3d4b9c7bd105f3a96bbad2d1a',
    HERE.parent / 'd090_bound_native_discovery_20261001/freeze.json':
        'fb70bc4f78061b44958141c993abe8b2e8829f2d3e638edeca04960cf4d4a8fd',
    ARCHIVE_PRIOR / 'archive_probe/worker.json':
        'c6c03a2f1f800f7df8173f6b669d0efd8d74ad9078042df8cda4474cc9d16c4f',
    ARCHIVE_PRIOR / 'archive_probe_supervisor/supervisor.json':
        '75c4bc03d6490e6d667a5318d531bffc5d624f0709a9ec62ba66ba08aeb493e8',
    HERE.parent / 'd094_compositional_gate_provenance_20261001/THEORY.md':
        'baec345ec9f883ac4a9d0b006fa83d275c476847b7237db42c9039ed7dfb327b',
    HERE.parent / 'd095_phase_birth_20261001/phase_birth.py':
        '8b854a71e8464b805d9aece5ce7bd617bfba61a0a0b762942bbc053175a4a630',
    HERE.parent / 'd095_phase_birth_20261001/DRAFT_REVIEW.md':
        'a2f2bd707bdc7bd31aa2e5848a73cd412570cff9a90918daadd418789daec81e',
    HERE.parent / 'd095_phase_birth_20261001/DRAFT_SHA256SUMS':
        '0a9d6ca80b351d25cd37f34a2491b2630aed0594bfd1ac9555cbfb75990a288b',
}
AS_CAP, MEMORY_CAP, RESERVE = 16*1024**3, 1024**3, 65536
CLOSURE_COUNT, CLOSURE_BYTES = 116, 2782307


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked identity: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path, cap=8*1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('missing, linked or oversized JSON')
    return json.loads(path.read_text())


def merge_identity(identities, path, digest):
    if (type(path) is not str or type(digest) is not str or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('invalid source identity')
    if path in identities and identities[path] != digest:
        raise ValueError('conflicting source identity: ' + path)
    identities[path] = digest


def project_closure(identities, record):
    """Recheck the already inherited D090 closure; admit no additional source.

    Added-file counts in record describe the historical D090 expansion only.
    This version neither recomputes that expansion nor treats it as a new one.
    Authentication reads and hashes files, never imports a project snapshot.
    """
    if (type(record) is not dict or record.get('path') != str(CLOSURE)
            or record.get('sha256') != identities.get(str(CLOSURE))
            or record.get('file_count') != CLOSURE_COUNT
            or record.get('total_bytes') != CLOSURE_BYTES
            or record.get('act_file_count') != CLOSURE_COUNT
            or record.get('act_total_bytes') != CLOSURE_BYTES
            or record.get('outside_act_paths') != []
            or record.get('source_hash_only') is not True
            or sha(CLOSURE) != record['sha256']):
        raise ValueError('inherited closure identity differs')
    closure = read(CLOSURE, RESERVE)
    required = {'schema', 'project_root', 'scope', 'files', 'file_count',
                'total_bytes', 'act_file_count', 'act_total_bytes',
                'outside_act_paths'}
    files = closure.get('files')
    if (set(closure) != required
            or closure.get('schema') != 'd090_project_import_closure_v1'
            or closure.get('project_root') != str(ROOT)
            or closure.get('scope') !=
                'all_existing_act_python_sources_plus_static_external_project_dependencies'
            or type(files) is not dict or list(files) != sorted(files)
            or closure.get('outside_act_paths') != []
            or any(type(closure.get(k)) is not int for k in
                   ('file_count', 'total_bytes', 'act_file_count', 'act_total_bytes'))
            or closure['file_count'] != CLOSURE_COUNT
            or closure['act_file_count'] != CLOSURE_COUNT
            or closure['total_bytes'] != CLOSURE_BYTES
            or closure['act_total_bytes'] != CLOSURE_BYTES
            or len(files) != CLOSURE_COUNT
            or record.get('source_files') != list(files)):
        raise ValueError('inherited project source population differs')
    if ACT_ROOT.is_symlink() or not ACT_ROOT.is_dir():
        raise ValueError('project source root is missing or linked')
    if sorted(str(p) for p in ACT_ROOT.rglob('*.py')) != list(files):
        raise ValueError('current project source population differs')
    total = 0
    for path, item in files.items():
        p = Path(path)
        if (not p.is_absolute() or str(p) != path or p.suffix != '.py'
                or not p.is_relative_to(ACT_ROOT) or p.resolve() != p
                or p.is_symlink() or not p.is_file()
                or type(item) is not dict or set(item) != {'sha256', 'bytes'}
                or type(item['bytes']) is not int
                or not 0 <= item['bytes'] <= CLOSURE_BYTES
                or p.stat().st_size != item['bytes']
                or identities.get(path) != item['sha256']
                or sha(p) != item['sha256']):
            raise ValueError('inherited project source drift: ' + path)
        total += item['bytes']
    if total != CLOSURE_BYTES:
        raise ValueError('project source total bytes differ')
    return record


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


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs = {}, {}
    helper, production, closure_record = None, None, None
    result = dict(component_tests_passed=False, mathematical_component_gate_passed=False,
                  source_component_qualified=False, worker_launched=False,
                  source_census_qualified=False, actual_model_binding_qualified=False,
                  actual_phase_column_binding_verified=False, native_HZ_admitted=False,
                  gpu_computation_completed=False, complete_physical_qualification=False,
                  candidate_physical_gate_evaluated=False,
                  production_snapshot_imported=False, new_tests=4, formal_gain=0)
    try:
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
        sys.path.insert(0, str(ROOT))
        frozen = read(FREEZE, RESERVE)
        sources = frozen.get('source_sha256')
        if (frozen.get('schema') != 'd096_verified_phase_birth_v1'
                or frozen.get('required_tests') != 3837 or frozen.get('required_test_files') != 186
                or frozen.get('new_test_names') != list(NAMES)
                or type(sources) is not dict or set(sources) != {str(HERE / n) for n in FILES}):
            raise ValueError('frozen source and test contract differs')
        for path, digest in sources.items():
            merge_identity(identities, path, digest)
        for path, digest in ANCHORS.items():
            merge_identity(identities, str(path), digest)
        merge_identity(identities, str(FREEZE), sha(FREEZE))
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or historical anchor differs: ' + path)
        prior, done, inventory = (read(PRIOR / n) for n in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        if (any(done.get(k) is not True for k in ('component_tests_passed',
                'mathematical_component_gate_passed',
                'all_stages_passed', 'host_observations_within_caps',
                'inventory_validated_before_execution'))
                or done.get('supervisor_exit') != 0 or done.get('tests_exit') != 0
                or done.get('tests_count') != 3833 or done.get('test_files') != 185
                or not 0 <= done.get('test_wall_s', 61) <= 60
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False or 'failure' in done
                or prior.get('required_tests') != 3833 or prior.get('required_test_files') != 185
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 3833 or inventory.get('files') != 185):
            raise ValueError('complete D093 success is not established')
        historical = prior.get('historical_contracts')
        if (type(historical) is not dict or set(historical) != {
                'prior_attempt', 'direct_prior_attempt', 'inherited_mathematical_gate',
                'inherited_native_predicate_gate', 'same_candidate_prior_attempt'}):
            raise ValueError('complete D093 historical contracts are missing')
        for path, digest in prior['source_sha256'].items():
            merge_identity(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = (PRIOR / name).resolve()
            if not path.is_relative_to(PRIOR.resolve()):
                raise ValueError('historical artifact escapes its directory')
            merge_identity(identities, str(path), digest)
        closure_record = project_closure(identities, prior.get('project_import_closure'))
        failed_archive = prior.get('same_candidate_archive_prior_attempt')
        if (type(failed_archive) is not dict
                or failed_archive.get('archive_probe_qualified') is not False
                or failed_archive.get('archive_loaded') is not False
                or failed_archive.get('path') != str(EXP /
                    'results/d088_native_structure_discovery_20261001_v1/archive_probe')):
            raise ValueError('inherited D088 archive failure record differs')
        old_worker = read(ARCHIVE_PRIOR / 'archive_probe/worker.json')
        old_supervisor = read(ARCHIVE_PRIOR / 'archive_probe_supervisor/supervisor.json')
        if (old_worker.get('archive_probe_qualified') is not False
                or old_worker.get('archive_loaded') is not True
                or old_worker.get('archive_authentication_completed') is not True
                or old_worker.get('archive_census_completed') is not False
                or old_worker.get('memory_gate_passed') is not False
                or old_worker.get('failure', {}).get('type') != 'MemoryError'
                or old_supervisor.get('worker_exit') != 1
                or old_supervisor.get('archive_probe_qualified') is not False
                or old_supervisor.get('worker_receipt', {}).get('sha256') !=
                    ANCHORS[ARCHIVE_PRIOR / 'archive_probe/worker.json']):
            raise ValueError('D090 archive failure identity differs')
        direct_archive = dict(path=str(ARCHIVE_PRIOR / 'archive_probe'),
            worker_sha256=ANCHORS[ARCHIVE_PRIOR / 'archive_probe/worker.json'],
            supervisor_sha256=ANCHORS[ARCHIVE_PRIOR / 'archive_probe_supervisor/supervisor.json'],
            archive_probe_qualified=False, archive_loaded=True,
            archive_census_completed=False, memory_gate_passed=False,
            failure=old_worker['failure'])
        if prior.get('direct_archive_prior_attempt') != direct_archive:
            raise ValueError('D093 inherited direct D090 failure record differs')
        # Preserve the complete prior value, not a replacement success claim.
        direct_archive = prior['direct_archive_prior_attempt']
        if len(inputs) != 9 or len(prior['gpu_dependency_files']) != 4417 or len(prior['decoder_dependency_files']) != 1011:
            raise ValueError('dependency or input population differs')
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('source/input drift: ' + path)
        if Path(sys.executable).resolve() != PYTHON.resolve() or sha(Path(sys.executable).resolve()) != identities[str(PYTHON.resolve())]:
            raise ValueError('interpreter differs')
        helper = load(HELPER, 'd096_readonly_source_helpers')
        gpu = load(GPU, 'd096_readonly_dependency_helpers')
        if (helper.select_sources(identities, inputs) != prior['selected_sources']
                or helper.bind_decoder(identities) != prior['decoder_dependency_files']
                or gpu.gpu_dependencies(helper, identities) != prior['gpu_dependency_files']):
            raise ValueError('dependency inventory drift')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        test_path = HERE / 'test_phase_birth.py'
        tests = [*prior['tests'], str(test_path)]
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if ([node.name for node in functions] != list(NAMES)
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)):
            raise ValueError('four plain new tests required')
        relative = str(test_path.relative_to(ROOT))
        expected = [*prior['expected_nodeids'], *(relative + '::' + name for name in NAMES)]
        if (len(tests) != 186 or len(set(tests)) != 186
                or len(expected) != 3837 or len(set(expected)) != 3837
                or tests[:185] != prior['tests']
                or expected[:3833] != prior['expected_nodeids']):
            raise ValueError('complete inherited population differs')
        if any(path not in identities for path in tests) or helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-execution identity drift')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            required_tests=3837, required_test_files=186, new_test_names=NAMES,
            inherited_tests=3833, inherited_test_files=185,
            inherited_success=dict(path=str(PRIOR), exit_sha256=sha(PRIOR / 'exit.json')),
            inherited_predecessor_success=prior['inherited_success'],
            inherited_predecessor_chain=prior.get('inherited_predecessor_success'),
            historical_contracts=historical,
            same_candidate_archive_prior_attempt=failed_archive,
            direct_archive_prior_attempt=direct_archive,
            project_import_closure=closure_record,
            inherited_test_population_unchanged=True, new_test_files=1,
            selected_sources=prior['selected_sources'],
            gpu_dependency_files=prior['gpu_dependency_files'],
            decoder_dependency_files=prior['decoder_dependency_files'],
            freeze_sha256=sha(FREEZE), same_process_collection_gate=True,
            single_pytest_process=True, collection_plugin=PLUGIN,
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, whole_work_cap=256_000_000,
            branch_work_cap=200_000_000, evidence_prepaid_work=40_000_000,
            retained_entry_cap=64_000_000, rational_bit_cap=512, worker_wall_cap_s=240,
            worker_stage_registered=False, single_archive_probe_registered=False,
            archive_worker_wall_cap_s=240, cuda_visible_devices='',
            complete_physical_qualification=False, formal_gain=0))
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D096_MANIFEST_SHA256'] = manifest_digest
        print(json.dumps(dict(event='frozen_before_candidate_import', tests=3837, files=186)), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
                   '-p', PLUGIN, '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic()-test_started,
                      tests_count=3837, test_files=186)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 3837
                or checked.get('files') != 186 or checked.get('manifest_sha256') != manifest_digest
                or checked.get('validated_before_execution') is not True
                or sha(RUN / 'preregistered.json') != manifest_digest):
            raise ValueError('pre-execution inventory contract differs')
        result['inventory_validated_before_execution'] = True
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/')+'.py::'+c.get('name', '') for c in cases]
        if (process.returncode != 0 or result['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))):
            raise ValueError('complete component test gate failed')
        result['component_tests_passed'] = True
    except BaseException as exc:
        if test_started is not None:
            result['test_wall_s'] = time.monotonic()-test_started
        result['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        try:
            if closure_record is not None and project_closure(identities, closure_record) != closure_record:
                raise ValueError('post-execution project closure differs')
            result['source_drift'] = [p for p, d in identities.items() if sha(p) != d]
            result['input_drift'] = [p for p, d in inputs.items() if sha(p) != d]
            result['provenance_drift'] = production is not None and helper.provenance() != production
            if result['source_drift'] or result['input_drift'] or result['provenance_drift']:
                raise ValueError('post-execution identity drift')
        except BaseException as exc:
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        try:
            result['artifacts'] = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*') if p.is_file()}
        except BaseException as exc:
            result['artifact_sealing_failure'] = str(exc)[:4096]
            result.setdefault('failure', dict(type=type(exc).__name__, reason='artifact sealing incomplete'))
        try:
            if not tracemalloc.is_tracing():
                raise ValueError('missing host telemetry')
            _, peak = tracemalloc.get_traced_memory()
            metadata = tracemalloc.get_tracemalloc_memory()
            growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
            result.update(traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
                rss_highwater_growth_bytes=growth, summary_reserve_bytes=RESERVE,
                host_observations_within_caps=growth+RESERVE <= MEMORY_CAP and peak+metadata+RESERVE <= MEMORY_CAP)
        except BaseException as exc:
            result['host_observations_within_caps'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        if not result['host_observations_within_caps']:
            result.setdefault('failure', dict(type='MemoryError', reason='supervisor memory gate failed'))
        passed = result['component_tests_passed'] and 'failure' not in result
        result.update(mathematical_component_gate_passed=passed,
                      all_stages_passed=passed, supervisor_exit=0 if passed else 1,
                      wall_s=time.monotonic()-started,
                      memory_scope='supervisor only; pytest AS/CPU/time, no full physical qualification')
        save('exit.json', result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
