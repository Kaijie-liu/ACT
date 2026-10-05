"""Single-use component qualification, with the exact D088 test population preserved.

This supervisor authenticates the prior complete record and all dependencies;
it never invokes an old main, modifies old module globals, or runs a model.
Passing these tests does not qualify an actual native/model binding.
The separately registered archive probe is never launched by this supervisor.
D090 changes dependency identities, not the D088 kernel, tests, or test paths.
Closure authentication hashes sources only; the unchanged tests keep their
original imports. No production snapshot is loaded by this supervisor.
"""
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
RUN = EXP / 'results/d090_bound_native_discovery_20261001_v1'
PRIOR = EXP / 'results/d088_native_structure_discovery_20261001_v1'
HELPER = HERE.parent / 'd015_batch_binding_20260928_v2/run_v2.py'
GPU = HERE.parent / 'd017_applicability_gpu_20260930/gpu_preflight.py'
FREEZE = HERE / 'freeze.json'
CLOSURE = HERE / 'project_import_closure.json'
ACT_ROOT = ROOT / 'act'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd090_bound_native_discovery_20261001.collection_contract')
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FILES = ('CONTRACT.md', 'PREREG.md', 'project_import_closure.json',
         'archive_probe.py', 'run_archive.py', 'run_math.py',
         'collection_contract.py')
NAMES = ()
ANCHORS = {
    PRIOR / 'preregistered.json': 'f39afe471e40b6cbbedb6f91ff930908e8e7764c85a4adb3e9b324d28d6abd70',
    PRIOR / 'exit.json': 'cf2b889bc85bb62a01db0698b58bdafce98e8a01f7a0c48c40340f7bc956c08c',
    PRIOR / 'inventory.json': 'f2b01a59b55bd531069c3017ea52a540ffb8cb278c84225de23197df0b1d461d',
    HERE.parent / 'd088_native_structure_discovery_20261001/freeze.json':
        '51de8cc7bf00570a04917f92443dea5af58ca632a605270f4337c9c74d25c203',
    PRIOR / 'archive_probe/worker.json':
        'b18d0c833259b5ec9a7a154e493805e3702169e34dd09d587185dbe411586698',
    PRIOR / 'archive_probe_supervisor/supervisor.json':
        '17211a863406983f569ff2c27f6ee752ad616282250861290edbf2be1ef3c495',
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


def project_closure(identities, inherited):
    """Bind the entire fixed project Python tree, without importing any file.

    This is the same source/hash scope as mathematical qualification, not an
    assertion that a production snapshot was imported or model execution ran.
    Membership, byte sizes and hashes are checked again after the test process.
    """
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
            or len(files) != CLOSURE_COUNT):
        raise ValueError('project import closure contract differs')
    if ACT_ROOT.is_symlink() or not ACT_ROOT.is_dir():
        raise ValueError('project source root is missing or linked')
    actual = sorted(str(p) for p in ACT_ROOT.rglob('*.py'))
    if actual != list(files):
        raise ValueError('project source population differs')
    total, added_count, added_bytes = 0, 0, 0
    for path, record in files.items():
        p = Path(path)
        if (not p.is_absolute() or str(p) != path or p.suffix != '.py'
                or not p.is_relative_to(ACT_ROOT) or p.resolve() != p
                or p.is_symlink() or not p.is_file()
                or type(record) is not dict or set(record) != {'sha256', 'bytes'}
                or type(record['bytes']) is not int
                or not 0 <= record['bytes'] <= CLOSURE_BYTES
                or p.stat().st_size != record['bytes']):
            raise ValueError('invalid project source entry: ' + path)
        merge_identity(identities, path, record['sha256'])
        if sha(p) != record['sha256']:
            raise ValueError('project source hash differs: ' + path)
        total += record['bytes']
        if path not in inherited:
            added_count += 1
            added_bytes += record['bytes']
    if total != CLOSURE_BYTES or added_count != 98 or added_bytes != 1954459:
        raise ValueError('project source total or newly bound bytes differ')
    return dict(path=str(CLOSURE), sha256=sha(CLOSURE),
                file_count=CLOSURE_COUNT, total_bytes=CLOSURE_BYTES,
                act_file_count=CLOSURE_COUNT, act_total_bytes=CLOSURE_BYTES,
                added_file_count=added_count, added_bytes=added_bytes,
                outside_act_paths=[], source_files=list(files),
                source_hash_only=True)


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
    helper, production, closure_record, inherited_sources = None, None, None, None
    result = dict(component_tests_passed=False, mathematical_component_gate_passed=False,
                  source_component_qualified=False, worker_launched=False,
                  source_census_qualified=False, actual_model_binding_qualified=False,
                  actual_phase_column_binding_verified=False, native_HZ_admitted=False,
                  gpu_computation_completed=False, complete_physical_qualification=False,
                  candidate_physical_gate_evaluated=False,
                  production_snapshot_imported=False, new_tests=0, formal_gain=0)
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
        if (frozen.get('schema') != 'd090_bound_native_discovery_v1'
                or frozen.get('required_tests') != 3825 or frozen.get('required_test_files') != 183
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
                or done.get('tests_count') != 3825 or done.get('test_files') != 183
                or not 0 <= done.get('test_wall_s', 61) <= 60
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False or 'failure' in done
                or prior.get('required_tests') != 3825 or prior.get('required_test_files') != 183
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 3825 or inventory.get('files') != 183):
            raise ValueError('complete D088 success is not established')
        historical = prior.get('historical_contracts')
        if (type(historical) is not dict or set(historical) != {
                'prior_attempt', 'direct_prior_attempt', 'inherited_mathematical_gate',
                'inherited_native_predicate_gate', 'same_candidate_prior_attempt'}):
            raise ValueError('complete D088 historical contracts are missing')
        for path, digest in prior['source_sha256'].items():
            merge_identity(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = (PRIOR / name).resolve()
            if not path.is_relative_to(PRIOR.resolve()):
                raise ValueError('historical artifact escapes its directory')
            merge_identity(identities, str(path), digest)
        inherited_sources = prior['source_sha256']
        closure_record = project_closure(identities, inherited_sources)
        old_worker = read(PRIOR / 'archive_probe/worker.json')
        old_supervisor = read(PRIOR / 'archive_probe_supervisor/supervisor.json')
        if (old_worker.get('archive_probe_qualified') is not False
                or old_worker.get('archive_loaded') is not False
                or old_worker.get('failure', {}).get('reason') !=
                    'unbound project import: /data1/Kane/FSE/ACT/act/__init__.py'
                or old_supervisor.get('worker_exit') != 1
                or old_supervisor.get('archive_probe_qualified') is not False
                or old_supervisor.get('worker_receipt', {}).get('sha256') !=
                    ANCHORS[PRIOR / 'archive_probe/worker.json']):
            raise ValueError('D088 archive failure identity differs')
        failed_archive = dict(path=str(PRIOR / 'archive_probe'),
            worker_sha256=ANCHORS[PRIOR / 'archive_probe/worker.json'],
            supervisor_sha256=ANCHORS[PRIOR / 'archive_probe_supervisor/supervisor.json'],
            archive_probe_qualified=False, archive_loaded=False,
            failure=old_worker['failure'])
        if len(inputs) != 9 or len(prior['gpu_dependency_files']) != 4417 or len(prior['decoder_dependency_files']) != 1011:
            raise ValueError('dependency or input population differs')
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('source/input drift: ' + path)
        if Path(sys.executable).resolve() != PYTHON.resolve() or sha(Path(sys.executable).resolve()) != identities[str(PYTHON.resolve())]:
            raise ValueError('interpreter differs')
        helper = load(HELPER, 'd090_readonly_source_helpers')
        gpu = load(GPU, 'd090_readonly_dependency_helpers')
        if (helper.select_sources(identities, inputs) != prior['selected_sources']
                or helper.bind_decoder(identities) != prior['decoder_dependency_files']
                or gpu.gpu_dependencies(helper, identities) != prior['gpu_dependency_files']):
            raise ValueError('dependency inventory drift')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        tests = list(prior['tests'])
        expected = list(prior['expected_nodeids'])
        if len(tests) != 183 or len(set(tests)) != 183 or len(expected) != 3825 or len(set(expected)) != 3825:
            raise ValueError('complete inherited population differs')
        if any(path not in identities for path in tests) or helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-execution identity drift')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            required_tests=3825, required_test_files=183, new_test_names=NAMES,
            inherited_tests=3825, inherited_test_files=183,
            inherited_success=dict(path=str(PRIOR), exit_sha256=sha(PRIOR / 'exit.json')),
            inherited_predecessor_success=prior['inherited_success'],
            historical_contracts=historical,
            same_candidate_archive_prior_attempt=failed_archive,
            project_import_closure=closure_record,
            test_population_unchanged=True, candidate_kernel_unchanged=True,
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
            worker_stage_registered=False, single_archive_probe_registered=True,
            archive_worker_wall_cap_s=240, cuda_visible_devices='',
            complete_physical_qualification=False, formal_gain=0))
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D090_MANIFEST_SHA256'] = manifest_digest
        print(json.dumps(dict(event='frozen_before_candidate_import', tests=3825, files=183)), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
                   '-p', PLUGIN, '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic()-test_started,
                      tests_count=3825, test_files=183)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 3825
                or checked.get('files') != 183 or checked.get('manifest_sha256') != manifest_digest
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
            if closure_record is not None and project_closure(identities, inherited_sources) != closure_record:
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
