"""Single-use component qualification, with all D086 tests preserved.

This supervisor authenticates the prior complete record and all dependencies;
it never invokes an old main, modifies old module globals, or runs a model.
Passing these tests does not qualify an actual native/model binding.
The separately registered archive probe is never launched by this supervisor.
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
RUN = EXP / 'results/d087_native_structure_discovery_20261001_v1'
PRIOR = EXP / 'results/d086_native_shared_transfer_20261001_v1'
HELPER = HERE.parent / 'd015_batch_binding_20260928_v2/run_v2.py'
GPU = HERE.parent / 'd017_applicability_gpu_20260930/gpu_preflight.py'
FREEZE = HERE / 'freeze.json'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd087_native_structure_discovery_20261001.collection_contract')
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FILES = ('CONTRACT.md', 'PREREG.md', 'native_discovery.py',
         'test_native_discovery.py', 'archive_probe.py', 'run_math.py', 'collection_contract.py')
NAMES = ('test_guard_catalog_and_uniform_groups',
         'test_full_bank_transfer_and_projection',
         'test_missing_ambiguous_and_compact_scope',
         'test_default_off_and_budget_rejection')
ANCHORS = {
    PRIOR / 'preregistered.json': '7310ba6455016267f4c8ccbc17f0e7bfa6572a9939f5a941d4ec0006ad2ac9f5',
    PRIOR / 'exit.json': '22974fc6669b1db95ae986181c3ac29fecfc4e0ab3b4e773f8064c9405814e6f',
    PRIOR / 'inventory.json': 'd6a756528b4cd32aafab92fb3ddb56996f63dea28fdfd7b3f68f102dc91a8a97',
    HERE.parent / 'd086_native_shared_transfer_20261001/freeze.json':
        '8e16ba6422e40ed2b1242037fff605567bfea81e67c4f3827543c4751e0797ae',
}
AS_CAP, MEMORY_CAP, RESERVE = 16*1024**3, 1024**3, 65536


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
    helper, production = None, None
    result = dict(component_tests_passed=False, mathematical_component_gate_passed=False,
                  source_component_qualified=False, worker_launched=False,
                  source_census_qualified=False, actual_model_binding_qualified=False,
                  actual_phase_column_binding_verified=False, native_HZ_admitted=False,
                  gpu_computation_completed=False, complete_physical_qualification=False,
                  candidate_physical_gate_evaluated=False, formal_gain=0)
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
        if (frozen.get('schema') != 'd087_native_structure_discovery_v1'
                or frozen.get('required_tests') != 3825 or frozen.get('required_test_files') != 183
                or frozen.get('new_test_names') != list(NAMES)
                or type(sources) is not dict or set(sources) != {str(HERE / n) for n in FILES}):
            raise ValueError('frozen source and test contract differs')
        identities.update(sources)
        identities.update({str(p): d for p, d in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
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
                or done.get('tests_count') != 3821 or done.get('test_files') != 182
                or not 0 <= done.get('test_wall_s', 61) <= 60
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False or 'failure' in done
                or prior.get('required_tests') != 3821 or prior.get('required_test_files') != 182
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 3821 or inventory.get('files') != 182):
            raise ValueError('complete D086 success is not established')
        historical = prior.get('historical_contracts')
        if (type(historical) is not dict or set(historical) != {
                'prior_attempt', 'direct_prior_attempt', 'inherited_mathematical_gate',
                'inherited_native_predicate_gate', 'same_candidate_prior_attempt'}):
            raise ValueError('complete D086 historical contracts are missing')
        for path, digest in prior['source_sha256'].items():
            if path in identities and identities[path] != digest:
                raise ValueError('conflicting inherited source identity')
            identities[path] = digest
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = (PRIOR / name).resolve()
            if not path.is_relative_to(PRIOR.resolve()):
                raise ValueError('historical artifact escapes its directory')
            identities[str(path)] = digest
        if len(inputs) != 9 or len(prior['gpu_dependency_files']) != 4417 or len(prior['decoder_dependency_files']) != 1011:
            raise ValueError('dependency or input population differs')
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('source/input drift: ' + path)
        if Path(sys.executable).resolve() != PYTHON.resolve() or sha(Path(sys.executable).resolve()) != identities[str(PYTHON.resolve())]:
            raise ValueError('interpreter differs')
        helper = load(HELPER, 'd087_readonly_source_helpers')
        gpu = load(GPU, 'd087_readonly_dependency_helpers')
        if (helper.select_sources(identities, inputs) != prior['selected_sources']
                or helper.bind_decoder(identities) != prior['decoder_dependency_files']
                or gpu.gpu_dependencies(helper, identities) != prior['gpu_dependency_files']):
            raise ValueError('dependency inventory drift')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        tests = [*prior['tests'], str(HERE / 'test_native_discovery.py')]
        tree = ast.parse((HERE / 'test_native_discovery.py').read_text())
        functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith('test_')]
        if ([n.name for n in functions] != list(NAMES)
                or any(not isinstance(n, ast.FunctionDef) or n.decorator_list
                    or n.args.args or n.args.posonlyargs or n.args.kwonlyargs
                    or n.args.vararg or n.args.kwarg for n in functions)):
            raise ValueError('four plain new tests required')
        relative = str((HERE / 'test_native_discovery.py').relative_to(ROOT))
        expected = [*prior['expected_nodeids'], *(relative + '::' + name for name in NAMES)]
        if len(tests) != 183 or len(set(tests)) != 183 or len(expected) != 3825 or len(set(expected)) != 3825:
            raise ValueError('complete inherited population differs')
        if any(path not in identities for path in tests) or helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-execution identity drift')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            required_tests=3825, required_test_files=183, new_test_names=NAMES,
            inherited_tests=3821, inherited_test_files=182,
            inherited_success=dict(path=str(PRIOR), exit_sha256=sha(PRIOR / 'exit.json')),
            inherited_predecessor_success=prior['inherited_success'],
            historical_contracts=historical,
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
        env['NEURAL_HZ_D087_MANIFEST_SHA256'] = manifest_digest
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
