"""Single-use, default-off rational reference test gate; no GPU/worker stage."""
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
D020 = HERE.parent / 'd020_gpu_init_trace_20260930'
PRIOR = EXP / 'results/d020_gpu_init_trace_20260930_v1'
RUN = EXP / 'results/d024_local_phase_capacity_20260930_v1'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    D020 / 'run_trace.py': '69d0c27a9cf6038dca40a1b1e79763d1f64101d127e571a8ad2a3acc0baa47f6',
    PRIOR / 'preregistered.json': '856fba3fe2ee485297fef0247ca20c9b0b1f6d208335507fd06f086ef58e5a15',
    PRIOR / 'inventory.json': '63b4161c6b520c743632e037fe1acb72031327dca3e39abf0d477d5abc91c760',
    PRIOR / 'exit.json': '1cc060804dfb3928ba4f099134f11519b9bce1574265fd9123aa7f2305a36487',
}
NEW_FILES = ('THEORY.md', 'INTERFACE_LIMITS.md', 'PREREG.md', 'capacity.py', 'test_capacity.py',
             'run_reference.py')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmSize:', 'VmRSS:', 'VmHWM:')):
            key, value, unit = line.split()
            if unit != 'kB':
                raise ValueError('unexpected proc memory unit')
            values[key[:-1] + '_bytes'] = int(value) * 1024
    return values


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit supervisor opt-in required')
    RUN.mkdir(exist_ok=False)
    start, test_start = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs, helper, frozen = {}, {}, None, None
    record = dict(component_tests_passed=False, reference_qualified=False,
                  host_observations_within_caps=False,
                  complete_physical_qualification=False, native_HZ_admitted=False,
                  gpu_computation_completed=False, source_census_completed=False,
                  formal_gain=0, initial_memory=memory(), worker_stage=False)
    tracemalloc.start()
    try:
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        if any(sha(path) != digest for path, digest in ANCHORS.items()):
            raise ValueError('frozen D020 authority drift')
        # Authenticated module imports define helpers only. Never call old
        # main(), worker(), save(), or numerical diagnostic entry points.
        inherited20 = load(D020 / 'run_trace.py', 'd024_d020_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited20.ANCHORS.items()):
            raise ValueError('frozen D017 authority drift')
        inherited17 = load(inherited20.D017 / 'gpu_preflight.py',
                           'd024_d017_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited17.ANCHORS.items()):
            raise ValueError('older helper authority drift')
        helper = load(inherited17.OLD, 'd024_old_readonly_helpers')
        prior = json.loads((PRIOR / 'preregistered.json').read_text())
        done = json.loads((PRIOR / 'exit.json').read_text())
        inventory = json.loads((PRIOR / 'inventory.json').read_text())
        if (done['component_tests_passed'] is not True or done['tests_exit'] != 0
                or done['tests_count'] != 3739 or done['trace_complete'] is not False
                or done['initialization_succeeded'] is not False
                or done['gpu_computation_completed'] is not False
                or done['worker_record_present'] is not False or not done.get('failure')
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or inventory['count'] != 3739 or inventory['files'] != 166
                or len(prior['tests']) != 166
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D020 population or failed trace status differs')
        identities.update(prior['source_sha256'])
        inputs.update(prior['input_sha256'])
        for path, digest in ANCHORS.items():
            helper.bind(identities, path, digest)
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        for name in NEW_FILES:
            helper.bind(identities, HERE / name, sha(HERE / name))
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('inherited interpreter drift')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('original source population changed')
        older15 = json.loads((inherited17.PRIOR / 'preregistered.json').read_text())
        if helper.bind_decoder(identities) != older15['decoder_dependency_files']:
            raise ValueError('decoder dependency population changed')
        older17 = json.loads((inherited20.PRIOR / 'preregistered.json').read_text())
        # Filesystem identity checks only; this helper never imports CUDA.
        gpu_files = inherited17.gpu_dependencies(helper, identities)
        if gpu_files != older17['gpu_dependency_files']:
            raise ValueError('inherited GPU dependency population changed')
        frozen = helper.provenance()
        if frozen != prior['provenance'] or frozen['branch'] != 'redu-hz':
            raise ValueError('production provenance drift')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run identity drift')
        test_path = HERE / 'test_capacity.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 7
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)):
            raise ValueError('exact seven plain top-level tests required')
        relative = str(test_path.relative_to(ROOT))
        expected = [*inventory['nodeids'], *(relative + '::' + node.name for node in functions)]
        tests = [*prior['tests'], str(test_path)]
        if len(expected) != 3746 or len(set(expected)) != 3746 or len(tests) != 167:
            raise ValueError('complete 3746-test population differs')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                   CUDA_CACHE_PATH=str(RUN / 'cuda_cache'), TORCH_HOME=str(RUN / 'torch_home'),
                   XDG_CACHE_HOME=str(RUN / 'xdg_cache'), TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
                   TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=frozen, tests=tests, expected_nodeids=expected, selected_sources=selected,
            required_tests=3746, required_test_files=167, inherited_tests=3739,
            new_test_names=[node.name for node in functions], gpu_dependency_files=gpu_files,
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, whole_work_cap=256_000_000,
            branch_work_cap=200_000_000, host_memory_cap_bytes=MEMORY_CAP,
            retained_entry_cap=64_000_000, rational_bit_cap=512,
            summary_reserve_bytes=RESERVE, prior_failure=done['failure'],
            scope='full inherited CPU test gate plus exact rational phase capacity reference',
            no_original_model_decode=True, caches_relocated_to_new_run=True,
            cuda_visible_devices='', worker_stage=False,
            complete_physical_qualification=False, native_HZ_admitted=False,
            gpu_computation_completed=False, formal_gain=0))
        print(json.dumps(dict(event='frozen_before_import', tests=3746, files=167)), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_start = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3746:
            raise ValueError('complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=len(ids), files=167))
        remaining = 60 - (time.monotonic() - test_start)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined budget')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_start,
                      tests_exit=tested.returncode, tests_count=len(ids))
        test_start = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (tested.returncode or sorted(actual) != sorted(expected) or record['test_wall_s'] > 60
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('full component gate failed')
        record['component_tests_passed'] = True
    except Exception as exc:
        if test_start is not None:
            record['test_wall_s'] = time.monotonic() - test_start
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        if helper is not None:
            record['source_drift'], record['input_drift'] = helper.drift(identities), helper.drift(inputs)
            try:
                record['provenance_drift'] = frozen is not None and helper.provenance() != frozen
            except Exception as exc:
                record['provenance_drift'] = True
                record['provenance_check_failure'] = str(exc)
            if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
                record['component_tests_passed'] = False
                record.setdefault('failure', dict(type='ValueError', reason='final identity drift'))
        record['artifacts'] = {str(path.relative_to(RUN)): sha(path)
                               for path in RUN.rglob('*') if path.is_file()}
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
        host_ok = growth + RESERVE <= MEMORY_CAP and peak + metadata + RESERVE <= MEMORY_CAP
        record.update(wall_s=time.monotonic() - start, final_memory=memory(),
                      rss_highwater_growth_bytes=growth, traced_peak_bytes=peak,
                      tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                      host_observations_within_caps=host_ok,
                      memory_scope='supervisor only; test children retain inherited gate scope')
        if not host_ok:
            record.setdefault('failure', dict(type='MemoryError', reason='supervisor host gate exceeded'))
        record['reference_qualified'] = (record['component_tests_passed'] and host_ok
                                         and 'failure' not in record)
        record['supervisor_exit'] = 0 if record['reference_qualified'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
