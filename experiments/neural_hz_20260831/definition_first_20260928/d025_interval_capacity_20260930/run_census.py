"""Single-use default-off full test gate and bounded CPU source census."""
import ast
import hashlib
import importlib.util
import json
import math
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
D024 = HERE.parent / 'd024_local_phase_capacity_20260930'
PRIOR = EXP / 'results/d024_local_phase_capacity_20260930_v1'
RUN = EXP / 'results/d025_interval_capacity_20260930_v1'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    D024 / 'run_reference.py': '904d041461b6f9603d1f61d5c3814f0ea112d09778d677d95134e36d7bae697b',
    PRIOR / 'preregistered.json': 'b14a4e7918f144869763c965ea13b593e23dd46a239b9c98d3baa577eb9eb5f4',
    PRIOR / 'inventory.json': 'fb3f37b1745bdb69660b957aa9ab3b554c38b9c5e9abfaea4ca7aa5359cd87c8',
    PRIOR / 'exit.json': '7ead43e7caafc2ff298de9f7389e13f0b57dbe5863766842de124c5a030f7ea2',
}
NEW_FILES = ('THEORY.md', 'PREREG.md', 'interval_capacity.py',
             'test_interval_capacity.py', 'census.py', 'evidence.py', 'run_census.py')
EVIDENCE_FILES = ('complete_0.json', 'complete_1.json', 'complete_2.json')


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


def check_diagnostic(result, selected):
    """Validate this worker's evidence; never call an older diagnostic stage."""
    if (result.get('source_census_completed') is not True
            or result.get('memory_gate_passed') is not True or result.get('failure')):
        raise ValueError('source census or measured worker gate failed')
    fields = ('rss_highwater_growth_bytes', 'final_summary_reserve_bytes',
              'traced_peak_bytes', 'tracer_metadata_bytes', 'retained_entries',
              'whole_work_used', 'branch_work_used', 'evidence_work_used')
    if any(type(result.get(key)) is not int or result[key] < 0 for key in fields):
        raise ValueError('invalid worker resource accounting')
    reserve = result['final_summary_reserve_bytes']
    if (reserve < RESERVE or result['rss_highwater_growth_bytes'] + reserve > MEMORY_CAP
            or result['traced_peak_bytes'] + result['tracer_metadata_bytes'] + reserve > MEMORY_CAP
            or result['retained_entries'] > 64_000_000
            or result['whole_work_used'] > 256_000_000
            or result['branch_work_used'] > 200_000_000
            or result['evidence_work_used'] > 40_000_000):
        raise ValueError('unchanged worker memory, entry or work limit exceeded')
    wall = result.get('wall_s')
    if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 <= wall <= 240:
        raise ValueError('worker wall limit exceeded')
    if any(type(result.get(key)) is not int or result[key] != 0
           for key in ('diagnostic_solver_calls', 'model_forward_calls',
                       'new_benchmark_solves', 'formal_gain')):
        raise ValueError('source census cannot execute a solver/forward or claim solves')
    if any(result.get(key) is not False for key in
           ('native_HZ_admitted', 'gpu_computation_completed', 'complete_physical_qualification')):
        raise ValueError('source census cannot claim native, GPU or whole-physical admission')
    models = result.get('models')
    if type(models) is not list or len(models) != 3:
        raise ValueError('three completed model records required')
    if result['evidence_work_used'] != sum(model['evidence_work'] for model in models):
        raise ValueError('global evidence work differs from model ledger')
    for name, model, source in zip(EVIDENCE_FILES, models, selected):
        path = RUN / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('complete source evidence missing: ' + name)
        if (model['model'] != source['model_relative_path'] or model['evidence_file'] != name
                or model['evidence_bytes'] != path.stat().st_size
                or model['evidence_sha256'] != sha(path)
                or model['summary']['rows'] != model['summary']['expected_rows']):
            raise ValueError('model evidence manifest or receiver population differs')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit supervisor opt-in required')
    RUN.mkdir(exist_ok=False)
    start, test_start = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs, helper, frozen = {}, {}, None, None
    record = dict(component_tests_passed=False, source_census_completed=False,
                  source_census_qualified=False, worker_launched=False,
                  host_observations_within_caps=False,
                  complete_physical_qualification=False, native_HZ_admitted=False,
                  gpu_computation_completed=False, formal_gain=0,
                  initial_memory=memory())
    tracemalloc.start()
    try:
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        if any(sha(path) != digest for path, digest in ANCHORS.items()):
            raise ValueError('frozen D024 authority drift')
        inherited24 = load(D024 / 'run_reference.py', 'd025_d024_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited24.ANCHORS.items()):
            raise ValueError('frozen D020 authority drift')
        inherited20 = load(inherited24.D020 / 'run_trace.py', 'd025_d020_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited20.ANCHORS.items()):
            raise ValueError('frozen D017 authority drift')
        inherited17 = load(inherited20.D017 / 'gpu_preflight.py', 'd025_d017_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited17.ANCHORS.items()):
            raise ValueError('older helper authority drift')
        helper = load(inherited17.OLD, 'd025_old_readonly_helpers')
        prior = json.loads((PRIOR / 'preregistered.json').read_text())
        done = json.loads((PRIOR / 'exit.json').read_text())
        inventory = json.loads((PRIOR / 'inventory.json').read_text())
        if (done['component_tests_passed'] is not True or done['reference_qualified'] is not True
                or done['tests_exit'] != 0 or done['tests_count'] != 3746
                or done['host_observations_within_caps'] is not True
                or done['source_census_completed'] is not False
                or done['gpu_computation_completed'] is not False or done.get('failure')
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or inventory['count'] != 3746 or inventory['files'] != 167
                or len(prior['tests']) != 167
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D024 component qualification or exact population differs')
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
            raise ValueError('fixed three-source population changed')
        older15 = json.loads((inherited17.PRIOR / 'preregistered.json').read_text())
        if helper.bind_decoder(identities) != older15['decoder_dependency_files']:
            raise ValueError('decoder dependency population changed')
        gpu_files = inherited17.gpu_dependencies(helper, identities)
        if gpu_files != prior['gpu_dependency_files']:
            raise ValueError('inherited GPU dependency population changed')
        frozen = helper.provenance()
        if frozen != prior['provenance'] or frozen['branch'] != 'redu-hz':
            raise ValueError('production provenance drift')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run source/input identity drift')
        test_path = HERE / 'test_interval_capacity.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 7
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)):
            raise ValueError('exact seven plain top-level new tests required')
        relative = str(test_path.relative_to(ROOT))
        expected = [*inventory['nodeids'], *(relative + '::' + node.name for node in functions)]
        tests = [*prior['tests'], str(test_path)]
        if len(expected) != 3753 or len(set(expected)) != 3753 or len(tests) != 168:
            raise ValueError('complete 3753-test population differs')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                   CUDA_CACHE_PATH=str(RUN / 'cuda_cache'), TORCH_HOME=str(RUN / 'torch_home'),
                   XDG_CACHE_HOME=str(RUN / 'xdg_cache'), TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
                   TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=frozen, tests=tests, expected_nodeids=expected, selected_sources=selected,
            required_tests=3753, required_test_files=168, inherited_tests=3746,
            new_test_names=[node.name for node in functions], gpu_dependency_files=gpu_files,
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            host_memory_cap_bytes=MEMORY_CAP, retained_entry_cap=64_000_000,
            rational_bit_cap=512, summary_reserve_bytes=RESERVE,
            evidence_prepaid_work=40_000_000, expected_evidence_files=EVIDENCE_FILES,
            scope='full inherited tests then fixed three-model first-bank capacity census',
            caches_relocated_to_new_run=True, cuda_visible_devices='',
            complete_physical_qualification=False, native_HZ_admitted=False,
            gpu_computation_completed=False, formal_gain=0))
        print(json.dumps(dict(event='frozen_before_import', tests=3753, files=168)), flush=True)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_start = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3753:
            raise ValueError('complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=len(ids), files=168))
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
        print(json.dumps(dict(event='full_component_pass', tests=3753,
                              wall_s=record['test_wall_s'])), flush=True)
        if helper.drift(identities) or helper.drift(inputs) or helper.provenance() != frozen:
            raise ValueError('identity drift before source worker')
        record['worker_launched'] = True
        with (RUN / 'diagnostic.log').open('x') as stream:
            result = subprocess.run([sys.executable, '-B', str(HERE / 'census.py'), '--enabled'],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=240, preexec_fn=limits)
        record['worker_exit'] = result.returncode
        path = RUN / 'diagnostic.json'
        record['worker_diagnostic_present'] = path.is_file()
        if not path.is_file():
            raise ValueError('worker diagnostic missing; no invented completion')
        report = json.loads(path.read_text())
        record['source_census_completed'] = report.get('source_census_completed') is True
        record['worker_memory_gate_passed'] = report.get('memory_gate_passed') is True
        if result.returncode != 0:
            record['worker_failure'] = report.get('failure')
            raise ValueError('source worker failed; partial evidence retained')
        check_diagnostic(report, selected)
        record['source_census_qualified'] = True
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
                record['component_tests_passed'] = record['source_census_qualified'] = False
                record.setdefault('failure', dict(type='ValueError', reason='final identity drift'))
        record['artifacts'] = {str(path.relative_to(RUN)): sha(path)
                               for path in RUN.rglob('*') if path.is_file()}
        record['retained_evidence_files'] = {
            name: dict(bytes=(RUN / name).stat().st_size, sha256=record['artifacts'][name])
            for name in EVIDENCE_FILES if (RUN / name).is_file()}
        record['worker_diagnostic_present'] = (RUN / 'diagnostic.json').is_file()
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
        host_ok = growth + RESERVE <= MEMORY_CAP and peak + metadata + RESERVE <= MEMORY_CAP
        record.update(wall_s=time.monotonic() - start, final_memory=memory(),
                      rss_highwater_growth_bytes=growth, traced_peak_bytes=peak,
                      tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                      host_observations_within_caps=host_ok,
                      memory_scope='supervisor and source worker measured separately; no aggregate qualification')
        if not host_ok:
            record['source_census_qualified'] = False
            record.setdefault('failure', dict(type='MemoryError', reason='supervisor host gate exceeded'))
        record['all_stages_passed'] = (record['component_tests_passed']
            and record['source_census_qualified'] and host_ok and 'failure' not in record)
        record['supervisor_exit'] = 0 if record['all_stages_passed'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
