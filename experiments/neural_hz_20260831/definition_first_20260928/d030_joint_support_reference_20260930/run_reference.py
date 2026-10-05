"""Single-use full component gate and isolated archived-source reference study."""
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
D025 = HERE.parent / 'd025_interval_capacity_20260930'
D029 = HERE.parent / 'd029_conditional_observations_20260930'
PRIOR = EXP / 'results/d025_interval_capacity_20260930_v1'
RUN = EXP / 'results/d030_joint_support_reference_20260930_v1'
ARCHIVE = PRIOR / 'complete_0.json'
ARCHIVE_SHA = 'fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0'
ARCHIVE_BYTES = 12_062_002
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    D025 / 'run_census.py': '7e8f09c585e9b662baf1dd0160f21004e332f05b4d85d0aedb76159b99172cac',
    PRIOR / 'preregistered.json': 'c23b0541c696375224a6aae6b530ecce9609cece8806b16d53e1bda517860a4a',
    PRIOR / 'inventory.json': '32a0b157cda8cb4c0bb4947bbb9a5b8264290cc8d50172c98e08e446dd623239',
    PRIOR / 'exit.json': '634c09d1ffdeeafa4bedf3b16c7e1febb058301b69340f1b83f306c0e40be2aa',
    D029 / 'THEORY.md': '65879b0d8d8a4eedd8e204f2d53a0fc8814e907677c0efb3bdfec7e4a2f01bda',
    D029 / 'LIMITS.md': '905f5891955bf774076459d760096fbb3a32e9e987c8b8aa610d292aa73c4910',
    D029 / 'CHECKPOINT.md': '6d2edba0744dba4ed2e78f2a05e4aa827d243a2eb9dfa796ca3a6592449608f1',
    D029 / 'SHA256SUMS': '1d55b7150333443e78e5432de084ac35201fe5e1ca892cc1f6069668096a3ade',
}
NEW_FILES = ('PREREG.md', 'joint_support.py', 'test_joint_support.py',
             'archive_worker.py', 'run_reference.py')
POSITIONS = ((0, 0), (0, 31), (16, 16), (31, 0), (31, 31))
SCOPE = 'archive-only complete CIFAR100-large first-bank 320-row reference; not a three-model qualification'


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
    if len(values) != 3:
        raise ValueError('incomplete process memory telemetry')
    return values


def check_diagnostic(result):
    if (result.get('archive_completed') is not True
            or result.get('memory_gate_passed') is not True or result.get('failure')):
        raise ValueError('archive reference or measured worker gate failed')
    fields = ('rss_highwater_growth_bytes', 'summary_reserve_bytes',
              'final_summary_reserve_bytes', 'traced_peak_bytes', 'tracer_metadata_bytes',
              'retained_entries', 'whole_work_used', 'branch_work_used', 'evidence_work_used')
    if any(type(result.get(key)) is not int or result[key] < 0 for key in fields):
        raise ValueError('invalid worker resource accounting')
    if (result['summary_reserve_bytes'] != RESERVE or result['final_summary_reserve_bytes'] != RESERVE
            or result['rss_highwater_growth_bytes'] + RESERVE > MEMORY_CAP
            or result['traced_peak_bytes'] + result['tracer_metadata_bytes'] + RESERVE > MEMORY_CAP
            or result['retained_entries'] > 64_000_000
            or result['whole_work_used'] > 256_000_000
            or result['branch_work_used'] > 200_000_000
            or result['evidence_work_used'] > 40_000_000):
        raise ValueError('unchanged worker memory, entry or work limit exceeded')
    wall = result.get('wall_s')
    if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 <= wall <= 240:
        raise ValueError('worker wall limit exceeded')
    if any(type(result.get(key)) is not int or result[key] != 0 for key in
           ('diagnostic_solver_calls', 'model_forward_calls', 'new_benchmark_solves', 'formal_gain')):
        raise ValueError('archive reference cannot execute solver/forward or claim solves')
    if any(result.get(key) is not False for key in
           ('native_HZ_admitted', 'gpu_computation_completed', 'complete_physical_qualification',
            'source_census_qualified')):
        raise ValueError('archive reference cannot claim broader qualification')
    if (any(type(result.get(key)) is not int or result[key] != 320
            for key in ('expected_rows', 'completed_rows'))
            or result.get('archive_sha256') != ARCHIVE_SHA):
        raise ValueError('complete frozen archive population or identity differs')
    path = RUN / 'complete.json'
    if (path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 40_000_000
            or result.get('evidence_file') != 'complete.json'
            or type(result.get('evidence_bytes')) is not int
            or result['evidence_bytes'] != path.stat().st_size
            or result.get('evidence_sha256') != sha(path)):
        raise ValueError('complete archive evidence file or digest differs')
    evidence = json.loads(path.read_text())
    rows = evidence.get('rows')
    if (evidence.get('archive_sha256') != ARCHIVE_SHA
            or type(rows) is not list or len(rows) != 320):
        raise ValueError('sealed evidence population or archive binding differs')
    observed = []
    for row in rows:
        if (type(row) is not dict or type(row.get('branch')) is not int or row['branch'] != 0
                or type(row.get('channel')) is not int or not 0 <= row['channel'] < 64
                or type(row.get('position')) is not list or len(row['position']) != 2
                or any(type(value) is not int for value in row['position'])):
            raise ValueError('invalid complete receiver row identity')
        observed.append((tuple(row['position']), row['channel']))
    expected = {(position, channel) for position in POSITIONS for channel in range(64)}
    if len(set(observed)) != 320 or set(observed) != expected:
        raise ValueError('sealed receiver rows omit or duplicate the frozen population')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit supervisor opt-in required')
    RUN.mkdir(exist_ok=False)
    start, test_start = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs, helper, frozen, log = {}, {}, None, None, None
    record = dict(component_tests_passed=False, archive_completed=False,
                  archive_qualified=False, worker_launched=False,
                  source_census_completed=False, source_census_qualified=False,
                  host_observations_within_caps=False, complete_physical_qualification=False,
                  native_HZ_admitted=False, gpu_computation_completed=False, formal_gain=0,
                  archive_sha256=ARCHIVE_SHA, scope=SCOPE)

    def emit(value):
        line = json.dumps(value, sort_keys=True, allow_nan=False)
        if log is not None:
            log.write(line + '\n')
            log.flush()
        print(line, flush=True)

    try:
        log = (RUN / 'supervisor.log').open('x')
        tracemalloc.start()
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        record['initial_memory'] = memory()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                   CUDA_CACHE_PATH=str(RUN / 'cuda_cache'), TORCH_HOME=str(RUN / 'torch_home'),
                   XDG_CACHE_HOME=str(RUN / 'xdg_cache'), TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
                   TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        if any(sha(path) != digest for path, digest in ANCHORS.items()):
            raise ValueError('frozen D025/D029 authority drift')
        # Authenticated imports define helpers only: never invoke old mains,
        # workers, numerical stages, or writers. Candidate imports occur in pytest.
        inherited25 = load(D025 / 'run_census.py', 'd030_d025_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited25.ANCHORS.items()):
            raise ValueError('frozen D024 authority drift')
        inherited24 = load(inherited25.D024 / 'run_reference.py', 'd030_d024_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited24.ANCHORS.items()):
            raise ValueError('frozen D020 authority drift')
        inherited20 = load(inherited24.D020 / 'run_trace.py', 'd030_d020_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited20.ANCHORS.items()):
            raise ValueError('frozen D017 authority drift')
        inherited17 = load(inherited20.D017 / 'gpu_preflight.py', 'd030_d017_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited17.ANCHORS.items()):
            raise ValueError('older helper authority drift')
        helper = load(inherited17.OLD, 'd030_old_readonly_helpers')
        prior = json.loads((PRIOR / 'preregistered.json').read_text())
        done = json.loads((PRIOR / 'exit.json').read_text())
        inventory = json.loads((PRIOR / 'inventory.json').read_text())
        if (done['component_tests_passed'] is not True or done['tests_exit'] != 0
                or done['tests_count'] != 3753 or done['host_observations_within_caps'] is not True
                or done['source_census_completed'] is not False
                or done['source_census_qualified'] is not False or done['all_stages_passed'] is not False
                or done['formal_gain'] != 0 or not done.get('failure')
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or inventory['count'] != 3753 or inventory['files'] != 168
                or len(inventory['nodeids']) != 3753 or len(set(inventory['nodeids'])) != 3753
                or len(prior['tests']) != 168 or len(set(prior['tests'])) != 168
                or prior['required_tests'] != 3753 or prior['required_test_files'] != 168
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D025 passed components/preserved failed census/population differs')
        record['prior_attempt'] = dict(component_tests_passed=True, tests_count=3753,
            source_census_qualified=False, failure=done['failure'], path=str(PRIOR))
        identities.update(prior['source_sha256'])
        inputs.update(prior['input_sha256'])
        if any(path not in identities for path in prior['tests']):
            raise ValueError('inherited test file has no frozen identity')
        for path, digest in ANCHORS.items():
            helper.bind(identities, path, digest)
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        for name in NEW_FILES:
            helper.bind(identities, HERE / name, sha(HERE / name))
        helper.bind(inputs, ARCHIVE, ARCHIVE_SHA)
        if ARCHIVE.stat().st_size != ARCHIVE_BYTES or sha(ARCHIVE) != ARCHIVE_SHA:
            raise ValueError('complete archived source identity or size differs')
        if (done['retained_evidence_files'].get('complete_0.json') !=
                dict(bytes=ARCHIVE_BYTES, sha256=ARCHIVE_SHA)):
            raise ValueError('D025 retained evidence binding differs')
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('inherited interpreter drift')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('three original source identities changed')
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
        test_path = HERE / 'test_joint_support.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 9 or len({node.name for node in functions}) != 9
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)):
            raise ValueError('exact nine plain top-level new tests required')
        relative = str(test_path.relative_to(ROOT))
        expected = [*inventory['nodeids'], *(relative + '::' + node.name for node in functions)]
        tests = [*prior['tests'], str(test_path)]
        if len(expected) != 3762 or len(set(expected)) != 3762 or len(tests) != 169:
            raise ValueError('complete 3762-test population differs')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=frozen, tests=tests, expected_nodeids=expected, selected_sources=selected,
            required_tests=3762, required_test_files=169, inherited_tests=3753,
            new_test_names=[node.name for node in functions], gpu_dependency_files=gpu_files,
            archive_path=str(ARCHIVE), archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_BYTES,
            archive_source=selected[0], expected_rows=320, positions=POSITIONS,
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            host_memory_cap_bytes=MEMORY_CAP, retained_entry_cap=64_000_000,
            rational_bit_cap=512, summary_reserve_bytes=RESERVE,
            evidence_prepaid_work=40_000_000, expected_evidence_file='complete.json', scope=SCOPE,
            caches_relocated_to_new_run=True, cuda_visible_devices='',
            source_census_qualified=False, complete_physical_qualification=False,
            native_HZ_admitted=False, gpu_computation_completed=False, formal_gain=0))
        emit(dict(event='frozen_before_import', tests=3762, files=169, scope=SCOPE))
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_start = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3762:
            raise ValueError('complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=len(ids), files=169))
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
        emit(dict(event='full_component_pass', tests=3762, wall_s=record['test_wall_s']))
        if helper.drift(identities) or helper.drift(inputs) or helper.provenance() != frozen:
            raise ValueError('identity drift before archive worker')
        record['worker_launched'] = True
        with (RUN / 'diagnostic.log').open('x') as stream:
            result = subprocess.run([sys.executable, '-B', str(HERE / 'archive_worker.py'), '--enabled'],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=240, preexec_fn=limits)
        record['worker_exit'] = result.returncode
        path = RUN / 'diagnostic.json'
        record['worker_diagnostic_present'] = path.is_file()
        if not path.is_file():
            raise ValueError('worker diagnostic missing; no invented completion')
        if path.is_symlink() or path.stat().st_size > RESERVE:
            raise ValueError('worker diagnostic exceeds the small-summary boundary')
        report = json.loads(path.read_text())
        record['archive_completed'] = report.get('archive_completed') is True
        record['worker_memory_gate_passed'] = report.get('memory_gate_passed') is True
        if result.returncode != 0:
            record['worker_failure'] = report.get('failure')
            raise ValueError('archive worker failed; partial evidence retained')
        check_diagnostic(report)
        record['archive_qualified'] = True
    except BaseException as exc:
        if test_start is not None:
            record['test_wall_s'] = time.monotonic() - test_start
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
        emit(dict(event='failed', failure=record['failure']))
    finally:
        if helper is not None:
            try:
                record['source_drift'], record['input_drift'] = helper.drift(identities), helper.drift(inputs)
                record['provenance_drift'] = frozen is not None and helper.provenance() != frozen
                if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
                    record['component_tests_passed'] = record['archive_qualified'] = False
                    record.setdefault('failure', dict(type='ValueError', reason='final identity drift'))
            except BaseException as exc:
                record['component_tests_passed'] = record['archive_qualified'] = False
                record['final_identity_check_failure'] = str(exc)
                record.setdefault('failure', dict(type=type(exc).__name__, reason='final identity check failed'))
        if log is not None:
            log.close()
            log = None
        record['artifacts'], sealing_errors = {}, []
        try:
            for path in RUN.rglob('*'):
                if path.is_file():
                    record['artifacts'][str(path.relative_to(RUN))] = sha(path)
        except BaseException as exc:
            sealing_errors.append(str(exc))
        if sealing_errors:
            record['archive_qualified'] = False
            record['artifact_sealing_errors'] = sealing_errors
            record.setdefault('failure', dict(type='ValueError', reason='artifact sealing incomplete'))
        record['retained_evidence_files'] = {}
        if 'complete.json' in record['artifacts']:
            record['retained_evidence_files']['complete.json'] = dict(
                bytes=(RUN / 'complete.json').stat().st_size,
                sha256=record['artifacts']['complete.json'])
        record['worker_diagnostic_present'] = (RUN / 'diagnostic.json').is_file()
        host_ok = False
        try:
            if not tracemalloc.is_tracing():
                raise ValueError('supervisor allocation telemetry unavailable')
            _, peak = tracemalloc.get_traced_memory()
            metadata = tracemalloc.get_tracemalloc_memory()
            growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
            host_ok = growth + RESERVE <= MEMORY_CAP and peak + metadata + RESERVE <= MEMORY_CAP
            record.update(final_memory=memory(), rss_highwater_growth_bytes=growth,
                          traced_peak_bytes=peak, tracer_metadata_bytes=metadata)
        except BaseException as exc:
            record['memory_check_failure'] = str(exc)
        record.update(wall_s=time.monotonic() - start, summary_reserve_bytes=RESERVE,
                      host_observations_within_caps=host_ok,
                      memory_scope='supervisor and archive worker measured separately; no aggregate qualification')
        if not host_ok:
            record['archive_qualified'] = False
            record.setdefault('failure', dict(type='MemoryError', reason='supervisor host gate failed'))
        record['all_stages_passed'] = (record['component_tests_passed']
            and record['archive_qualified'] and host_ok and 'failure' not in record)
        record['supervisor_exit'] = 0 if record['all_stages_passed'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
