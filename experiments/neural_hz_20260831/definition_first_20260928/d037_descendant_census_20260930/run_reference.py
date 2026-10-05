"""D037 single-use archive census; no old draft, model or GPU execution."""
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
D036 = HERE.parent / 'd036_descendant_amplitude_20260930'
PRIOR = EXP / 'results/d025_interval_capacity_20260930_v1'
RUN = EXP / 'results/d037_descendant_census_20260930_v1'
FREEZE = HERE / 'freeze.json'
ARCHIVE = PRIOR / 'complete_0.json'
ARCHIVE_SHA = 'fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0'
ARCHIVE_BYTES = 12_062_002
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    D036 / 'THEORY.md': '09547270ac8d613ebeb18298b2f07acc64aaa058492be43d257828a41034bb23',
    D036 / 'CHECKPOINT.md': '66e08405734093cb72d13ed1089ab4bdbdcda680175437b3c4aef790179db266',
    D036 / 'SHA256SUMS': '4dc52e6e50c89dacb4d55cfa721891ea568f9ab3ea314c90a7a26c1cdeb350da',
    D025 / 'run_census.py': '7e8f09c585e9b662baf1dd0160f21004e332f05b4d85d0aedb76159b99172cac',
    PRIOR / 'preregistered.json': 'c23b0541c696375224a6aae6b530ecce9609cece8806b16d53e1bda517860a4a',
    PRIOR / 'inventory.json': '32a0b157cda8cb4c0bb4947bbb9a5b8264290cc8d50172c98e08e446dd623239',
    PRIOR / 'exit.json': '634c09d1ffdeeafa4bedf3b16c7e1febb058301b69340f1b83f306c0e40be2aa',
}
NEW_FILES = ('PREREG.md', 'THEORY.md', 'amplitude.py', 'test_amplitude.py',
             'archive_worker.py', 'run_reference.py')
POSITIONS = ((0, 0), (0, 31), (16, 16), (31, 0), (31, 31))
POPULATION = dict(canonical_edges=184320, valid_edges=102400, padding_edges=81920)
COUNT_FIELDS = (*POPULATION, 'potential_rows', 'potential_nnz', 'potential_edges')
FORMULA = 'amplitude.compile_receiver:d037_interval_lowercuts_v1'
SCOPE = ('archive-only complete CIFAR100-large first-bank 320 rows and 576 slots each; '
         'not a three-model qualification, production phase binding, or verification gain')



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


def check_freeze():
    if FREEZE.is_symlink() or not FREEZE.is_file() or not 0 < FREEZE.stat().st_size <= RESERVE:
        raise ValueError('root-authored small freeze.json required before first execution')
    frozen = json.loads(FREEZE.read_text())
    expected_paths = {str(HERE / name) for name in NEW_FILES}
    sources = frozen.get('source_sha256')
    names = frozen.get('new_test_names')
    if (frozen.get('schema') != 'd037_frozen_v1'
            or type(sources) is not dict or set(sources) != expected_paths
            or frozen.get('required_tests') != 3759 or frozen.get('required_test_files') != 169
            or type(names) is not list or len(names) != 6 or len(set(names)) != 6
            or any(type(name) is not str or not name.startswith('test_') for name in names)):
        raise ValueError('frozen six-file/six-test contract differs')
    for path, digest in sources.items():
        if (type(digest) is not str or len(digest) != 64 or Path(path).is_symlink()
                or sha(path) != digest):
            raise ValueError('new source differs from pre-execution freeze: ' + path)
    return frozen


def count_summary(summary):
    if type(summary) is not dict or any(type(summary.get(key)) is not int
            or summary[key] < 0 for key in COUNT_FIELDS):
        raise ValueError('invalid complete population/count summary')
    if (summary['canonical_edges'] != summary['valid_edges'] + summary['padding_edges']
            or summary['potential_edges'] > summary['valid_edges']
            or summary['potential_rows'] > 2 * summary['potential_edges']
            or summary['potential_rows'] < summary['potential_edges']
            or summary['potential_nnz'] > 6 * summary['potential_edges']):
        raise ValueError('potential count or coordinate-nnz upper bound differs')
    counts = summary.get('row_unresolved_counts')
    if (type(counts) is not list or len(counts) != 4
            or any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) != summary['potential_rows']
            or sum(count * weight for count, weight in zip(counts, (2, 3, 3, 4)))
                != summary['potential_nnz']):
        raise ValueError('unresolved row masks/counts disagree')
    return summary


def check_diagnostic(result, identities):
    if (result.get('archive_completed') is not True
            or result.get('memory_gate_passed') is not True or result.get('failure')):
        raise ValueError('archive census or measured worker gate failed')
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
        raise ValueError('archive census cannot execute solver/forward or claim solves')
    if any(result.get(key) is not False for key in
           ('native_HZ_admitted', 'gpu_computation_completed', 'complete_physical_qualification',
            'source_census_qualified', 'actual_phase_column_binding_verified')):
        raise ValueError('archive census cannot claim broader qualification or native binding')
    if (result.get('binding_mathematical_only') is not True
            or any(type(result.get(key)) is not int or result[key] != 320
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
    if (evidence.get('schema') != 'd037_descendant_census_v1'
            or evidence.get('rule') != 'd036_interval_lowercuts_all_canonical_slots'
            or evidence.get('formula_ref') != FORMULA
            or evidence.get('archive_sha256') != ARCHIVE_SHA
            or evidence.get('archive_path') != str(ARCHIVE)
            or evidence.get('kernel_sha256') != identities[str(HERE / 'amplitude.py')]
            or evidence.get('binding_mathematical_only') is not True
            or evidence.get('actual_phase_column_binding_verified') is not False
            or any(type(evidence.get(key)) is not int or evidence[key] != 320
                   for key in ('expected_rows', 'completed_rows'))
            or type(rows) is not list or len(rows) != 320):
        raise ValueError('sealed evidence population, formula, or binding differs')
    count_summary(result)
    summary = count_summary(evidence.get('summary'))
    if any(type(summary.get(key)) is not int or summary[key] != 320
           for key in ('expected_rows', 'completed_rows')):
        raise ValueError('sealed complete receiver count differs')
    for key, value in POPULATION.items():
        if summary[key] != value:
            raise ValueError('sealed complete edge population differs: ' + key)
    if any(result.get(key) != summary[key] for key in (*COUNT_FIELDS, 'row_unresolved_counts')):
        raise ValueError('diagnostic and complete evidence counts disagree')
    totals, row_totals, observed = dict.fromkeys(COUNT_FIELDS, 0), [0, 0, 0, 0], []
    for row in rows:
        if (type(row) is not dict or type(row.get('branch')) is not int or row['branch'] != 0
                or type(row.get('channel')) is not int or not 0 <= row['channel'] < 64
                or type(row.get('position')) is not list or len(row['position']) != 2
                or any(type(value) is not int for value in row['position'])):
            raise ValueError('invalid complete receiver row identity')
        position, channel = tuple(row['position']), row['channel']
        if (position not in POSITIONS or row.get('input_window_ref') != [0, *position]
                or row.get('receiver_coefficients_ref') != [0, channel]
                or row.get('formula_ref') != FORMULA):
            raise ValueError('row lossless original-input/formula reference differs')
        counts = count_summary(row.get('summary'))
        masks = row.get('row_masks')
        if type(masks) is not list or len(masks) != 576:
            raise ValueError('all 576 canonical row masks must be retained')
        valid, bits, potential_edges = 0, [0, 0, 0, 0], 0
        for offset, mask in enumerate(masks):
            if type(mask) is not int or not 0 <= mask < 16:
                raise ValueError('invalid unresolved row mask')
            # The authenticated worker verifies the frozen 64x3x3, pad-one,
            # stride-one 32x32 Conv geometry before these masks are produced.
            ky, kx = divmod(offset % 9, 3)
            real = 0 <= position[0] + ky - 1 < 32 and 0 <= position[1] + kx - 1 < 32
            valid += int(real)
            if not real and mask:
                raise ValueError('padding cannot acquire a source bit or unresolved row')
            members = [int(bool(mask & (1 << bit))) for bit in range(4)]
            if sum(members) > 2 or sum(x * w for x, w in zip(members, (2, 3, 3, 4))) > 6:
                raise ValueError('unresolved mask exceeds structural per-edge bound')
            bits = [left + right for left, right in zip(bits, members)]
            potential_edges += int(bool(mask))
        if (counts['canonical_edges'] != 576 or counts['valid_edges'] != valid
                or counts['padding_edges'] != 576 - valid
                or counts['potential_edges'] != potential_edges
                or counts['row_unresolved_counts'] != bits):
            raise ValueError('receiver summary differs from its complete masks')
        for key in COUNT_FIELDS:
            totals[key] += counts[key]
        row_totals = [left + right for left, right in zip(row_totals, bits)]
        observed.append((position, channel))
    expected = {(position, channel) for position in POSITIONS for channel in range(64)}
    if len(set(observed)) != 320 or set(observed) != expected:
        raise ValueError('sealed receiver rows omit or duplicate the frozen population')
    if any(totals[key] != summary[key] for key in COUNT_FIELDS) or row_totals != summary['row_unresolved_counts']:
        raise ValueError('whole census differs from sum of all complete rows')


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
                  archive_sha256=ARCHIVE_SHA, scope=SCOPE, binding_mathematical_only=True,
                  actual_phase_column_binding_verified=False, potential_counts_only=True)

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
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        prefreeze = check_freeze()
        record['freeze_sha256'] = sha(FREEZE)
        if any(sha(path) != digest for path, digest in ANCHORS.items()):
            raise ValueError('frozen D025/D036 authority drift')
        # Authenticated imports define helpers only: never invoke old mains,
        # workers, numerical stages, or writers. Candidate imports occur in pytest.
        inherited25 = load(D025 / 'run_census.py', 'd037_d025_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited25.ANCHORS.items()):
            raise ValueError('frozen D024 authority drift')
        inherited24 = load(inherited25.D024 / 'run_reference.py', 'd037_d024_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited24.ANCHORS.items()):
            raise ValueError('frozen D020 authority drift')
        inherited20 = load(inherited24.D020 / 'run_trace.py', 'd037_d020_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited20.ANCHORS.items()):
            raise ValueError('frozen D017 authority drift')
        inherited17 = load(inherited20.D017 / 'gpu_preflight.py', 'd037_d017_readonly_helpers')
        if any(sha(path) != digest for path, digest in inherited17.ANCHORS.items()):
            raise ValueError('older helper authority drift')
        helper = load(inherited17.OLD, 'd037_old_readonly_helpers')
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
            helper.bind(identities, HERE / name, prefreeze['source_sha256'][str(HERE / name)])
        helper.bind(identities, FREEZE, record['freeze_sha256'])
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
        test_path = HERE / 'test_amplitude.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 6 or len({node.name for node in functions}) != 6
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)):
            raise ValueError('exact six plain top-level new tests required')
        if [node.name for node in functions] != prefreeze['new_test_names']:
            raise ValueError('AST test population differs from the root pre-execution freeze')
        relative = str(test_path.relative_to(ROOT))
        expected = [*inventory['nodeids'], *(relative + '::' + node.name for node in functions)]
        tests = [*prior['tests'], str(test_path)]
        if len(expected) != 3759 or len(set(expected)) != 3759 or len(tests) != 169:
            raise ValueError('complete 3759-test population differs')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=frozen, tests=tests, expected_nodeids=expected, selected_sources=selected,
            required_tests=3759, required_test_files=169, inherited_tests=3753,
            new_test_names=[node.name for node in functions], gpu_dependency_files=gpu_files,
            freeze_path=str(FREEZE), freeze_sha256=record['freeze_sha256'],
            binding_mathematical_only=True, actual_phase_column_binding_verified=False,
            potential_counts_only=True, **POPULATION,
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
        emit(dict(event='frozen_before_import', tests=3759, files=169, scope=SCOPE))
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_start = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if (collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3759
                or len({node.split('::', 1)[0] for node in ids}) != 169):
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
        emit(dict(event='full_component_pass', tests=3759, wall_s=record['test_wall_s']))
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
        check_diagnostic(report, identities)
        record['archive_qualified'] = True
        record['summary'] = {key: report[key] for key in (*COUNT_FIELDS, 'row_unresolved_counts')}
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
