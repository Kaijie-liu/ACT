"""D046 single-use complete mathematical component gate; no extra worker."""
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
RUN = EXP / 'results/d046_multiphase_envelopes_20260930_v1'
PRIOR = EXP / 'results/d044_relational_generator_20260930_v1'
D044 = HERE.parent / 'd044_relational_generator_20260930'
D038 = HERE.parent / 'd038_descendant_census_20260930'
D017 = HERE.parent / 'd017_applicability_gpu_20260930'
D015 = HERE.parent / 'd015_batch_binding_20260928_v2'
OLDER = EXP / 'results/d015_source_shielding_20260928_v2/preregistered.json'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FREEZE = HERE / 'freeze.json'
NEW_FILES = ('PREREG.md', 'THEORY.md', 'CONTROL.md', 'multiphase.py', 'test_multiphase.py', 'run_math.py')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    HERE.parent / 'd042_bounded_relational_transfer_20260930/THEORY.md': 'b5d172b2ce7dd60bfc91eebae901824be2bce61a6d9c4d422a5c56496e0f73e7',
    HERE.parent / 'd042_bounded_relational_transfer_20260930/CONTROL.md': '054bf8740b3612fea03a00b66c2b5825fc5e91dd9e12a57dd58226f3314a6566',
    PRIOR / 'preregistered.json': 'e7ade555ca971a3226657d5079519057b7b64bfea5327143b2ed182007ab8b85',
    PRIOR / 'inventory.json': '6eafb2c5588e9637dbdb991b8a5ccc6e36f56d7375698662f8b20423fdc92a41',
    PRIOR / 'exit.json': '3275ce7f6debecd3328748b217757c414c3b9a58d3349529048c9e974ddb42d5',
    D044 / 'freeze.json': '44a4c4aba18ba81d92e27aa2d9d6a25d1cba3f71270bca5be3ee90373ae4152d',
    D044 / 'run_math.py': '72528e0220cf840c50abd3777c28e0a34e3b1fe954955b35922d9272fe1b7849',
    D038 / 'freeze.json': '8f24ae5d3e7c674a66e80874e6f607733aeff9d46174434a8a68ce723cf8fc73',
    D038 / 'run_reference.py': 'f4abd1608c70aa308a590307b8f3369eff2ab11b217d54be597c99bb77c0efb5',
    D017 / 'gpu_preflight.py': '902437443f6847482d21d0af227b7fc36234868444f11f4dfc580f2abbf89c01',
    D015 / 'run_v2.py': '8ff9d296ce56b8dd9481d0c8367484b8ff37148db5dfa522062136d9d04e12ce',
    OLDER: '4697d2bbc1b86e7732e825e0e8a3b6d9c595fc275516688d746e738d2bba50cd',
}
SCOPE = ('complete inherited component tests plus eight multiphase mathematics tests; '
         'no additional worker, source census, native HZ, GPU or physical qualification')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path, cap=8 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('missing, linked or oversized metadata: ' + str(path))
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


def check_freeze():
    frozen = read_json(FREEZE, RESERVE)
    sources, names = frozen.get('source_sha256'), frozen.get('new_test_names')
    if (frozen.get('schema') != 'd046_frozen_v1'
            or frozen.get('required_tests') != 3777 or frozen.get('required_test_files') != 171
            or type(sources) is not dict or set(sources) != {str(HERE / name) for name in NEW_FILES}
            or type(names) is not list or len(names) != 8
            or any(type(name) is not str or not name.startswith('test_') for name in names)
            or len(set(names)) != 8):
        raise ValueError('frozen six-file/eight-test contract differs')
    for path, digest in sources.items():
        if (type(digest) is not str or len(digest) != 64 or Path(path).is_symlink()
                or sha(path) != digest):
            raise ValueError('new source differs from pre-execution freeze: ' + path)
    return frozen


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    helper, shared, production, log = None, None, None, None
    identities, inputs = {}, {}
    record = dict(scope=SCOPE, component_tests_passed=False, mathematical_component_gate_passed=False,
        worker_stage_registered=False, worker_launched=False, worker_exit=None,
        worker_wall_cap_s=240, host_observations_within_caps=False,
        complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
        native_HZ_admitted=False, gpu_computation_completed=False,
        source_census_completed=False, source_census_qualified=False,
        actual_phase_column_binding_verified=False, formal_gain=0)

    def emit(value):
        line = json.dumps(value, sort_keys=True, allow_nan=False)
        if log is not None:
            log.write(line + '\n')
            log.flush()
        print(line, flush=True)

    try:
        log = (RUN / 'supervisor.log').open('x')
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
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        prefreeze = check_freeze()
        identities.update(prefreeze['source_sha256'])
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in ANCHORS.items():
            if sha(path) != digest:
                raise ValueError('frozen D044/helper authority drift: ' + str(path))
            identities[str(path)] = digest
        # Authenticated stdlib-only modules: no old main, worker, writer or globals mutation.
        shared = load(D038 / 'run_reference.py', 'd046_d038_readonly_helpers')
        helper = load(D015 / 'run_v2.py', 'd046_d015_readonly_helpers')
        gpu = load(D017 / 'gpu_preflight.py', 'd046_d017_readonly_helpers')
        record['initial_memory'] = shared.memory()
        prior, done = read_json(PRIOR / 'preregistered.json'), read_json(PRIOR / 'exit.json')
        inventory = read_json(PRIOR / 'inventory.json')
        if (done['component_tests_passed'] is not True or done['tests_exit'] != 0
                or done['tests_count'] != 3769 or done['all_stages_passed'] is not True
                or done['host_observations_within_caps'] is not True or done['formal_gain'] != 0
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or prior['required_tests'] != 3769 or prior['required_test_files'] != 170
                or len(prior['tests']) != 170 or len(set(prior['tests'])) != 170
                or len(prior['expected_nodeids']) != 3769
                or len(set(prior['expected_nodeids'])) != 3769
                or inventory['count'] != 3769 or inventory['files'] != 170
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D044 passed complete component population differs')
        for path, digest in prior['source_sha256'].items():
            helper.bind(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        if any(path not in identities for path in prior['tests']):
            raise ValueError('inherited test file lacks a frozen identity')
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('frozen interpreter differs')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('three original read-only source identities differ')
        decoder = helper.bind_decoder(identities)
        if decoder != read_json(OLDER)['decoder_dependency_files']:
            raise ValueError('frozen decoder dependency population differs')
        dependencies = gpu.gpu_dependencies(helper, identities)
        if dependencies != prior['gpu_dependency_files']:
            raise ValueError('frozen GPU dependency population differs')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run source/input drift')
        test_path = HERE / 'test_multiphase.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 8 or len({node.name for node in functions}) != 8
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)
                or [node.name for node in functions] != prefreeze['new_test_names']):
            raise ValueError('exact eight frozen plain top-level new tests required')
        relative = str(test_path.relative_to(ROOT))
        tests = [*prior['tests'], str(test_path)]
        expected = [*inventory['nodeids'], *(relative + '::' + node.name for node in functions)]
        if len(tests) != 171 or len(set(tests)) != 171 or len(expected) != 3777 or len(set(expected)) != 3777:
            raise ValueError('complete 3777/171 population differs')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            selected_sources=selected, decoder_dependency_files=decoder,
            gpu_dependency_files=dependencies, freeze_path=str(FREEZE), freeze_sha256=sha(FREEZE),
            required_tests=3777, required_test_files=171, inherited_tests=3769,
            inherited_test_files=170, new_test_names=prefreeze['new_test_names'],
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            worker_stage_registered=False, scope=SCOPE, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, whole_work_cap=256_000_000,
            branch_work_cap=200_000_000, evidence_prepaid_work=40_000_000,
            retained_entry_cap=64_000_000, rational_bit_cap=512,
            caches_relocated_to_new_run=True, cuda_visible_devices='',
            complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
            native_HZ_admitted=False, gpu_computation_completed=False, formal_gain=0))
        emit(dict(event='frozen_before_candidate_import', tests=3777, files=171, scope=SCOPE))
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_started = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if (collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3777
                or len({node.split('::', 1)[0] for node in ids}) != 171):
            raise ValueError('exact complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=3777, files=171))
        remaining = 60 - (time.monotonic() - test_started)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined 60-second budget')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_started,
                      tests_exit=tested.returncode, tests_count=3777)
        test_started = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (tested.returncode or record['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('complete inherited and mathematical component gate failed')
        record['component_tests_passed'] = True
        emit(dict(event='complete_component_pass', tests=3777, files=171, wall_s=record['test_wall_s']))
    except BaseException as exc:
        if test_started is not None:
            record['test_wall_s'] = time.monotonic() - test_started
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
        emit(dict(event='failed', failure=record['failure']))
    finally:
        try:
            record['source_drift'] = [p for p, d in identities.items() if sha(p) != d]
            record['input_drift'] = [p for p, d in inputs.items() if sha(p) != d]
            record['provenance_drift'] = (production is not None
                and (helper is None or helper.provenance() != production))
            if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
                raise ValueError('final source/input/provenance drift')
        except BaseException as exc:
            record['component_tests_passed'] = False
            record['final_identity_check_failure'] = str(exc)[:4096]
            record.setdefault('failure', dict(type=type(exc).__name__, reason='final identity check failed'))
        if log is not None:
            log.close()
            log = None
        record['artifacts'] = {}
        try:
            for path in RUN.rglob('*'):
                if path.is_file():
                    record['artifacts'][str(path.relative_to(RUN))] = sha(path)
        except BaseException as exc:
            record['artifact_sealing_failure'] = str(exc)[:4096]
            record.setdefault('failure', dict(type=type(exc).__name__, reason='artifact sealing incomplete'))
        try:
            if not tracemalloc.is_tracing() or shared is None:
                raise ValueError('supervisor host telemetry unavailable')
            _, peak = tracemalloc.get_traced_memory()
            metadata = tracemalloc.get_tracemalloc_memory()
            growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
            record.update(final_memory=shared.memory(), rss_highwater_growth_bytes=growth,
                traced_peak_bytes=peak, tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                host_observations_within_caps=(growth + RESERVE <= MEMORY_CAP
                    and peak + metadata + RESERVE <= MEMORY_CAP))
        except BaseException as exc:
            record['host_observations_within_caps'] = False
            record['memory_check_failure'] = str(exc)[:4096]
        if not record['host_observations_within_caps']:
            record.setdefault('failure', dict(type='MemoryError', reason='supervisor host gate failed'))
        record.update(wall_s=time.monotonic() - started,
            memory_scope='supervisor only; pytest has AS/CPU/time limits, not full physical qualification')
        record['mathematical_component_gate_passed'] = (record['component_tests_passed']
            and record['host_observations_within_caps'] and 'failure' not in record)
        record['all_stages_passed'] = record['mathematical_component_gate_passed']
        record['supervisor_exit'] = 0 if record['all_stages_passed'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
