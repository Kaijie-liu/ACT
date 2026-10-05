"""D053 single-use mathematical gate; preserve all inherited attempt outcomes."""
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
RUN = EXP / 'results/d053_static_source_component_20260930_v1'
PRIOR = EXP / 'results/d049_mixed_source_envelopes_20260930_v1'
FAILED_CENSUS = EXP / 'results/d047_multiphase_source_census_20260930_v1'
D049 = HERE.parent / 'd049_mixed_source_envelopes_20260930'
D052 = HERE.parent / 'd052_static_source_certificates_20260930'
D038 = HERE.parent / 'd038_descendant_census_20260930'
D017 = HERE.parent / 'd017_applicability_gpu_20260930'
D015 = HERE.parent / 'd015_batch_binding_20260928_v2'
OLDER = EXP / 'results/d015_source_shielding_20260928_v2/preregistered.json'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FREEZE = HERE / 'freeze.json'
NEW_FILES = ('PREREG.md', 'THEORY.md', 'CONTROL.md', 'static_source.py',
             'test_static_source.py', 'run_math.py')
TEST_NAMES = ('test_exact_matching_and_projection', 'test_nonpoint_parameters_and_zero_phases',
              'test_strict_control_and_forward_use', 'test_disabled_identity_and_rejection')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    D052 / 'THEORY.md': 'ce51a4a3f34444aa33325d6e897431fc51ed9c39155d30163cc1c32d3fbcbff8',
    D052 / 'CONTROL.md': 'bdf2372d462a19a97e32ccc2dbf8d378af79ded69ea19a448bd451f8f6c7f34e',
    D052 / 'SHA256SUMS': '253f527daa78aaac5d3d932f2e64f0ade947d04607a8427ab7b1513462bc6a1b',
    PRIOR / 'preregistered.json': '36999d4c75dfbf96932f619b93c5f3ad54a277b8700779f66e2eef5bb4967df6',
    PRIOR / 'inventory.json': '8c37f8da1887621ff703424478fa9008e32cad40727c2b55a3c93fbb38bcdd9d',
    PRIOR / 'exit.json': 'bd8b3afe47c8dbbf2fea97a9cb710df09121c18201922f8362088c5d7c3e5801',
    D049 / 'freeze.json': 'c9e2493bfe156c055c6378337a494b4035ba247d54db4eb82085f721d6da918c',
    D049 / 'run_math.py': '4aee6e0fe34d9f6ab45c3987a063aba8edc55820926179c607e4af702a8588e7',
    D038 / 'freeze.json': '8f24ae5d3e7c674a66e80874e6f607733aeff9d46174434a8a68ce723cf8fc73',
    D038 / 'run_reference.py': 'f4abd1608c70aa308a590307b8f3369eff2ab11b217d54be597c99bb77c0efb5',
    D017 / 'gpu_preflight.py': '902437443f6847482d21d0af227b7fc36234868444f11f4dfc580f2abbf89c01',
    D015 / 'run_v2.py': '8ff9d296ce56b8dd9481d0c8367484b8ff37148db5dfa522062136d9d04e12ce',
    OLDER: '4697d2bbc1b86e7732e825e0e8a3b6d9c595fc275516688d746e738d2bba50cd',
}
SCOPE = ('complete inherited component tests plus four static-source mathematics tests; '
         'D049 mathematical success and nested D047 failed census stay unchanged; '
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
    if (frozen.get('schema') != 'd053_frozen_v1'
            or frozen.get('required_tests') != 3789 or frozen.get('required_test_files') != 174
            or type(sources) is not dict or set(sources) != {str(HERE / name) for name in NEW_FILES}
            or type(names) is not list or names != list(TEST_NAMES)):
        raise ValueError('frozen six-file/four-test contract differs')
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
                raise ValueError('frozen D049/helper authority drift: ' + str(path))
            identities[str(path)] = digest
        # Authenticated stdlib-only modules: no old main, worker, writer or globals mutation.
        shared = load(D038 / 'run_reference.py', 'd053_d038_readonly_helpers')
        helper = load(D015 / 'run_v2.py', 'd053_d015_readonly_helpers')
        gpu = load(D017 / 'gpu_preflight.py', 'd053_d017_readonly_helpers')
        record['initial_memory'] = shared.memory()
        prior, done = read_json(PRIOR / 'preregistered.json'), read_json(PRIOR / 'exit.json')
        inventory = read_json(PRIOR / 'inventory.json')
        if (done['component_tests_passed'] is not True
                or done['mathematical_component_gate_passed'] is not True
                or done['tests_exit'] != 0 or done['tests_count'] != 3785
                or done['all_stages_passed'] is not True or done.get('failure')
                or done['worker_stage_registered'] is not False
                or done['worker_launched'] is not False or done['worker_exit'] is not None
                or done['source_census_completed'] is not False
                or done['source_census_qualified'] is not False
                or done['host_observations_within_caps'] is not True or done['formal_gain'] != 0
                or done['native_HZ_admitted'] is not False or done['gpu_computation_completed'] is not False
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or prior['required_tests'] != 3785 or prior['required_test_files'] != 173
                or len(prior['tests']) != 173 or len(set(prior['tests'])) != 173
                or len(prior['expected_nodeids']) != 3785
                or len(set(prior['expected_nodeids'])) != 3785
                or inventory['count'] != 3785 or inventory['files'] != 173
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D049 complete passed mathematical gate differs')
        prior_attempt = prior['prior_attempt']
        if (type(prior_attempt) is not dict or prior_attempt != done['prior_attempt']
                or prior.get('prior_census_failure_preserved') is not True
                or prior_attempt.get('path') != str(FAILED_CENSUS)
                or prior_attempt.get('tests_count') != 3781 or prior_attempt.get('test_files') != 172
                or prior_attempt.get('mathematical_component_gate_passed') is not True
                or prior_attempt.get('all_stages_passed') is not False
                or prior_attempt.get('source_census_completed') is not False
                or prior_attempt.get('source_census_qualified') is not False
                or prior_attempt.get('worker_exit') != 1 or prior_attempt.get('formal_gain') != 0
                or not prior_attempt.get('failure') or not prior_attempt.get('worker_failure')):
            raise ValueError('nested D047 failed census record differs')
        inherited_gate = dict(path=str(PRIOR), tests_count=3785, test_files=173,
                              mathematical_component_gate_passed=True)
        record.update(prior_attempt=prior_attempt, prior_census_failure_preserved=True,
                      inherited_mathematical_gate=inherited_gate)
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
        test_path = HERE / 'test_static_source.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 4 or len({node.name for node in functions}) != 4
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)
                or [node.name for node in functions] != prefreeze['new_test_names']):
            raise ValueError('exact four frozen plain top-level new tests required')
        relative = str(test_path.relative_to(ROOT))
        tests = [*prior['tests'], str(test_path)]
        expected = [*inventory['nodeids'], *(relative + '::' + node.name for node in functions)]
        if len(tests) != 174 or len(set(tests)) != 174 or len(expected) != 3789 or len(set(expected)) != 3789:
            raise ValueError('complete 3789/174 population differs')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            selected_sources=selected, decoder_dependency_files=decoder,
            gpu_dependency_files=dependencies, freeze_path=str(FREEZE), freeze_sha256=sha(FREEZE),
            required_tests=3789, required_test_files=174, inherited_tests=3785,
            inherited_test_files=173, new_test_names=prefreeze['new_test_names'],
            prior_attempt=prior_attempt, prior_census_failure_preserved=True,
            inherited_mathematical_gate=inherited_gate,
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            worker_stage_registered=False, scope=SCOPE, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, whole_work_cap=256_000_000,
            branch_work_cap=200_000_000, evidence_prepaid_work=40_000_000,
            retained_entry_cap=64_000_000, rational_bit_cap=512,
            caches_relocated_to_new_run=True, cuda_visible_devices='',
            complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
            source_census_completed=False, source_census_qualified=False,
            actual_phase_column_binding_verified=False,
            native_HZ_admitted=False, gpu_computation_completed=False, formal_gain=0))
        emit(dict(event='frozen_before_candidate_import', tests=3789, files=174, scope=SCOPE))
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_started = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if (collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3789
                or len({node.split('::', 1)[0] for node in ids}) != 174):
            raise ValueError('exact complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=3789, files=174))
        remaining = 60 - (time.monotonic() - test_started)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined 60-second budget')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_started,
                      tests_exit=tested.returncode, tests_count=3789)
        test_started = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (tested.returncode or record['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('complete inherited and mathematical component gate failed')
        record['component_tests_passed'] = True
        emit(dict(event='complete_component_pass', tests=3789, files=174, wall_s=record['test_wall_s']))
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
