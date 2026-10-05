"""Single-use D072 same-session gate for the unchanged D070 mathematics.

One pytest process checks its exact collection before executing all tests.
D070 remains a failed attempt; D066 success and D064/D057/D047 records keep
their original scopes. No worker, model, candidate modification or GPU run.
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
RUN = EXP / 'results/d072_single_session_gate_20261001_v1'
CANDIDATE = HERE.parent / 'd070_joint_negative_component_20261001'
FAILED_SAME = EXP / 'results/d070_joint_negative_component_20261001_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd072_single_session_gate_20261001.collection_contract')
PRIOR = EXP / 'results/d066_wide_phase_interface_20261001_v1'
FAILED_DIRECT = EXP / 'results/d057_triangle_source_census_20260930_v1'
FAILED_CENSUS = EXP / 'results/d047_multiphase_source_census_20260930_v1'
D066 = HERE.parent / 'd066_wide_phase_interface_20261001'
D064 = HERE.parent / 'd064_native_predicate_binding_20261001'
D038 = HERE.parent / 'd038_descendant_census_20260930'
D017 = HERE.parent / 'd017_applicability_gpu_20260930'
D015 = HERE.parent / 'd015_batch_binding_20260928_v2'
OLDER = EXP / 'results/d015_source_shielding_20260928_v2/preregistered.json'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FREEZE = HERE / 'freeze.json'
NEW_FILES = ('PREREG.md', 'run_math.py', 'collection_contract.py')
CANDIDATE_FILES = ('PREREG.md', 'THEORY.md', 'hinge_support.py',
                   'negative_interface.py', 'test_joint_support.py', 'run_math.py')
TEST_NAMES = ('test_primal_support_and_sparse_updates',
              'test_shared_negative_forward_control',
              'test_interval_signed_wide_dominance',
              'test_identity_default_off_and_limits')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    CANDIDATE / 'freeze.json': '409cb91c4a8f1ca7bde397132190587f220e54762402f5de622a91e28bb86825',
    FAILED_SAME / 'exit.json': '2211f568b60801bc98b259b924c92972bcc8a096fc69f7bafdfae1a569c504cf',
    FAILED_SAME / 'preregistered.json': '4493c00a58daa11e1d1182a5bb3fa623deb683100d7307fde4bb6039414443cb',
    FAILED_SAME / 'inventory.json': '1f7b4d21b26907fdc1bd1d7bb80e42bf4fd9ec343cc87e6f50ddd6bd1a0c8cca',
    PRIOR / 'preregistered.json': '7b37f77c20fe684322545f58ccbba9c8b0dc9b4be53f137afbc833188156a16e',
    PRIOR / 'inventory.json': '388a42e21633e0b7ad92c5f81d2a8c95c2c8d900db38e4bd3db26072836aac4e',
    PRIOR / 'exit.json': '6311281591973a82651146387726b149f0b1234c5bc1bfb2610dc4da2f5c6d07',
    D066 / 'freeze.json': 'ac09b9d4129211a17793273ef16053609a64ac5859bf1d02dc025cf6b53a149e',
    D066 / 'run_math.py': 'a5d69332ee36eedef75a0d48d7dcfc9281e0fe051eca08cacb260ae4e84dce82',
    D066 / 'RESULTS.md': 'e36a480a7c5a1f1b4f60327db68ab0d32772c7b1e6ab3bd6d838359a148dcf94',
    D064 / 'freeze.json': '65511f07dd2969377e5c87ab58c83047e92e919dfa690ba5f25eba233c088f75',
    D064 / 'run_math.py': 'd63d2d8a8f4d031e50332f17ad50d0406d92f22728fb0d58675b77f1a426b5d4',
    D064 / 'RESULTS.md': 'aaa5e366e1587a8db12b5826f84cfe35012ef986e849adf4649895fd3d845025',
    D038 / 'freeze.json': '8f24ae5d3e7c674a66e80874e6f607733aeff9d46174434a8a68ce723cf8fc73',
    D038 / 'run_reference.py': 'f4abd1608c70aa308a590307b8f3369eff2ab11b217d54be597c99bb77c0efb5',
    D017 / 'gpu_preflight.py': '902437443f6847482d21d0af227b7fc36234868444f11f4dfc580f2abbf89c01',
    D015 / 'run_v2.py': '8ff9d296ce56b8dd9481d0c8367484b8ff37148db5dfa522062136d9d04e12ce',
    OLDER: '4697d2bbc1b86e7732e825e0e8a3b6d9c595fc275516688d746e738d2bba50cd',
    HERE.parent / 'd065_shared_phase_tables_20261001/THEORY.md':
        '2c6c4eda5908ecdd2e95d7397939b385fa92e05a6ce09791e503ee9edec269a1',
    EXP / 'literature_mechanism_test_20261001/AUDIT.md':
        'e70fdb66223458a5db2c66a06a763db76e22cd7241e25f97b27de43733fcf6ba',
    HERE.parent / 'd068_primal_hinge_support_20261001/THEORY.md':
        'd04c6c5faa33af069ae9642da65ab87b86c4613092f1b7b4900776e1aed0ac6e',
    HERE.parent / 'd069_joint_negative_support_20261001/THEORY.md':
        '3b4f68aa5c29adbdaf25331d6d4c3c25c7bbfd2771638a49ed4aacd17a145d63',
}
SCOPE = ('unchanged D070 3813/180 mathematical population in one collection-and-test process; '
         'D070 timeout permanently failed; D066/D064 success and D057/D047 failures preserved; '
         'joint negative fixture exercised only on this new execution configuration passing; '
         'actual model binding unqualified; no worker, archive evaluation, model, GPU, LP, '
         'source census, shadow, native HZ admission, complete physical qualification or formal gain')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


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
    sources = frozen.get('source_sha256')
    if (frozen.get('schema') != 'd072_same_session_v1'
            or frozen.get('required_tests') != 3813 or frozen.get('required_test_files') != 180
            or type(sources) is not dict or set(sources) != {str(HERE / n) for n in NEW_FILES}
            or frozen.get('new_test_names') != list(TEST_NAMES)):
        raise ValueError('frozen three-file/unchanged-four-test contract differs')
    for path, digest in sources.items():
        if (type(digest) is not str or len(digest) != 64 or Path(path).is_symlink()
                or sha(path) != digest):
            raise ValueError('source differs from pre-execution freeze: ' + path)
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
        joint_negative_fixture_exercised=False, actual_model_binding_qualified=False,
        worker_stage_registered=False, worker_launched=False, worker_exit=None,
        worker_wall_cap_s=240, host_observations_within_caps=False,
        complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
        native_HZ_admitted=False, gpu_computation_completed=False,
        source_census_completed=False, source_census_qualified=False,
        actual_phase_column_binding_verified=False, formal_gain=0,
        same_process_collection_gate=True, single_pytest_process=True)

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
        frozen = check_freeze()
        identities.update(frozen['source_sha256'])
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in ANCHORS.items():
            if sha(path) != digest:
                raise ValueError('frozen prior/helper drift: ' + str(path))
            identities[str(path)] = digest
        # Authenticated stdlib-only helpers, without invoking old mains/workers/writers.
        shared = load(D038 / 'run_reference.py', 'd072_d038_readonly_helpers')
        helper = load(D015 / 'run_v2.py', 'd072_d015_readonly_helpers')
        gpu = load(D017 / 'gpu_preflight.py', 'd072_d017_readonly_helpers')
        record['initial_memory'] = shared.memory()
        prior, done = read_json(PRIOR / 'preregistered.json'), read_json(PRIOR / 'exit.json')
        inventory = read_json(PRIOR / 'inventory.json')
        if (done['component_tests_passed'] is not True
                or done['mathematical_component_gate_passed'] is not True
                or done['relational_interface_fixture_exercised'] is not True
                or done['actual_model_binding_qualified'] is not False
                or prior['relational_interface_fixture_registered'] is not True
                or prior['actual_model_binding_qualified'] is not False
                or done['tests_exit'] != 0 or done['tests_count'] != 3809
                or done['test_wall_s'] > 60 or done['all_stages_passed'] is not True
                or done['worker_stage_registered'] is not False or done['worker_launched'] is not False
                or done['worker_exit'] is not None or done['supervisor_exit'] != 0
                or done['source_census_completed'] is not False or done['source_census_qualified'] is not False
                or done['host_observations_within_caps'] is not True or done['formal_gain'] != 0
                or done['native_HZ_admitted'] is not False or done['gpu_computation_completed'] is not False
                or done['actual_phase_column_binding_verified'] is not False
                or done['complete_physical_qualification'] is not False
                or done['candidate_physical_gate_evaluated'] is not False
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or 'failure' in done
                or prior['required_tests'] != 3809 or prior['required_test_files'] != 179
                or len(prior['tests']) != 179 or len(set(prior['tests'])) != 179
                or len(prior['expected_nodeids']) != 3809 or len(set(prior['expected_nodeids'])) != 3809
                or inventory['count'] != 3809 or inventory['files'] != 179
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D066 complete mathematical/interface success contract differs')
        native_gate = prior['inherited_mathematical_gate']
        expected_native = dict(
            path=str(EXP / 'results/d064_native_predicate_binding_20261001_v1'),
            exit_sha256='8deebfb4efad74ecb1a183c6734c8fecabe14bf520e68f68bbed6bb1df33a97f',
            tests_count=3805, test_files=178, mathematical_component_gate_passed=True,
            native_predicate_fixture_exercised=True, actual_model_binding_qualified=False,
            all_stages_passed=True, worker_stage_registered=False, worker_launched=False,
            source_census_qualified=False, native_HZ_admitted=False,
            gpu_computation_completed=False, formal_gain=0)
        if native_gate != expected_native or native_gate != done['inherited_mathematical_gate']:
            raise ValueError('inherited D064 native mathematical fixture record differs')
        prior_attempt = prior['prior_attempt']
        if (type(prior_attempt) is not dict or prior_attempt != done['prior_attempt']
                or prior.get('prior_census_failure_preserved') is not True
                or done.get('prior_census_failure_preserved') is not True
                or prior_attempt.get('path') != str(FAILED_CENSUS)
                or prior_attempt.get('tests_count') != 3781 or prior_attempt.get('test_files') != 172
                or prior_attempt.get('mathematical_component_gate_passed') is not True
                or prior_attempt.get('all_stages_passed') is not False
                or prior_attempt.get('source_census_completed') is not False
                or prior_attempt.get('source_census_qualified') is not False
                or prior_attempt.get('worker_exit') != 1 or prior_attempt.get('formal_gain') != 0
                or prior_attempt.get('failure') != dict(type='ValueError',
                    reason='source census worker failed; partial evidence retained')
                or prior_attempt.get('worker_failure') != dict(type='BudgetExceeded',
                    reason='whole-work limit exceeded before operation')):
            raise ValueError('nested D047 failed census record differs')
        direct_attempt = prior['direct_prior_attempt']
        if (type(direct_attempt) is not dict or direct_attempt != done['direct_prior_attempt']
                or prior.get('direct_prior_failure_preserved') is not True
                or done.get('direct_prior_failure_preserved') is not True
                or direct_attempt.get('path') != str(FAILED_DIRECT)
                or direct_attempt.get('exit_sha256') != '6c1521af25dfb446e920774cfcbadd5e5ceb1aed4cad831a85bd46a4600d1b9c'
                or direct_attempt.get('tests_count') != 3797 or direct_attempt.get('test_files') != 176
                or direct_attempt.get('mathematical_component_gate_passed') is not True
                or direct_attempt.get('all_stages_passed') is not False
                or direct_attempt.get('source_census_completed') is not False
                or direct_attempt.get('source_census_qualified') is not False
                or direct_attempt.get('worker_exit') != 1 or direct_attempt.get('formal_gain') != 0
                or direct_attempt.get('failure') != dict(type='ValueError',
                    reason='source census worker failed; partial evidence retained')
                or direct_attempt.get('worker_failure') != dict(type='ValueError',
                    reason='evidence budget exhausted before operation')):
            raise ValueError('unchanged D057 failed census record differs')
        inherited_gate = dict(path=str(PRIOR), exit_sha256=sha(PRIOR / 'exit.json'),
            tests_count=3809, test_files=179, mathematical_component_gate_passed=True,
            relational_interface_fixture_exercised=done['relational_interface_fixture_exercised'],
            actual_model_binding_qualified=done['actual_model_binding_qualified'],
            all_stages_passed=True, worker_stage_registered=False, worker_launched=False,
            source_census_qualified=False, native_HZ_admitted=False,
            gpu_computation_completed=False, formal_gain=0)
        record.update(prior_attempt=prior_attempt, prior_census_failure_preserved=True,
            direct_prior_attempt=direct_attempt, direct_prior_failure_preserved=True,
            inherited_mathematical_gate=inherited_gate, inherited_native_predicate_gate=native_gate)
        for path, digest in prior['source_sha256'].items():
            helper.bind(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        # Authenticate the consumed D070 attempt without granting it success.
        candidate_done = read_json(FAILED_SAME / 'exit.json')
        if (candidate_done.get('failure', {}).get('type') != 'TimeoutExpired'
                or candidate_done.get('component_tests_passed') is not False
                or candidate_done.get('mathematical_component_gate_passed') is not False
                or candidate_done.get('joint_negative_fixture_exercised') is not False
                or candidate_done.get('all_stages_passed') is not False
                or candidate_done.get('supervisor_exit') != 1
                or candidate_done.get('source_drift') or candidate_done.get('input_drift')
                or candidate_done.get('provenance_drift') is not False
                or candidate_done.get('formal_gain') != 0
                or candidate_done.get('worker_launched') is not False
                or candidate_done.get('source_census_qualified') is not False
                or candidate_done.get('native_HZ_admitted') is not False
                or candidate_done.get('gpu_computation_completed') is not False
                or 'tests_exit' in candidate_done or 'tests.xml' in candidate_done['artifacts']):
            raise ValueError('D070 consumed timeout record differs')
        candidate_freeze = read_json(CANDIDATE / 'freeze.json', RESERVE)
        candidate_prior = read_json(FAILED_SAME / 'preregistered.json')
        candidate_inventory = read_json(FAILED_SAME / 'inventory.json')
        candidate_sources = candidate_freeze.get('source_sha256')
        if (candidate_freeze.get('schema') != 'd070_frozen_v1'
                or candidate_freeze.get('required_tests') != 3813
                or candidate_freeze.get('required_test_files') != 180
                or candidate_freeze.get('new_test_names') != list(TEST_NAMES)
                or type(candidate_sources) is not dict
                or set(candidate_sources) != {str(CANDIDATE / n) for n in CANDIDATE_FILES}
                or candidate_prior['required_tests'] != 3813
                or candidate_prior['required_test_files'] != 180
                or candidate_prior['input_sha256'] != prior['input_sha256']
                or candidate_prior['provenance'] != prior['provenance']
                or candidate_prior['selected_sources'] != prior['selected_sources']
                or candidate_prior['gpu_dependency_files'] != prior['gpu_dependency_files']
                or candidate_prior['decoder_dependency_files'] != prior['decoder_dependency_files']
                or candidate_prior['prior_attempt'] != prior_attempt
                or candidate_prior['direct_prior_attempt'] != direct_attempt
                or candidate_done['prior_attempt'] != prior_attempt
                or candidate_done['direct_prior_attempt'] != direct_attempt
                or candidate_prior['inherited_mathematical_gate'] != inherited_gate
                or candidate_done['inherited_mathematical_gate'] != inherited_gate
                or candidate_prior['inherited_native_predicate_gate'] != native_gate
                or candidate_done['inherited_native_predicate_gate'] != native_gate):
            raise ValueError('D070 frozen candidate/provenance contract differs')
        for path, digest in candidate_sources.items():
            if (candidate_prior['source_sha256'].get(path) != digest
                    or Path(path).is_symlink()):
                raise ValueError('D070 six-source manifest differs: ' + path)
            helper.bind(identities, path, digest)
        for path, digest in candidate_prior['source_sha256'].items():
            helper.bind(identities, path, digest)
        for name, digest in candidate_done['artifacts'].items():
            helper.bind(identities, helper.original_path(FAILED_SAME, name), digest)
        same_attempt = dict(path=str(FAILED_SAME),
            exit_sha256=sha(FAILED_SAME / 'exit.json'),
            freeze_sha256=sha(CANDIDATE / 'freeze.json'),
            manifest_sha256=sha(FAILED_SAME / 'preregistered.json'),
            inventory_sha256=sha(FAILED_SAME / 'inventory.json'),
            collected_tests=candidate_inventory['count'],
            collected_test_files=candidate_inventory['files'],
            component_tests_passed=False, mathematical_component_gate_passed=False,
            joint_negative_fixture_exercised=False, all_stages_passed=False,
            test_wall_s=candidate_done['test_wall_s'], supervisor_exit=1,
            tests_exit_recorded=False, junit_available=False,
            failure=candidate_done['failure'], formal_gain=0)
        record.update(same_candidate_prior_attempt=same_attempt,
            same_candidate_prior_failure_preserved=True)
        if any(path not in identities for path in prior['tests']):
            raise ValueError('inherited test file lacks frozen identity')
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('frozen interpreter differs')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('three original source identities differ')
        decoder = helper.bind_decoder(identities)
        if len(decoder) != 1011 or decoder != read_json(OLDER)['decoder_dependency_files']:
            raise ValueError('frozen decoder population differs')
        dependencies = gpu.gpu_dependencies(helper, identities)
        if len(dependencies) != 4417 or dependencies != prior['gpu_dependency_files']:
            raise ValueError('frozen GPU dependencies differ')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run source/input drift')
        test_path = CANDIDATE / 'test_joint_support.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 4 or len({node.name for node in functions}) != 4
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)
                or [node.name for node in functions] != list(TEST_NAMES)):
            raise ValueError('exact four frozen plain new tests required')
        relative = str(test_path.relative_to(ROOT))
        tests = [*prior['tests'], str(test_path)]
        expected = [*inventory['nodeids'], *(relative + '::' + n.name for n in functions)]
        if len(tests) != 180 or len(set(tests)) != 180 or len(expected) != 3813 or len(set(expected)) != 3813:
            raise ValueError('complete 3813/180 population differs')
        if (candidate_prior['tests'] != tests
                or candidate_prior['expected_nodeids'] != expected
                or candidate_inventory['nodeids'] != expected
                or candidate_inventory['count'] != 3813 or candidate_inventory['files'] != 180):
            raise ValueError('unchanged D070 ordered complete population differs')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            selected_sources=selected, decoder_dependency_files=decoder,
            gpu_dependency_files=dependencies, freeze_path=str(FREEZE), freeze_sha256=sha(FREEZE),
            required_tests=3813, required_test_files=180, inherited_tests=3809, inherited_test_files=179,
            new_test_names=list(TEST_NAMES), prior_attempt=prior_attempt, prior_census_failure_preserved=True,
            direct_prior_attempt=direct_attempt, direct_prior_failure_preserved=True,
            inherited_mathematical_gate=inherited_gate, inherited_native_predicate_gate=native_gate,
            same_candidate_prior_attempt=same_attempt, same_candidate_prior_failure_preserved=True,
            same_process_collection_gate=True, single_pytest_process=True,
            collection_plugin=PLUGIN, candidate_source_directory=str(CANDIDATE),
            cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=AS_CAP, tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            worker_stage_registered=False, scope=SCOPE, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, whole_work_cap=256_000_000, branch_work_cap=200_000_000,
            evidence_prepaid_work=40_000_000, retained_entry_cap=64_000_000, rational_bit_cap=512,
            caches_relocated_to_new_run=True, cuda_visible_devices='', complete_physical_qualification=False,
            candidate_physical_gate_evaluated=False, source_census_completed=False, source_census_qualified=False,
            joint_negative_fixture_registered=True, actual_model_binding_qualified=False,
            actual_phase_column_binding_verified=False, native_HZ_admitted=False,
            gpu_computation_completed=False, formal_gain=0))
        emit(dict(event='frozen_before_candidate_import', tests=3813, files=180, scope=SCOPE))
        manifest_sha = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D072_MANIFEST_SHA256'] = manifest_sha
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', '-p', PLUGIN,
                   '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            tested = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_started,
            tests_exit=tested.returncode, tests_count=3813)
        test_started = None
        checked = read_json(RUN / 'inventory.json')
        ids = checked.get('nodeids')
        if (checked.get('count') != 3813 or checked.get('files') != 180
                or ids != expected or len(set(ids)) != 3813
                or len({n.split('::', 1)[0] for n in ids}) != 180
                or checked.get('manifest_sha256') != manifest_sha
                or checked.get('validated_before_execution') is not True
                or sha(RUN / 'preregistered.json') != manifest_sha):
            raise ValueError('same-session pre-execution inventory contract differs')
        record['inventory_validated_before_execution'] = True
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if (tested.returncode or record['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))):
            raise ValueError('complete inherited and mathematical component gate failed')
        record['component_tests_passed'] = True
        emit(dict(event='complete_component_pass', tests=3813, files=180, wall_s=record['test_wall_s']))
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
        record['joint_negative_fixture_exercised'] = record['mathematical_component_gate_passed']
        record['all_stages_passed'] = record['mathematical_component_gate_passed']
        record['supervisor_exit'] = 0 if record['all_stages_passed'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
