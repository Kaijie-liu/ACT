"""Once-only original-parameter audit; no candidate or mathematical rerun.

Reuse D179's authenticated bounded-read diagnostic shape and D025's evidence
writer.  The D240 receipt authenticates history, not this new reader's math.
All hashing has the registered fixed per-64KiB bookkeeping charge and reports
actual bytes; that charge is not a claim about hash instruction complexity.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import tracemalloc
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d241_residual_source_binding_20261005_v1'
PRIOR = EXP / 'results/d240_phase_supported_component_20261005_v1'
BASE = HERE.parent / 'd240_phase_supported_component_20261005'
STRUCTURES = tuple(EXP / ('results/d179_preterminal_domain_20261004_v1/model_' + str(i) + '.json')
                   for i in range(3))
SCHEMA = 'd241_residual_source_binding_v1'
FILES = ('THEORY.md', 'PREREG.md', 'source_audit.py', 'run_audit.py')
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
COMMIT = 'f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac'
DIFF_SHA = '29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5'
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
WORK_CAP, MODEL_WORK_CAP = 256_000_000, 200_000_000
EVIDENCE_CAP, ENTRY_CAP, WALL_CAP = 40_000_000, 64_000_000, 240
MODEL_BYTES_CAP, JSON_BYTES_CAP = 64 * 1024**2, 8 * 1024**2
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
# Identical ordered production-byte definition used by the inherited D015
# provenance receipt; tracked git diff is an additional D179 custody check.
PRODUCTION = (
    'act/back_end/solver/neural_hz.py', 'act/back_end/solver/solver_hz.py',
    'act/back_end/hybridz_tf/hybridz_tf.py', 'act/back_end/hybridz_tf/tf_mlp.py',
    'act/back_end/hybridz_tf/tf_cnn.py', 'act/back_end/hybridz_tf/exact_linear_op.py',
    'act/back_end/verifier.py', 'act/config/config.py',
    'experiments/neural_hz_20260831/shadow_worker.py',
    'experiments/neural_hz_20260831/shadow_worker_dtype_v2.py',
    'act/pipeline/verification/torch2act.py',
    'act/pipeline/verification/batchnorm_graph.py',
    'act/front_end/vnnlib_loader/onnx_converter.py')
KERNEL = HERE.parent / 'd015_source_shielding_20260928/shield_kernel_v1.py'
READER = HERE.parent / 'd015_source_shielding_20260928/source_packet_v1.py'
HELPER = HERE.parent / 'd015_batch_binding_20260928_v2/census_worker_v2.py'
EVIDENCE = HERE.parent / 'd025_interval_capacity_20260930/evidence.py'
ANCHORS = {
    PRIOR / 'preregistered.json': '7a7c7663bfd2496a840bac459ea136acb76a49393c8211978dec1669a49106ff',
    PRIOR / 'inventory.json': 'f499a8db427c58f945cedf2e508aec204072f2c8406088a1d4efb79c129bdc68',
    PRIOR / 'exit.json': 'e0418f0b687ef18d1fb2d579011cc1517691e2393d766e97e45398ed9649f664',
    PRIOR / 'tests.xml': 'f541c676bc7d3d7dcc20d5f732b0f078f0a8aa2a7edc792c295eb1b0fd431654',
    PRIOR / 'summary.json': 'b7312330eb45265e4a7288e9c32233f8c334e0421a39c212d055cc0d18388cba',
    BASE / 'freeze.json': '332eead93b3e37aa07c65984059134512700a7db8b12a343955ef54fa8f718ec',
    BASE / 'ARCHIVE.sha256': '81fb9219fc5025fa00d1d28539712fc2317fd496e08da2909cc43268a3115f61',
}
PRIOR_TRUE = ('mathematical_stage_only', 'fixed_component_lp_controls_registered',
              'domain_definition_changed', 'new_component_solver_free')
PRIOR_FALSE = ('worker_stage_registered', 'worker_launched', 'source_component_qualified',
    'source_census_completed', 'source_census_qualified', 'actual_model_binding_qualified',
    'actual_phase_column_binding_verified', 'native_HZ_admitted', 'gpu_computation_completed',
    'complete_physical_qualification', 'candidate_physical_gate_evaluated',
    'production_snapshot_imported', 'solver_rescue_registered', 'negative_audit_only',
    'new_set_class', 'new_domain_qualified', 'new_capability_qualified',
    'capability_improvement_claimed')
FALSE_FLAGS = ('candidate_executed', 'd240_executed', 'model_forward_executed', 'solver_executed',
    'mathematical_tests_executed', 'mathematical_component_gate_passed',
    'source_component_qualified', 'actual_model_binding_qualified',
    'actual_phase_column_binding_verified', 'native_HZ_admitted',
    'gpu_computation_completed', 'complete_physical_qualification',
    'new_domain_qualified', 'new_capability_qualified', 'capability_improvement_claimed',
    'activation_bounds_propagated', 'd240_guard_evaluated', 'domain_definition_changed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Deadline(RuntimeError):
    pass


def stop(signum, frame):
    raise Deadline('registered 240-second deadline or termination: ' + str(signum))


class Meter:
    """One bootstrap ledger, transferred once into the authenticated kernel."""

    def __init__(self):
        self.started = time.monotonic()
        self.rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.bootstrap = EVIDENCE_CAP + RESERVE
        self.budget = None
        self.hash_files = self.hash_bytes = self.metadata_entries = 0
        self.peak_entries = self.model_work = 0
        self.evidence = None

    @property
    def used(self):
        return self.bootstrap if self.budget is None else self.budget.used

    def charge(self, amount, entries=0):
        require(type(amount) is int and amount >= 0 and type(entries) is int and entries >= 0,
                'invalid audit accounting charge')
        require(self.metadata_entries + entries <= ENTRY_CAP, 'audit metadata entry cap')
        if self.budget is None:
            require(amount <= WORK_CAP - self.bootstrap, 'bootstrap whole-work cap')
            self.bootstrap += amount
        else:
            self.budget.charge(amount)
        self.metadata_entries += entries

    def attach(self, kernel):
        require(self.budget is None, 'bootstrap work may be transferred only once')
        self.budget = kernel.WorkBudget(enabled=True, limit=WORK_CAP, max_bits=512)
        self.budget.charge(self.bootstrap)

    def snapshot(self):
        current, peak = tracemalloc.get_traced_memory()
        hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return dict(wall_s=time.monotonic() - self.started, whole_work_used=self.used,
            maximum_model_work_used=self.model_work, evidence_prepaid_work=EVIDENCE_CAP,
            evidence_work_used=self.evidence.used if self.evidence else 0,
            metadata_entries_upper=self.metadata_entries, peak_entries_upper=self.peak_entries,
            rss_initial_high_water_bytes=self.rss0, rss_high_water_bytes=hwm,
            rss_highwater_growth_bytes=max(0, hwm - self.rss0),
            traced_current_bytes=current, traced_peak_bytes=peak,
            tracer_metadata_bytes=tracemalloc.get_tracemalloc_memory(),
            summary_reserve_bytes=RESERVE, hashed_files=self.hash_files, hashed_bytes=self.hash_bytes,
            hash_bookkeeping_work_per_64KiB_block=16,
            complete_physical_qualification=False)

    def check(self):
        view = self.snapshot()
        require(view['rss_highwater_growth_bytes'] + RESERVE <= MEMORY_CAP
                and view['traced_peak_bytes'] + view['tracer_metadata_bytes'] + RESERVE <= MEMORY_CAP,
                'host memory observation cap')
        require(self.used <= WORK_CAP and self.model_work <= MODEL_WORK_CAP
                and self.metadata_entries <= ENTRY_CAP and self.peak_entries <= ENTRY_CAP,
                'audit work or numeric-entry cap')
        if view['wall_s'] >= WALL_CAP:
            raise Deadline('240-second complete audit deadline')
        return view


def checked(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            'ordinary absolute file required: ' + str(path))
    return path


def digest(path, meter):
    value = hashlib.sha256()
    with checked(path).open('rb') as stream:
        while True:
            meter.charge(16)
            chunk = stream.read(65536)
            if not chunk:
                break
            value.update(chunk)
            meter.hash_bytes += len(chunk)
            meter.check()
    meter.hash_files += 1
    return value.hexdigest()


def read_bytes(path, expected, cap, meter, *, json_payload=False):
    path = checked(path)
    size = path.stat().st_size
    require(0 < size <= cap, 'input size cap: ' + str(path))
    if json_payload:
        meter.charge(4096 + 5 * size, entries=size)
    # Raw-model/spec parsing is prepaid by extract, not charged a second time
    # here. These I/O/hash buffer visits retain their D179 bookkeeping charge.
    chunks, actual, value = [], 0, hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            meter.charge(16)
            chunk = stream.read(min(65536, cap + 1 - actual))
            if not chunk:
                break
            actual += len(chunk)
            require(actual <= cap, 'input grew beyond registered size cap')
            value.update(chunk)
            meter.hash_bytes += len(chunk)
            chunks.append(chunk)
            meter.check()
    meter.hash_files += 1
    require(actual == size and value.hexdigest() == expected, 'authenticated byte mismatch: ' + str(path))
    return b''.join(chunks)


def json_read(path, expected, meter):
    raw = read_bytes(path, expected, JSON_BYTES_CAP, meter, json_payload=True)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonstandard JSON number: ' + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def identities(value, count=None):
    require(type(value) is dict and (count is None or len(value) == count), 'identity population differs')
    for path, sha in value.items():
        require(type(path) is str and Path(path).is_absolute() and type(sha) is str
                and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'malformed identity')
    return value


def merge(mapping, path, sha):
    path = str(path)
    require(path not in mapping or mapping[path] == sha, 'conflicting historical identity: ' + path)
    mapping[path] = sha


def verify_all(mapping, meter):
    for path, sha in mapping.items():
        require(digest(path, meter) == sha, 'identity drift: ' + path)


def provenance(meter):
    def git(*args):
        meter.charge(16)
        meter.check()
        return subprocess.check_output(['git', *args], cwd=ROOT, timeout=3)

    value = dict(branch=git('branch', '--show-current').decode().strip(),
        commit=git('rev-parse', 'HEAD').decode().strip(),
        tracked_diff_sha256=hashlib.sha256(git('diff', '--binary', 'HEAD', '--')).hexdigest())
    require(value == dict(branch='redu-hz', commit=COMMIT, tracked_diff_sha256=DIFF_SHA),
            'production provenance drift')
    candidate = hashlib.sha256()
    for name in PRODUCTION:
        candidate.update(name.encode())
        with checked(ROOT / name).open('rb') as stream:
            while True:
                meter.charge(16)
                chunk = stream.read(65536)
                if not chunk:
                    break
                candidate.update(chunk)
                meter.hash_bytes += len(chunk)
                meter.check()
        meter.hash_files += 1
    value['candidate_sha256'] = candidate.hexdigest()
    return value


def load_checked(path, name, sources, meter):
    require(str(path) in sources and digest(path, meter) == sources[str(path)],
            'unauthenticated audit dependency: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prior_receipt(meter):
    prior = json_read(PRIOR / 'preregistered.json', ANCHORS[PRIOR / 'preregistered.json'], meter)
    done = json_read(PRIOR / 'exit.json', ANCHORS[PRIOR / 'exit.json'], meter)
    inventory = json_read(PRIOR / 'inventory.json', ANCHORS[PRIOR / 'inventory.json'], meter)
    expected, tests = prior.get('expected_nodeids'), prior.get('tests')
    require(prior.get('schema') == 'd240_phase_supported_component_v1'
        and prior.get('required_tests') == 4169 and prior.get('required_test_files') == 222
        and type(expected) is list and len(expected) == len(set(expected)) == 4169
        and type(tests) is list and len(tests) == len(set(tests)) == 222
        and prior.get('cpu_affinity') == [0]
        and all(prior.get(key) is True for key in PRIOR_TRUE)
        and all(prior.get(key) is False for key in PRIOR_FALSE), 'D240 frozen mathematical registration differs')
    require(all(done.get(key) is True for key in ('component_tests_passed',
        'mathematical_component_gate_passed', 'all_registered_stages_passed',
        'inventory_validated_before_execution', 'host_observations_within_caps'))
        and all(done.get(key) is False for key in PRIOR_FALSE)
        and done.get('tests_exit') == done.get('supervisor_exit') == 0
        and done.get('tests_count') == 4169 and done.get('test_files') == 222
        and 0 <= done.get('test_wall_s', 61) <= 60 and 'failure' not in done
        and done.get('source_drift') == done.get('input_drift') == []
        and done.get('provenance_drift') is False
        and all(done.get(key) == 0 for key in ('formal_gain', 'independent_e0_gain', 'new_benchmark_solves')),
        'D240 successful complete mathematical receipt differs')
    require(inventory.get('count') == 4169 and inventory.get('files') == 222
        and inventory.get('nodeids') == expected and inventory.get('validated_before_execution') is True
        and inventory.get('manifest_sha256') == ANCHORS[PRIOR / 'preregistered.json'],
        'D240 complete ordered inventory differs')
    xml = read_bytes(PRIOR / 'tests.xml', ANCHORS[PRIOR / 'tests.xml'], JSON_BYTES_CAP, meter)
    meter.charge(4 * len(xml), entries=len(xml))
    cases = ET.fromstring(xml).findall('.//testcase')
    actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '') for case in cases]
    require(actual == expected and not any(case.find(key) is not None for case in cases
            for key in ('failure', 'error', 'skipped')), 'D240 JUnit population or verdict differs')
    sources = dict(identities(prior.get('source_sha256'), 7655))
    inputs = dict(identities(prior.get('input_sha256'), 14))
    require(type(done.get('artifacts')) is dict and len(done['artifacts']) == 29,
            'D240 complete artifact population differs')
    for name, sha in done['artifacts'].items():
        path = PRIOR / name
        require(path.resolve() == path and path.is_relative_to(PRIOR), 'historical artifact escapes RUN')
        merge(sources, path, sha)
    for path, sha in ANCHORS.items():
        merge(sources, path, sha)
    selected = prior.get('selected_sources')
    require(type(selected) is list and len(selected) == 3
        and len({source['model_path'] for source in selected}) == 3, 'fixed three-source population differs')
    for source, structure in zip(selected, STRUCTURES):
        require(inputs.get(source['model_path']) == source['model_sha256']
            and inputs.get(source['spec_path']) == source['spec_sha256']
            and sources.get(source['manifest_path']) == source['manifest_sha256']
            and str(structure) in sources, 'original source/spec/structure identity missing')
    return prior, sources, inputs, selected


def validate_record(record, source, structure, meter):
    meter.charge(256)
    require(type(record) is dict and record.get('schema') == 'd241_complete_residual_source_v1'
            and record.get('source') == source
            and all(record.get(key) is True for key in ('source_binding_complete',
                'raw_parameters_verified', 'through_third_relu', 'all_prefix_consumers_accounted')),
            'complete source evidence identity differs')
    summary = record.get('summary', {})
    require(summary.get('relu_banks') == 3 and summary.get('prefix_node_count') == summary.get('final_relu_index', -1) + 1
            and len(record.get('prefix_nodes', ())) == summary['prefix_node_count']
            and len(record.get('logical_phases', ())) == 3
            and summary.get('final_relu_output') == record['logical_phases'][-1]['output']
            and summary.get('selected_windows', False) is None
            and record.get('original_graph_node_count') == len(structure['nodes'])
            and summary.get('input_coordinates') == len(record.get('input_box', {}))
            and summary.get('decoded_parameter_scalars', 0) > 0, 'incomplete fixed residual prefix')
    upper = (8 * summary['decoded_parameter_scalars'] + 128 * summary['bn_channels']
             + 32 * summary['input_coordinates'] + 64 * record['original_graph_node_count'] + 65536)
    require(record.get('decoded_scalar_count') == summary['decoded_parameter_scalars']
        and record.get('prefix_node_count') == summary['prefix_node_count']
        and record.get('third_relu_port') == summary['final_relu_output']
        and record.get('temp_numeric_entries_upper') == summary.get('temp_numeric_entries_upper') == upper
        and summary.get('temp_numeric_entry_work_prepaid') == 8 * upper
        and summary.get('decoded_weight_arrays_released_before_return') is True
        and record.get('reconstruction', {}).get('decoded_values_retained_in_roots') is False,
        'declared decoder population, paid peak bound, or lifetime differs')
    require(all(summary.get(key) is False for key in ('activation_bounds_propagated',
        'actual_complete_child_affine_map_constructed', 'actual_native_phase_columns_bound',
        'native_HZ_admitted', 'actual_model_verification_qualified', 'd240_executed'))
        and all(summary.get(key) == 0 for key in ('model_forward_calls', 'solver_calls', 'formal_gain')),
        'source evidence claims an unregistered qualification')
    return summary


def save_small(name, value, reserve):
    """Only the globally prepaid small terminal reports; never large-root rescue."""
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(len(raw) <= reserve[0], 'shared final report reserve exhausted')
    reserve[0] -= len(raw)
    with (RUN / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
    return hashlib.sha256(raw).hexdigest()


def main():
    require(sys.argv[1:] == ['--enabled'], 'explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    meter = Meter()
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(WALL_CAP)
    status, model_start = 1, None
    sources, inputs, artifacts = {}, {}, {}
    freeze_sha = manifest_sha = None
    report = dict(schema=SCHEMA, diagnostic_complete=False, models=[], formal_gain=0,
        independent_e0_gain=0, new_benchmark_solves=0, source_precheck_complete=False,
        input_precheck_complete=False, source_postcheck_complete=False, input_postcheck_complete=False,
        prior_math_receipt_validated=False, historical_math_tests=4169, historical_math_files=222,
        historical_math_population_reduced=False)
    report.update({key: False for key in FALSE_FLAGS})
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        require(0 in os.sched_getaffinity(0), 'CPU 0 is unavailable')
        os.sched_setaffinity(0, {0})
        sys.dont_write_bytecode = True
        os.environ.update({name: '1' for name in THREADS})
        os.environ.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED='0',
            CUDA_CACHE_PATH=str(RUN / 'cuda_cache'), TORCH_HOME=str(RUN / 'torch_home'),
            XDG_CACHE_HOME=str(RUN / 'xdg_cache'), TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        (RUN / 'tmp').mkdir()
        tracemalloc.start()
        require(__debug__ and os.environ.get('PYTHONOPTIMIZE') in (None, '', '0'), 'assertions required')
        prior, sources, inputs, selected = prior_receipt(meter)
        freeze_sha = digest(HERE / 'freeze.json', meter)
        freeze = json_read(HERE / 'freeze.json', freeze_sha, meter)
        require(freeze.get('schema') == SCHEMA, 'D241 freeze schema')
        fresh = identities(freeze.get('source_sha256'), len(FILES))
        require(set(fresh) == {str(HERE / name) for name in FILES}, 'D241 frozen four-source population')
        for path, sha in fresh.items():
            merge(sources, path, sha)
        merge(sources, HERE / 'freeze.json', freeze_sha)
        verify_all(sources, meter)
        verify_all(inputs, meter)
        require(Path(sys.executable).resolve() == PYTHON.resolve()
                and str(PYTHON.resolve()) in sources, 'authenticated interpreter required')
        before = provenance(meter)
        require({key: before[key] for key in ('branch', 'commit', 'candidate_sha256')}
                == prior['provenance'], 'prior production provenance differs')
        report.update(prior_math_receipt_validated=True, source_precheck_complete=True,
            input_precheck_complete=True, source_count=len(sources), inherited_source_count=7655,
            input_count=len(inputs), freeze_sha256=freeze_sha, provenance_before=before,
            cpu_affinity=[0], cuda_visible_devices='', bytecode_disabled=True)
        sys.path.insert(0, str(ROOT))
        k = load_checked(KERNEL, '_d241_kernel', sources, meter)
        meter.attach(k)
        evidence = load_checked(EVIDENCE, '_d241_evidence', sources, meter)
        meter.evidence = evidence.Meter(limit=EVIDENCE_CAP)
        helper = load_checked(HELPER, '_d241_source_helpers', sources, meter)
        base = load_checked(READER, '_d241_source_reader', sources, meter)
        audit = load_checked(HERE / 'source_audit.py', '_d241_source_audit', sources, meter)
        onnx_spec = importlib.util.find_spec('onnx')
        require(onnx_spec is not None and onnx_spec.origin
                and str(Path(onnx_spec.origin).resolve()) in sources, 'ONNX provenance')
        import onnx
        require(Path(onnx.__file__).resolve() == Path(onnx_spec.origin).resolve(),
                'ONNX imported module identity differs')
        registration = dict(schema=SCHEMA, source_sha256=sources, input_sha256=inputs,
            selected_sources=selected, structures=[dict(path=str(path), sha256=sources[str(path)]) for path in STRUCTURES],
            provenance=before, freeze_sha256=freeze_sha, inherited_tests=4169, inherited_test_files=222,
            inherited_ordered_nodeids=prior['expected_nodeids'], inherited_test_paths=prior['tests'],
            prior_math_receipt=dict(path=str(PRIOR), manifest_sha256=ANCHORS[PRIOR / 'preregistered.json'],
                inventory_sha256=ANCHORS[PRIOR / 'inventory.json'], exit_sha256=ANCHORS[PRIOR / 'exit.json'],
                mathematical_component_gate_passed=True, qualification_transferred=False),
            cpu_affinity=[0], address_space_bytes=AS_CAP, audit_wall_cap_s=WALL_CAP,
            whole_work_cap=WORK_CAP, per_model_work_cap=MODEL_WORK_CAP, evidence_work_cap=EVIDENCE_CAP,
            retained_entry_cap=ENTRY_CAP, rational_endpoint_bits=512, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, cuda_visible_devices='', formal_gain=0,
            independent_e0_gain=0, new_benchmark_solves=0,
            scope='all three fixed original sources through their third ReLU, with all original consumers')
        registration.update({key: False for key in FALSE_FLAGS})
        written = evidence.write_evidence(RUN / 'preregistered.json', registration, meter.evidence, {})
        manifest_sha = artifacts['preregistered.json'] = written['sha256']
        del registration, prior
        for index, (source, structure_path) in enumerate(zip(selected, STRUCTURES)):
            meter.check()
            model_start = meter.budget.used
            meter.budget.limit = min(WORK_CAP, model_start + MODEL_WORK_CAP)
            item = dict(index=index, source=source, complete=False)
            report['models'].append(item)
            structure = json_read(structure_path, sources[str(structure_path)], meter)
            require(structure.get('structure_complete') is True and structure.get('metadata_complete') is True
                and structure.get('source', {}).get('model_path') == source['model_path']
                and structure.get('source', {}).get('model_sha256') == source['model_sha256'], 'D179 model identity differs')
            raw = read_bytes(source['model_path'], source['model_sha256'], MODEL_BYTES_CAP, meter)
            spec = read_bytes(source['spec_path'], source['spec_sha256'], JSON_BYTES_CAP, meter)
            roots, record = audit.extract(raw, spec, source, structure, meter.budget, k, helper, base)
            summary = validate_record(record, source, structure, meter)
            temporary_upper = summary.get('temp_numeric_entries_upper')
            require(type(temporary_upper) is int and 0 <= temporary_upper <= ENTRY_CAP,
                    'prepaid decoder numeric-entry upper bound missing')
            evidence_start = meter.evidence.used
            held = evidence.bounded_ledger((roots, report, sources, inputs), meter.evidence)
            entries = meter.metadata_entries + held['retained_entries'] + temporary_upper
            require(entries <= ENTRY_CAP, 'complete retained plus decoder numeric-entry upper bound exceeds cap')
            meter.peak_entries = max(meter.peak_entries, entries)
            name = 'source_' + str(index) + '.json'
            partial = RUN / (name + '.partial')
            written = evidence.write_evidence(partial, record, meter.evidence,
                {id(raw): source['model_sha256'], id(spec): source['spec_sha256']})
            require(not (RUN / name).exists(), 'completed source evidence already exists')
            partial.rename(RUN / name)
            artifacts[name] = written['sha256']
            item.update(complete=True, summary=summary, evidence_file=name,
                evidence_sha256=written['sha256'], evidence_bytes=written['bytes'],
                evidence_work=meter.evidence.used - evidence_start, retained_ledger=held,
                temporary_numeric_entries_upper=temporary_upper, combined_entries_upper=entries,
                model_work=meter.budget.used - model_start)
            meter.model_work = max(meter.model_work, item['model_work'])
            meter.budget.limit = WORK_CAP
            model_start = None
            del roots, record, structure, raw, spec, summary
            meter.check()
        verify_all(sources, meter)
        verify_all(inputs, meter)
        require(digest(RUN / 'preregistered.json', meter) == manifest_sha, 'new registration changed')
        report.update(source_postcheck_complete=True, input_postcheck_complete=True,
                      provenance_after=provenance(meter))
        require(report['provenance_after'] == report['provenance_before']
                and list(os.sched_getaffinity(0)) == [0], 'postcheck provenance or CPU drift')
        for name, sha in artifacts.items():
            require(digest(RUN / name, meter) == sha, 'new artifact identity differs: ' + name)
        meter.check()
        require(len(report['models']) == 3 and all(item['complete'] for item in report['models']),
                'incomplete three-model audit')
        report['diagnostic_complete'] = True
        status = 0
    except BaseException as error:
        report['failure'] = dict(type=type(error).__name__, reason=str(error)[:4096])
    finally:
        # A deadline/failure must still leave bounded terminal evidence. No old
        # identity retry, incomplete-root traversal, or candidate run occurs here.
        signal.alarm(0)
        if model_start is not None and meter.budget is not None:
            meter.model_work = max(meter.model_work, meter.budget.used - model_start)
        if not (report['source_postcheck_complete'] and report['input_postcheck_complete']):
            report['postcheck_reason'] = 'not completed after failure/deadline; audit remains unqualified'
        observations = meter.snapshot()
        within = (observations['wall_s'] < WALL_CAP
            and observations['rss_highwater_growth_bytes'] + RESERVE <= MEMORY_CAP
            and observations['traced_peak_bytes'] + observations['tracer_metadata_bytes'] + RESERVE <= MEMORY_CAP
            and meter.used <= WORK_CAP and meter.model_work <= MODEL_WORK_CAP
            and meter.metadata_entries <= ENTRY_CAP and meter.peak_entries <= ENTRY_CAP
            and (meter.evidence is None or meter.evidence.used <= EVIDENCE_CAP))
        if not within:
            report['diagnostic_complete'] = False
            report.setdefault('failure', dict(type='ResourceError', reason='final time/work/entries/host gate'))
            status = 1
        if 'failure' in report:
            report['diagnostic_complete'] = False
            status = 1
        report.update(host_observations=observations, host_observations_within_caps=within,
                      exit_status=status, artifact_sha256=dict(artifacts))
        reserve = [RESERVE]
        artifacts['report.json'] = save_small('report.json', report, reserve)
        final_wall = time.monotonic() - meter.started
        if final_wall >= WALL_CAP:
            status = 1
        terminal = dict(schema=SCHEMA, status=status, diagnostic_complete=status == 0,
            wall_s=final_wall, artifact_sha256=artifacts, formal_gain=0, independent_e0_gain=0,
            new_benchmark_solves=0, report_complete_before_final_write=report['diagnostic_complete'],
            report_reserve_remaining_bytes=reserve[0])
        terminal.update({key: False for key in FALSE_FLAGS})
        save_small('exit.json', terminal, reserve)
        print(json.dumps(dict(status=status, diagnostic_complete=status == 0,
            result_path=str(RUN / 'report.json'), failure=report.get('failure')), sort_keys=True), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
