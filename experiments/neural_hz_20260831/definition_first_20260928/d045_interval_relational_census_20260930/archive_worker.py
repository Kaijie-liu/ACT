"""Single-use complete archived paired-consumer census; never a model run.

Authenticate stdlib-only D038 readers before reusing them, then authenticate
all actual project imports.  Old mains/writers/authenticate functions are never
called.  Continuous/native phase bindings and GPU qualification remain absent.
"""
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d045_interval_relational_census_20260930_v1'
PRIOR = EXP / 'results/d044_relational_generator_20260930_v1'
OLD = HERE.parent / 'd038_descendant_census_20260930/archive_worker.py'
OLD_SHA = '8ba8cbc88845c48724524f39ea9379c386526441b26aa30a5352e88388b49b8f'
PRIOR_MANIFEST_SHA = 'e7ade555ca971a3226657d5079519057b7b64bfea5327143b2ed182007ab8b85'
ARCHIVE = EXP / 'results/d025_interval_capacity_20260930_v1/complete_0.json'
ARCHIVE_SHA = 'fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0'
ARCHIVE_BYTES = 12_062_002
CAP, MODEL_CAP, EVIDENCE_CAP = 256_000_000, 200_000_000, 40_000_000
MEMORY_CAP, RESERVE, ENTRY_CAP = 1024**3, 65_536, 64_000_000
POSITIONS = ((0, 0), (0, 31), (16, 16), (31, 0), (31, 31))
NEW_FILES = ('PREREG.md', 'THEORY.md', 'interval_relation.py', 'test_interval_relation.py',
             'archive_worker.py', 'run_reference.py')
STATUSES = ('OK', 'PADDING', 'SEED_CONTRADICTION')
STATES = ('P', 'A', 'I', 'Z+', 'Z-', 'Z0', 'C')
BASE_FIELDS = ('baseline_difference', 'baseline_relu_difference', 'baseline_w', 'baseline_companion')
UNCOND_FIELDS = ('unconditional_difference', 'unconditional_relu_difference',
                 'unconditional_w', 'unconditional_companion')
COND_FIELDS = ('conditional_difference', 'relu_difference', 'conditional_w', 'relu_companion')
FORMULA = 'interval_relation.compile_pair:d045_interval_relational_v1'


def bootstrap_reader():
    """Fixed prepayment authenticates the helper before its first execution."""
    if OLD.is_symlink() or not OLD.is_file():
        raise ValueError('ordinary frozen D038 helper required')
    size = OLD.stat().st_size
    if not 0 < size <= 8_000_000:
        raise ValueError('old helper size outside bootstrap boundary')
    # Reserve before reading, hashing, loader rereading, and module construction.
    prepaid = 16_384 + 2 * size
    if prepaid > min(MODEL_CAP, CAP - EVIDENCE_CAP - RESERVE):
        raise ValueError('bootstrap exceeds unchanged work boundary')
    raw = OLD.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != OLD_SHA:
        raise ValueError('frozen D038 helper identity differs')
    spec = importlib.util.spec_from_file_location('d045_d038_readonly_readers', OLD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    budget = module.PreflightBudget()
    budget.charge(prepaid)
    return module, budget, raw


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def authenticate(old, budget):
    prior_raw, _ = old.read_bytes(PRIOR / 'preregistered.json', budget,
                                  expected=PRIOR_MANIFEST_SHA)
    prior = old.parse_json(prior_raw, budget)
    manifest_raw, _ = old.read_bytes(RUN / 'preregistered.json', budget)
    frozen = old.parse_json(manifest_raw, budget)
    freeze_raw, freeze_sha = old.read_bytes(HERE / 'freeze.json', budget, maximum=RESERVE)
    freeze = old.parse_json(freeze_raw, budget)
    if any(type(value) is not dict for value in (prior, frozen, freeze)):
        raise ValueError('manifest objects required')
    identities = old.digest_map(frozen.get('source_sha256'), budget)
    inputs = old.digest_map(frozen.get('input_sha256'), budget)
    prior_sources = old.digest_map(prior.get('source_sha256'), budget)
    prior_inputs = old.digest_map(prior.get('input_sha256'), budget)
    budget.charge(4 * (len(prior_sources) + len(prior_inputs)))
    if (any(identities.get(path) != digest for path, digest in prior_sources.items())
            or any(inputs.get(path) != digest for path, digest in prior_inputs.items())
            or prior_sources.get(str(OLD)) != OLD_SHA or identities.get(str(OLD)) != OLD_SHA
            or identities.get(str(PRIOR / 'preregistered.json')) != PRIOR_MANIFEST_SHA
            or identities.get(str(HERE / 'freeze.json')) != freeze_sha
            or frozen.get('freeze_path') != str(HERE / 'freeze.json')
            or frozen.get('freeze_sha256') != freeze_sha
            or frozen.get('selected_sources') != prior.get('selected_sources')
            or type(frozen.get('selected_sources')) is not list or len(frozen['selected_sources']) != 3
            or frozen.get('provenance') != prior.get('provenance')
            or frozen.get('gpu_dependency_files') != prior.get('gpu_dependency_files')
            or frozen.get('decoder_dependency_files') != prior.get('decoder_dependency_files')
            or inputs.get(str(ARCHIVE)) != ARCHIVE_SHA
            or frozen.get('archive_path') != str(ARCHIVE)
            or frozen.get('archive_sha256') != ARCHIVE_SHA
            or frozen.get('archive_bytes') != ARCHIVE_BYTES
            or frozen.get('positions') != [list(p) for p in POSITIONS]):
        raise ValueError('complete inherited source/input/dependency/freeze binding differs')
    caps = dict(address_space_bytes=16 * 1024**3, tests_combined_wall_cap_s=60,
        worker_wall_cap_s=240, whole_work_cap=CAP, branch_work_cap=MODEL_CAP,
        host_memory_cap_bytes=MEMORY_CAP, retained_entry_cap=ENTRY_CAP,
        rational_bit_cap=512, summary_reserve_bytes=RESERVE, evidence_prepaid_work=EVIDENCE_CAP,
        expected_groups=160, expected_records=46080, source_pairs_per_window=288,
        consumer_pairs_per_window=32)
    budget.charge(8 * len(caps))
    if any(type(frozen.get(key)) is not int or frozen[key] != value for key, value in caps.items()):
        raise ValueError('frozen resource or complete population limits differ')
    sources = old.digest_map(freeze.get('source_sha256'), budget)
    if (freeze.get('schema') != 'd045_frozen_v1'
            or set(sources) != {str(HERE / name) for name in NEW_FILES}
            or freeze.get('required_tests') != 3773 or freeze.get('required_test_files') != 171
            or frozen.get('required_tests') != 3773 or frozen.get('required_test_files') != 171
            or prior.get('required_tests') != 3769 or prior.get('required_test_files') != 170):
        raise ValueError('six-file/four-new-test freeze differs')
    names, nodeids, oldids = freeze.get('new_test_names'), frozen.get('expected_nodeids'), prior.get('expected_nodeids')
    if (type(names) is not list or len(names) != 4
            or any(type(name) is not str or not re.fullmatch(r'test_[A-Za-z0-9_]+', name) for name in names)
            or len(set(names)) != 4 or type(nodeids) is not list or len(nodeids) != 3773
            or type(oldids) is not list or len(oldids) != 3769
            or any(type(name) is not str or len(name) > 4096 for name in nodeids + oldids)):
        raise ValueError('complete inherited test identities required')
    budget.charge(12 * (len(nodeids) + len(oldids)))
    prefix = str((HERE / 'test_interval_relation.py').relative_to(ROOT)) + '::'
    if (len(set(nodeids)) != 3773 or len(set(oldids)) != 3769
            or set(nodeids) != set(oldids) | {prefix + name for name in names}):
        raise ValueError('old test omitted or new test population differs')
    for path, digest in sources.items():
        if identities.get(path) != digest:
            raise ValueError('new source not bound by supervisor')
        old.read_bytes(Path(path), budget, expected=digest)
    imports = (HERE / 'interval_relation.py',
        HERE.parent / 'd025_interval_capacity_20260930/evidence.py',
        HERE.parent / 'd025_interval_capacity_20260930/interval_capacity.py',
        HERE.parent / 'd015_source_shielding_20260928/shield_kernel_v1.py')
    checked = {str(OLD): OLD_SHA}
    for path in imports:
        package = path.parent
        while package != ROOT:
            budget.charge(64)
            initializer = package / '__init__.py'
            if initializer.exists() and str(initializer) not in checked:
                digest = identities.get(str(initializer))
                if digest is None or prior_sources.get(str(initializer)) != digest:
                    raise ValueError('package initializer lacks inherited authentication')
                old.read_bytes(initializer, budget, expected=digest)
                checked[str(initializer)] = digest
            package = package.parent
        authority = sources if path.parent == HERE else prior_sources
        digest = identities.get(str(path))
        if digest is None or authority.get(str(path)) != digest:
            raise ValueError('actual project import lacks authentication')
        old.read_bytes(path, budget, expected=digest)
        checked[str(path)] = digest
    return frozen, (prior_raw, prior, manifest_raw, frozen, freeze_raw, freeze, checked)


def pair_premises(data, windows, old, budget):
    """Read all 1440 fixed pair occurrences, including canonical padding."""
    result = []
    budget.charge(256 + 32 * 1440)
    for raw_window, window in zip(data['result']['windows'], windows):
        keys, values = raw_window.get('pair_keys'), raw_window.get('pair_bounds')
        if type(keys) is not list or type(values) is not list or len(keys) != 288 or len(values) != 288:
            raise ValueError('complete fixed pair premises absent')
        bounds = []
        for index, (key, value) in enumerate(zip(keys, values)):
            offset = 2 * index
            expected = raw_window['source_slots'][offset:offset + 2]
            if key != expected:
                raise ValueError('fixed pair orientation/source identity differs')
            bounds.append(old.interval(value, budget))
        if len(window['slot_bounds']) != 576:
            raise ValueError('complete source slots absent')
        result.append(tuple(bounds))
    return tuple(result)


def state(bound, real):
    lo, hi = bound
    if not real:
        return 'P'
    if lo > 0:
        return 'A'
    if hi < 0:
        return 'I'
    if lo == hi == 0:
        return 'Z0'
    if lo == 0:
        return 'Z+'
    if hi == 0:
        return 'Z-'
    return 'C'


def checked_interval(value, budget):
    budget.charge(12)
    if (type(value) is not tuple or len(value) != 2
            or any(type(v) is not Fraction or max(abs(v.numerator).bit_length(),
                       v.denominator.bit_length()) > 512 for v in value)
            or value[0] > value[1]):
        raise ValueError('kernel interval invalid or outside bit cap')
    return value


def empty_counts():
    return dict(records=0, status_counts={key: 0 for key in STATUSES},
        anchor_state_counts={key: 0 for key in STATES},
        tightening_independent=[0, 0, 0, 0], tightening_paired=[0, 0, 0, 0],
        potential_independent_records=0, potential_paired_records=0)


def check_result(result, window, channel, budget):
    budget.charge(1024 + 192 * 288)
    if (type(result) is not dict or result.get('baseline_h') != window['old_rows'][channel]
            or result.get('baseline_w') != window['old_rows'][channel + 1]
            or type(result.get('pairs')) is not tuple or len(result['pairs']) != 288
            or result.get('canonical_slots') != 576 or result.get('pair_count') != 288
            or result.get('valid_slots') != sum(window['real_slots'])
            or result.get('padding_slots') != 576 - sum(window['real_slots'])):
        raise ValueError('kernel population or original receiver ordinary bound differs')
    baseline = {key: checked_interval(result[key], budget)
                for key in ('baseline_h', *BASE_FIELDS)}
    compact, summary = [], empty_counts()
    for index, record in enumerate(result['pairs']):
        if type(record) is not dict or record.get('index') != (2 * index, 2 * index + 1):
            raise ValueError('complete canonical pair index differs')
        anchor_state = state(window['slot_bounds'][2 * index], window['real_slots'][2 * index])
        status = record.get('status')
        if (status not in STATUSES or record.get('anchor_state') != anchor_state
                or (status == 'PADDING') != (anchor_state == 'P')):
            raise ValueError('original-anchor state or padding classification differs')
        independent, paired = [], []
        for base_name, plain_name, conditional_name in zip(BASE_FIELDS, UNCOND_FIELDS, COND_FIELDS):
            plain = checked_interval(record[plain_name], budget)
            conditioned = record.get(conditional_name)
            if status != 'OK':
                if conditioned is not None:
                    raise ValueError('failed/padding record must not claim a conditional band')
                independent.append(0)
                paired.append(0)
                continue
            if type(conditioned) is not tuple or len(conditioned) != 2:
                raise ValueError('both original-anchor states must remain represented')
            comparisons = [checked_interval(band, budget) for band in conditioned]
            baseline_band = baseline[base_name]
            independent.append(sum(int(lo > baseline_band[0]) + int(hi < baseline_band[1])
                                   for lo, hi in comparisons))
            paired.append(sum(int(lo > plain[0]) + int(hi < plain[1]) for lo, hi in comparisons))
        independent, paired = tuple(independent), tuple(paired)
        if (record.get('tightening_independent') != independent
                or record.get('tightening_paired') != paired):
            raise ValueError('kernel endpoint-tightening counts disagree with retained bands')
        summary['records'] += 1
        summary['status_counts'][status] += 1
        summary['anchor_state_counts'][anchor_state] += 1
        summary['potential_independent_records'] += int(any(independent))
        summary['potential_paired_records'] += int(any(paired))
        for category in range(4):
            summary['tightening_independent'][category] += independent[category]
            summary['tightening_paired'][category] += paired[category]
        compact.append((status, anchor_state, independent, paired))
    reported = result.get('status_counts')
    if (type(reported) is not dict or any(key not in STATUSES for key in reported)
            or any(reported.get(key, 0) != summary['status_counts'][key] for key in STATUSES)):
        raise ValueError('kernel status counters disagree with the full population')
    return baseline, tuple(compact), summary


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required; no default execution')
    started, initial = time.monotonic(), None
    bootstrap = budget = meter = None
    bootstrap_cost, entries, numerical_start = 0, None, EVIDENCE_CAP + RESERVE
    report = dict(archive_completed=False, memory_gate_passed=False,
        expected_groups=160, completed_groups=0, expected_records=46080, completed_records=0,
        archive_sha256=ARCHIVE_SHA, formal_gain=0, diagnostic_solver_calls=0,
        model_forward_calls=0, new_benchmark_solves=0, native_HZ_admitted=False,
        gpu_computation_completed=False, complete_physical_qualification=False,
        source_census_completed=False, source_census_qualified=False,
        binding_mathematical_only=True, binding_math_verified=False,
        actual_phase_column_binding_verified=False, original_bits_deleted=0)
    sys.dont_write_bytecode = True
    tracemalloc.start()
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                initial = int(line.split()[1]) * 1024
                break
        if (initial is None or len(os.sched_getaffinity(0)) != 1
                or resource.getrlimit(resource.RLIMIT_AS) != (16 * 1024**3,) * 2
                or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not __debug__):
            raise RuntimeError('CPU1 AS16GiB assertions CUDA-disabled and RSS telemetry required')
        old, bootstrap, helper_raw = bootstrap_reader()
        frozen, authentication_roots = authenticate(old, bootstrap)
        sys.path.insert(0, str(ROOT))
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence
        from experiments.neural_hz_20260831.definition_first_20260928.d045_interval_relational_census_20260930 import interval_relation
        budget = interval_relation.WorkBudget(enabled=True)
        bootstrap_cost = bootstrap.used
        budget.charge(numerical_start + bootstrap_cost)
        budget.limit = min(CAP, numerical_start + MODEL_CAP)
        meter = evidence.Meter(limit=EVIDENCE_CAP)
        raw, _ = old.read_bytes(ARCHIVE, budget, maximum=ARCHIVE_BYTES, expected=ARCHIVE_SHA)
        if len(raw) != ARCHIVE_BYTES:
            raise ValueError('full archived input byte count differs')
        data = old.parse_json(raw, budget)
        decoded = old.decode(data, frozen, budget)
        box, forms, bounds, phases, receivers, windows = decoded
        pairs = pair_premises(data, windows, old, budget)
        budget.charge(1024 + 24 * len(bounds))
        source_counts = {key: 0 for key in STATES}
        for bound in bounds.values():
            source_counts[state(bound, True)] += 1
        rows, stats, result = [], empty_counts(), None
        for window_index, window in enumerate(windows):
            emit(dict(event='window_started', position=window['position'],
                completed_groups=len(rows), completed_records=stats['records'], whole_work_used=budget.used))
            for channel in range(0, 64, 2):
                budget.charge(2048)
                left_weights, left_bias = receivers[channel]
                right_weights, right_bias = receivers[channel + 1]
                # Release the previous full result BEFORE constructing the next;
                # retained compact records and baselines remain physical roots.
                result = None
                result = interval_relation.compile_pair(left_weights, left_bias,
                    right_weights, right_bias, window['slot_bounds'], pairs[window_index],
                    window['real_slots'], enabled=True, budget=budget)
                baseline, compact, counts = check_result(result, window, channel, budget)
                for key in ('records', 'potential_independent_records', 'potential_paired_records'):
                    stats[key] += counts[key]
                for key in ('status_counts', 'anchor_state_counts'):
                    for category in stats[key]:
                        stats[key][category] += counts[key][category]
                for key in ('tightening_independent', 'tightening_paired'):
                    for category in range(4):
                        stats[key][category] += counts[key][category]
                rows.append(dict(branch=0, position=window['position'], channels=(channel, channel + 1),
                    source_window_ref=(0, *window['position']),
                    receiver_coefficients_ref=((0, channel), (0, channel + 1)),
                    ordinary_bounds_match=True, baselines=baseline, pair_records=compact,
                    bands_ref=FORMULA, summary=counts))
                report['completed_groups'], report['completed_records'] = len(rows), stats['records']
        if (len(rows) != 160 or stats['records'] != 46080
                or sum(stats['status_counts'].values()) != 46080
                or sum(stats['anchor_state_counts'].values()) != 46080
                or sum(source_counts.values()) != 1600):
            raise ValueError('complete fixed population not established')
        payload = dict(schema='d045_interval_relational_census_v1', archive_sha256=ARCHIVE_SHA,
            archive_path=str(ARCHIVE), kernel_sha256=frozen['source_sha256'][str(HERE / 'interval_relation.py')],
            frame_identity=data['result']['frame_identity'], source=data['source'],
            formula_ref=FORMULA, expected_groups=160, completed_groups=160,
            expected_records=46080, completed_records=46080, groups=rows, summary=stats,
            source_state_counts=source_counts,
            reconstruction=dict(record_fields=('status', 'anchor_state', 'tightening_independent', 'tightening_paired'),
                category_order=('difference', 'relu_difference', 'w', 'companion'),
                pair_index_rule='record j uses canonical slots (2*j,2*j+1), j=0..287',
                phase_ref='archive result.windows[position].original_phases[2*j]; None means no anchor',
                bands='not serialized: reconstruct full pairs[j] by frozen compile_pair on the bound archive window and receiver channels',
                comparison='POTENTIAL endpoint improvements, not nonredundancy or a property result'),
            binding_mathematical_only=True, binding_math_verified=True,
            actual_phase_column_binding_verified=False, original_bits_deleted=0, formal_gain=0)
        physical = evidence.bounded_ledger((helper_raw, raw, data, decoded, pairs, rows, payload,
            result, baseline, compact, counts, source_counts, authentication_roots,
            bootstrap.__dict__, budget.__dict__, meter.__dict__, report), meter)
        transient_reserve = 256 * 576 + 8192
        entries = physical['retained_entries'] + transient_reserve
        if entries > ENTRY_CAP:
            raise ValueError('retained plus conservative transient entry cap exceeded')
        partial, destination = RUN / 'complete.json.partial', RUN / 'complete.json'
        if destination.exists():
            raise ValueError('complete evidence already exists')
        written = evidence.write_evidence(partial, payload, meter, {})
        partial.rename(destination)
        report.update(archive_completed=True, binding_math_verified=True,
            evidence_file='complete.json', evidence_sha256=written['sha256'], evidence_bytes=written['bytes'],
            held_ledger=physical, transient_entry_reserve=transient_reserve,
            summary=stats, source_state_counts=source_counts)
    except Exception as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = (max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial)
                  if initial is not None else None)
        if bootstrap is not None:
            bootstrap_cost = bootstrap.used
        whole = budget.used if budget is not None else numerical_start + bootstrap_cost
        wall = time.monotonic() - started
        report.update(wall_s=wall, whole_work_used=whole, branch_work_used=whole - numerical_start,
            bootstrap_work_used=bootstrap_cost, evidence_work_used=meter.used if meter is not None else 0,
            retained_entries=entries, retained_entries_measured=entries is not None,
            rss_highwater_growth_bytes=growth, traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
            initial_rss_bytes=initial, summary_reserve_bytes=RESERVE, final_summary_reserve_bytes=RESERVE)
        report['memory_gate_passed'] = (report['archive_completed'] and 'failure' not in report
            and growth is not None and growth + RESERVE <= MEMORY_CAP
            and peak + metadata + RESERVE <= MEMORY_CAP and wall <= 240
            and whole <= CAP and whole - numerical_start <= MODEL_CAP
            and entries is not None and entries <= ENTRY_CAP and report['evidence_work_used'] <= EVIDENCE_CAP)
        encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        if len(encoded) > RESERVE:
            raise ValueError('final diagnostic exceeds reserved boundary')
        with (RUN / 'diagnostic.json').open('xb') as stream:
            stream.write(encoded)
        print(encoded.decode(), end='', flush=True)
        tracemalloc.stop()
    return 0 if report['archive_completed'] and report['memory_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
