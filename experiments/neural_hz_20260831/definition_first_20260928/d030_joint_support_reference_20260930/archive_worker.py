"""Single-use CPU arithmetic on a frozen complete archive, not a model replay.

Every archived window, receiver, canonical slot and coefficient is retained.
No old main/writer, model decode, solver or network forward is invoked.
"""
from fractions import Fraction
import hashlib
import json
import math
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
RUN = EXP / 'results/d030_joint_support_reference_20260930_v1'
ARCHIVE = EXP / 'results/d025_interval_capacity_20260930_v1/complete_0.json'
ARCHIVE_SHA = 'fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0'
ARCHIVE_BYTES = 12_062_002
CAP, MODEL_CAP, EVIDENCE_CAP = 256_000_000, 200_000_000, 40_000_000
MEMORY_CAP, RESERVE = 1024**3, 65536
POSITIONS = ((0, 0), (0, 31), (16, 16), (31, 0), (31, 31))
MODEL_SHA = '5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16'
SPEC_SHA = 'caca9ef2245019dd70883661bd5c102544f64152ddc0ea8f967eb8ed0882e996'
ZERO = Fraction(0)


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def fraction(value, budget):
    budget.charge(10)
    if (type(value) is not list or len(value) != 2
            or any(type(v) is not int for v in value)
            or value[1] <= 0
            or max(abs(v).bit_length() for v in value) > budget.max_bits):
        raise ValueError('invalid exact archive Fraction')
    return Fraction(value[0], value[1])


def interval(value, budget):
    budget.charge(5)
    if type(value) is not list or len(value) != 2:
        raise ValueError('invalid archive interval')
    lower, upper = fraction(value[0], budget), fraction(value[1], budget)
    if lower > upper:
        raise ValueError('reversed archive interval')
    return lower, upper


def scalar_add(left, right, budget):
    budget.charge(3)
    answer = left + right
    if max(abs(answer.numerator).bit_length(), answer.denominator.bit_length()) > budget.max_bits:
        raise ValueError('archive comparison arithmetic exceeds bit bound')
    return answer


def form(value, budget):
    budget.charge(6)
    if type(value) is not list or len(value) != 2 or type(value[1]) is not dict:
        raise ValueError('invalid archive affine form')
    constant = interval(value[0], budget)
    coefficients = {}
    budget.charge(5 * len(value[1]))
    for key, coefficient in value[1].items():
        if (type(key) is not str or not key.isdecimal() or len(key) > 7
                or str(int(key)) != key or not 0 <= int(key) < 3072):
            raise ValueError('invalid original input coordinate')
        coefficients[int(key)] = interval(coefficient, budget)
    return constant, coefficients


def read_archive(budget):
    if ARCHIVE.stat().st_size != ARCHIVE_BYTES:
        raise ValueError('frozen archive byte count differs')
    budget.charge(4096 + ARCHIVE_BYTES)
    raw = ARCHIVE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA:
        raise ValueError('frozen archive digest differs')

    def integer(text):
        budget.charge(3 + len(text))
        if len(text) > 160:
            raise ValueError('archive integer text exceeds bound')
        answer = int(text)
        if abs(answer).bit_length() > 512:
            raise ValueError('archive integer exceeds 512 bits')
        return answer

    def real(text):
        budget.charge(3 + len(text))
        answer = float(text)
        if not math.isfinite(answer):
            raise ValueError('nonfinite archive float')
        return answer

    def constant(text):
        raise ValueError('nonstandard archive number: ' + text)

    def object_pairs(pairs):
        budget.charge(5 + 5 * len(pairs))
        answer = {}
        for key, value in pairs:
            if key in answer:
                raise ValueError('duplicate archive JSON key')
            answer[key] = value
        return answer

    data = json.loads(raw, parse_int=integer, parse_float=real,
                      parse_constant=constant, object_pairs_hook=object_pairs)
    return raw, data


def decode(data, frozen, budget):
    budget.charge(1024)
    if type(data) is not dict or set(data) != {'box', 'model_raw', 'packet', 'result', 'source', 'spec_raw'}:
        raise ValueError('complete source archive schema differs')
    source, packet, result = data['source'], data['packet'], data['result']
    if (source != frozen['selected_sources'][0]
            or source['model_sha256'] != MODEL_SHA or source['spec_sha256'] != SPEC_SHA
            or data['model_raw']['sha256'] != MODEL_SHA
            or data['spec_raw']['sha256'] != SPEC_SHA
            or packet['raw_model_sha256'] != MODEL_SHA
            or packet['schema'] != 'd015_raw_first_bank_v1'
            or packet['input_name'] != 'modelInput'
            or result['frame_identity'] != [MODEL_SHA, 'modelInput']
            or result['first_shape'] != [1, 64, 32, 32]
            or packet['first_relu']['output'] != '127'):
        raise ValueError('original source/frame binding differs')
    if len(packet['branches']) != 1 or set(result['receivers']) != {'0'}:
        raise ValueError('archive branch population differs')
    conv = packet['branches'][0]['conv']
    if (conv['weight_shape'] != [64, 64, 3, 3] or conv['strides'] != [1, 1]
            or conv['pads'] != [1, 1, 1, 1] or conv['dilations'] != [1, 1]
            or conv['group'] != 1 or conv['input_shape'] != [1, 64, 32, 32]
            or conv['output_shape'] != [1, 64, 32, 32]):
        raise ValueError('frozen canonical convolution geometry differs')
    if len(data['box']) != 3072 or len(result['source_forms']) != 1600:
        raise ValueError('full box or saved affine population differs')
    budget.charge(8 * 3072 + 12 * 1600)
    box = {i: interval(data['box'][str(i)], budget) for i in range(3072)}
    forms, bounds, phases = {}, {}, {}
    for key, record in result['source_forms'].items():
        if type(key) is not str or len(key) > 40:
            raise ValueError('invalid source tuple key')
        match = re.fullmatch(r'\((\d+), (\d+), (\d+)\)', key)
        if match is None:
            raise ValueError('source key is not canonical tuple syntax')
        index = tuple(int(v) for v in match.groups())
        if not (0 <= index[0] < 64 and 0 <= index[1] < 32 and 0 <= index[2] < 32):
            raise ValueError('source tensor coordinate out of range')
        if index in forms or record['original_phase'] != ['127', *index]:
            raise ValueError('duplicate or incorrectly bound source phase')
        forms[index] = form(record['form'], budget)
        bounds[index] = interval(record['bounds'], budget)
        phases[index] = tuple(record['original_phase'])
    receivers = result['receivers']['0']
    if len(receivers) != 64:
        raise ValueError('all 64 receiver channels required')
    budget.charge(64 * (32 + 8 * 576))
    decoded_receivers = []
    for channel, receiver in enumerate(receivers):
        if receiver['channel'] != channel or len(receiver['weights']) != 576:
            raise ValueError('full receiver coefficient population differs')
        decoded_receivers.append((tuple(interval(v, budget) for v in receiver['weights']),
                                  interval(receiver['bias'], budget)))
    windows = result['windows']
    if len(windows) != 5 or result['summary']['rows'] != 320:
        raise ValueError('full archive window/row population differs')
    seen_sources, decoded_windows = set(), []
    zero_form = ((ZERO, ZERO), {})
    for position, window in zip(POSITIONS, windows):
        budget.charge(128 + 32 * 576 + 16 * 64)
        if (window['branch'] != 0 or window['position'] != list(position)
                or len(window['source_slots']) != 576
                or len(window['original_phases']) != 576
                or len(window['source_bounds']) != 576 or len(window['rows']) != 64):
            raise ValueError('canonical window shape differs')
        row, col = position
        slot_forms = []
        for offset in range(576):
            channel, location = divmod(offset, 9)
            dy, dx = divmod(location, 3)
            y, x = row + dy - 1, col + dx - 1
            index = (channel, y, x) if 0 <= y < 32 and 0 <= x < 32 else None
            expected_slot = list(index) if index is not None else None
            if window['source_slots'][offset] != expected_slot:
                raise ValueError('canonical padding/source identity differs')
            expected_phase = list(phases[index]) if index is not None else None
            if window['original_phases'][offset] != expected_phase:
                raise ValueError('canonical original phase differs')
            if interval(window['source_bounds'][offset], budget) != (
                    bounds[index] if index is not None else (ZERO, ZERO)):
                raise ValueError('saved source premise differs')
            slot_forms.append(forms[index] if index is not None else zero_form)
            if index is not None:
                seen_sources.add(index)
        old_rows = []
        for channel, old in enumerate(window['rows']):
            if (old['channel'] != channel or old['receiver_coefficients_ref'] != [0, channel]
                    or old['all_slot_and_pair_premises_ref'] != [0, row, col]):
                raise ValueError('saved receiver premise binding differs')
            ordinary = interval(old['ordinary_bounds'], budget)
            totals = old['capacity_coefficient_sums']
            positive = scalar_add(fraction(old['bias_positive'], budget),
                                  fraction(totals['positive'], budget), budget)
            negative = scalar_add(fraction(old['bias_negative'], budget),
                                  fraction(totals['negative'], budget), budget)
            budget.charge(6)
            if positive < ZERO or negative < ZERO:
                raise ValueError('negative old phase-capacity total')
            capacity_projection = -negative, positive
            prior = max(ordinary[0], -negative), min(ordinary[1], positive)
            if prior[0] > prior[1]:
                raise ValueError('old scalar premises conflict')
            old_rows.append((ordinary, capacity_projection, prior))
        decoded_windows.append((position, tuple(slot_forms), tuple(old_rows)))
    if seen_sources != set(forms):
        raise ValueError('not all saved source forms are consumed')
    return box, forms, bounds, phases, tuple(decoded_receivers), tuple(decoded_windows)


def main():
    if sys.argv[1:] != ['--enabled']:
        raise RuntimeError('archive component requires explicit --enabled')
    start = time.monotonic()
    initial = None
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            initial = int(line.split()[1]) * 1024
            break
    if initial is None:
        raise RuntimeError('initial RSS missing')
    budget = meter = None
    numerical_start, entries = None, 0
    report = dict(archive_completed=False, memory_gate_passed=False, expected_rows=320,
                  completed_rows=0, archive_sha256=ARCHIVE_SHA, formal_gain=0,
                  diagnostic_solver_calls=0, model_forward_calls=0, new_benchmark_solves=0,
                  native_HZ_admitted=False, gpu_computation_completed=False,
                  complete_physical_qualification=False, source_census_qualified=False)
    tracemalloc.start()
    try:
        if (len(os.sched_getaffinity(0)) != 1
                or resource.getrlimit(resource.RLIMIT_AS) != (16 * 1024**3,) * 2
                or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not __debug__):
            raise RuntimeError('worker requires CPU1 AS16GiB assertions CUDA-disabled')
        from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as kernel
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence
        from experiments.neural_hz_20260831.definition_first_20260928.d030_joint_support_reference_20260930 import joint_support
        budget = kernel.WorkBudget(enabled=True)
        budget.charge(EVIDENCE_CAP + RESERVE)
        meter = evidence.Meter(limit=EVIDENCE_CAP)
        numerical_start = budget.used
        budget.limit = min(CAP, numerical_start + MODEL_CAP)
        preregistered_path = RUN / 'preregistered.json'
        budget.charge(4096 + preregistered_path.stat().st_size)
        frozen = json.loads(preregistered_path.read_text())
        raw, data = read_archive(budget)
        decoded = decode(data, frozen, budget)
        box, forms, bounds, phases, receivers, windows = decoded
        rows = []
        stats = dict(rows=0, paired_strict_rows=0, upper_vs_old_rows=0,
                     lower_vs_old_rows=0, old_outer_crossing_rows=0, newly_stable_rows=0)
        prepared = None
        for position, slot_forms, old_rows in windows:
            emit(dict(event='window_started', position=position, completed_rows=len(rows),
                      whole_work_used=budget.used))
            prepared = joint_support.prepare_source(slot_forms, box, enabled=True, budget=budget)
            for channel, (weights, bias) in enumerate(receivers):
                budget.charge(128)
                result = joint_support.compile_prepared(prepared, weights, (bias, {}),
                                                        enabled=True, budget=budget)
                ordinary, capacity_projection, prior = old_rows[channel]
                old_lower, old_upper = prior
                if (result['lower'] > result['upper']
                        or result['lower'] < result['separate_lower']
                        or result['upper'] > result['separate_upper']):
                    raise ValueError('joint support sound-order/dominance invariant failed')
                combined = max(old_lower, result['lower']), min(old_upper, result['upper'])
                if combined[0] > combined[1]:
                    raise ValueError('certified intervals conflict on nonempty source')
                crossing = old_lower < ZERO < old_upper
                stats['rows'] += 1
                stats['paired_strict_rows'] += int(result['lower'] > result['separate_lower']
                                                 or result['upper'] < result['separate_upper'])
                stats['upper_vs_old_rows'] += int(result['upper'] < old_upper)
                stats['lower_vs_old_rows'] += int(result['lower'] > old_lower)
                stats['old_outer_crossing_rows'] += int(crossing)
                stats['newly_stable_rows'] += int(crossing and (combined[0] >= ZERO or combined[1] <= ZERO))
                rows.append(dict(branch=0, position=position, channel=channel, result=result,
                                 old_ordinary_bounds=ordinary,
                                 old_capacity_scalar_projection=capacity_projection,
                                 prior_combined_bounds=prior, combined_bounds=combined,
                                 input_window_ref=(0, *position), receiver_coefficients_ref=(0, channel)))
                report['completed_rows'] = len(rows)
                if (channel + 1) % 16 == 0:
                    emit(dict(event='rows_completed', completed_rows=len(rows),
                              whole_work_used=budget.used))
        if len(rows) != 320:
            raise ValueError('complete archive observation population missing')
        payload = dict(archive_sha256=ARCHIVE_SHA, archive_path=str(ARCHIVE),
                       frame_identity=data['result']['frame_identity'], source=data['source'],
                       rule='d029_joint_support_with_explicit_interval_error', rows=rows,
                       summary=stats, scope='archive-only scalar support component; no native HZ admission',
                       original_bits_deleted=0, formal_gain=0)
        prepared_roots = prepared.evidence_roots()
        physical = evidence.bounded_ledger((raw, data, decoded, payload, prepared_roots, frozen), meter)
        # Include the immutable capsule header and bounded temporary row/index reserve.
        meter.charge(8)
        physical['held_instance_bytes'] += sys.getsizeof(prepared)
        physical['unique_objects'] += 1
        entries = physical['retained_entries'] + 64 * 576 + 4096
        if entries > 64_000_000:
            raise ValueError('retained numeric-entry limit exceeded')
        partial = RUN / 'complete.json.partial'
        written = evidence.write_evidence(partial, payload, meter, {})
        destination = RUN / 'complete.json'
        if destination.exists():
            raise ValueError('complete archive output already exists')
        partial.rename(destination)
        report.update(archive_completed=True, evidence_file='complete.json',
                      evidence_sha256=written['sha256'], evidence_bytes=written['bytes'],
                      held_ledger=physical, summary=stats)
    except Exception as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial)
        wall = time.monotonic() - start
        whole = budget.used if budget is not None else 0
        branch = whole - numerical_start if numerical_start is not None else 0
        report.update(wall_s=wall, whole_work_used=whole, branch_work_used=branch,
                      evidence_work_used=meter.used if meter is not None else 0,
                      retained_entries=entries, rss_highwater_growth_bytes=growth,
                      traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
                      initial_rss_bytes=initial, summary_reserve_bytes=RESERVE,
                      final_summary_reserve_bytes=RESERVE)
        report['memory_gate_passed'] = (report['archive_completed'] and 'failure' not in report
            and growth + RESERVE <= MEMORY_CAP and peak + metadata + RESERVE <= MEMORY_CAP
            and wall <= 240 and whole <= CAP and branch <= MODEL_CAP
            and entries <= 64_000_000 and report['evidence_work_used'] <= EVIDENCE_CAP)
        encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        if len(encoded) > RESERVE:
            raise ValueError('summary reserve exceeded')
        with (RUN / 'diagnostic.json').open('xb') as stream:
            stream.write(encoded)
        print(encoded.decode(), end='', flush=True)
        tracemalloc.stop()
    return 0 if report['archive_completed'] and report['memory_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
