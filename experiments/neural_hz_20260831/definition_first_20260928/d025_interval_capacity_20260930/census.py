"""Single-use source-derived interval capacity census, never a verifier."""
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d025_interval_capacity_20260930_v1'
sys.path.insert(0, str(ROOT))
CAP, MODEL_CAP, EVIDENCE_WORK = 256_000_000, 200_000_000, 40_000_000
MEMORY_CAP, RESERVE = 1024**3, 65536
ZERO = F(0)


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def first_form(packet, key, post, helper, k, budget):
    """Original NCHW source; padding occurs after input preprocessing."""
    channel, row, col = key
    conv, shape, pre = packet['first_conv'], packet['input_shape'], packet['pre_affine']
    fanin = len(conv['weights']) // conv['weight_shape'][0]
    budget.charge(32 + 12 * fanin)
    constant, terms = k.point(conv['bias'][channel], budget), {}
    for ci, iy, ix, offset in helper.receptive(conv, shape, row, col):
        weight = k.point(conv['weights'][channel * fanin + offset], budget)
        coefficient = k.mul(weight, k.point(pre['scale'][ci], budget), budget)
        constant = k.add(constant, k.mul(weight, k.point(pre['bias'][ci], budget), budget), budget)
        source_id = (ci * shape[2] + iy) * shape[3] + ix
        terms[source_id] = k.add(terms.get(source_id, (ZERO, ZERO)), coefficient, budget)
    alpha, shift = post[channel]
    scaled = k.affine_scale((constant, terms), alpha, budget)
    return k.affine_add(scaled, (shift, {}), budget)


def exact_sum(values, k, budget):
    answer = ZERO
    budget.charge(len(values))
    for value in values:
        budget.charge(1)
        answer += value
        k._checked((answer, answer), budget)
    return answer


def receiver_interval(weights, bias, activation_bounds, k, budget):
    """Box bound using nonnegative activation intervals; two products per slot.

    Coefficients and source bounds have already been checked by the capacity
    lemma. Since r>=0, the minimum uses w_lo and maximum uses w_hi; their
    signs decide which r endpoint is used. No midpoint or dependence claim.
    """
    budget.charge(4 + 10 * len(weights))
    lower, upper = bias
    for (wl, wu), (rl, ru) in zip(weights, activation_bounds):
        budget.charge(4)
        left, right = wl * (ru if wl < 0 else rl), wu * (rl if wu < 0 else ru)
        k._checked((left, left), budget)
        k._checked((right, right), budget)
        budget.charge(2)
        lower, upper = lower + left, upper + right
        k._checked((lower, upper), budget)
    return lower, upper


def source_census(packet, box, helper, k, compile_capacities, budget):
    first_shape = helper.conv_shape(packet['first_conv'], packet['input_shape'])
    budget.charge(64 + 8 * first_shape[1] + 32 * len(packet['branches']))
    post = tuple(helper.post_affine(packet['first_post_ops'], c, k, budget)
                 for c in range(first_shape[1]))
    cache, pair_cache, windows, receivers = {}, {}, [], {}
    stats = dict(rows=0, branches=len(packet['branches']), admitted_branches=0,
        windows=0, canonical_slots=0, valid_slots=0, padding_slots=0,
        guaranteed_matches=0, positive_coefficients_reduced=0, negative_coefficients_reduced=0,
        rows_reduced=0, outer_crossing_rows=0, outer_crossing_rows_reduced=0,
        receiver_added_nnz=0, raw_consumers_unchanged=True, original_bits_deleted=0)
    expected_rows, max_slots = 0, 0
    for branch_index, branch in enumerate(packet['branches']):
        if branch['target_relu'] is None:
            continue
        stats['admitted_branches'] += 1
        conv = branch['conv']
        next_shape = helper.conv_shape(conv, first_shape)
        co = next_shape[1]
        fanin = len(conv['weights']) // co
        if fanin != conv['weight_shape'][1] * conv['weight_shape'][2] * conv['weight_shape'][3]:
            raise ValueError('complete Conv coefficient population differs')
        if not 1 <= fanin <= 65536:
            raise ValueError('unsupported canonical Conv slot count')
        max_slots = max(max_slots, fanin)
        positions = helper.anchors(next_shape[2], next_shape[3])
        expected_rows += co * len(positions)
        budget.charge(64 + co * (24 + 8 * fanin))
        channel_receivers = []
        for oc in range(co):
            alpha, shift = helper.post_affine(branch['post_ops'], oc, k, budget)
            bias = k.add(k.mul(alpha, k.point(conv['bias'][oc], budget), budget), shift, budget)
            weights = tuple(k.mul(alpha, k.point(w, budget), budget)
                            for w in conv['weights'][oc * fanin:(oc + 1) * fanin])
            channel_receivers.append(dict(channel=oc, alpha=alpha, shift=shift,
                                          bias=bias, weights=weights))
        receivers[branch_index] = channel_receivers
        for row, col in positions:
            # Slot ordering, geometry, cache lookups and records are prepaid.
            budget.charge(128 + 48 * fanin + 32 * co)
            emit(dict(event='window_started', branch=branch_index, position=(row, col),
                      slots=fanin, channels=co, whole_work_used=budget.used))
            slots = [None] * fanin
            for ci, iy, ix, offset in helper.receptive(conv, first_shape, row, col):
                key = (ci, iy, ix)
                if slots[offset] is not None:
                    raise ValueError('duplicate canonical source slot')
                slots[offset] = key
                if key not in cache:
                    form = first_form(packet, key, post, helper, k, budget)
                    bound = k.source_box_bounds(form, box, budget)
                    cache[key] = dict(form=form, bounds=bound,
                        original_phase=(packet['first_relu']['output'], *key))
            bounds = tuple((ZERO, ZERO) if key is None else cache[key]['bounds'] for key in slots)
            activation_bounds = tuple(k.nonnegative_part(bound, budget) for bound in bounds)
            phases = tuple(None if key is None else cache[key]['original_phase'] for key in slots)
            real_phases = [phase for phase in phases if phase is not None]
            if len(set(real_phases)) != len(real_phases):
                raise ValueError('repeated original phase inside canonical window')
            pair_keys, differences = [], []
            for i in range(0, fanin - 1, 2):
                pair_key = (slots[i], slots[i + 1])
                if pair_key not in pair_cache:
                    left = ((ZERO, ZERO), {}) if pair_key[0] is None else cache[pair_key[0]]['form']
                    right = ((ZERO, ZERO), {}) if pair_key[1] is None else cache[pair_key[1]]['form']
                    difference = k.affine_add(left, k.affine_scale(right, k.point(-1, budget), budget), budget)
                    pair_cache[pair_key] = k.source_box_bounds(difference, box, budget)
                pair_keys.append(pair_key)
                differences.append(pair_cache[pair_key])
            differences = tuple(differences)
            window = dict(branch=branch_index, position=(row, col), source_slots=tuple(slots),
                original_phases=phases, source_bounds=bounds, pair_keys=tuple(pair_keys),
                pair_bounds=differences, rows=[])
            windows.append(window)
            stats['windows'] += 1
            stats['canonical_slots'] += fanin
            stats['valid_slots'] += len(real_phases)
            stats['padding_slots'] += fanin - len(real_phases)
            for receiver in channel_receivers:
                budget.charge(96 + 24 * fanin)
                weights, bias = receiver['weights'], receiver['bias']
                caps = compile_capacities(weights, bias, bounds, differences, enabled=True, budget=budget)
                if any(len(caps[key]) != fanin for key in
                       ('positive', 'negative', 'unpaired_positive', 'unpaired_negative')):
                    raise ValueError('capacity coefficient population differs')
                ordinary = receiver_interval(weights, bias, activation_bounds, k, budget)
                crossing = ordinary[0] < ZERO < ordinary[1]
                positive_changes = sum(a < b for a, b in zip(caps['positive'], caps['unpaired_positive']))
                negative_changes = sum(a < b for a, b in zip(caps['negative'], caps['unpaired_negative']))
                if (any(a > b for a, b in zip(caps['positive'], caps['unpaired_positive']))
                        or any(a > b for a, b in zip(caps['negative'], caps['unpaired_negative']))):
                    raise ValueError('paired coefficient domination failed')
                totals = {key: exact_sum(caps[key], k, budget) for key in
                          ('positive', 'negative', 'unpaired_positive', 'unpaired_negative')}
                reduced = positive_changes + negative_changes > 0
                # This references every certified premise and the frozen uniform
                # formula. No zero or unhelpful row is omitted from the evidence.
                window['rows'].append(dict(channel=receiver['channel'],
                    capacity_rule='interval_capacity.compile_capacities',
                    receiver_coefficients_ref=(branch_index, receiver['channel']),
                    all_slot_and_pair_premises_ref=(branch_index, row, col),
                    ordinary_bounds=ordinary, outer_crossing=crossing,
                    positive_coefficients_reduced=positive_changes,
                    negative_coefficients_reduced=negative_changes,
                    guaranteed_matches=len(caps['matches']), added_nnz=caps['nnz'],
                    bias_positive=caps['bias_positive'], bias_negative=caps['bias_negative'],
                    capacity_coefficient_sums=totals))
                stats['rows'] += 1
                stats['guaranteed_matches'] += len(caps['matches'])
                stats['positive_coefficients_reduced'] += positive_changes
                stats['negative_coefficients_reduced'] += negative_changes
                stats['rows_reduced'] += int(reduced)
                stats['outer_crossing_rows'] += int(crossing)
                stats['outer_crossing_rows_reduced'] += int(crossing and reduced)
                stats['receiver_added_nnz'] += caps['nnz']
            emit(dict(event='window_completed', branch=branch_index, position=(row, col),
                      completed_rows=stats['rows'], rows_reduced=stats['rows_reduced'],
                      outer_crossing_rows=stats['outer_crossing_rows'],
                      outer_crossing_rows_reduced=stats['outer_crossing_rows_reduced'],
                      whole_work_used=budget.used))
    if not windows or stats['rows'] != expected_rows:
        raise ValueError('complete direct-branch receiver population not established')
    budget.charge(32 + 16 * len(cache))
    stats.update(expected_rows=expected_rows, source_forms=len(cache), certified_pairs=len(pair_cache),
        source_strict_active=sum(v['bounds'][0] > 0 for v in cache.values()),
        source_strict_inactive=sum(v['bounds'][1] < 0 for v in cache.values()),
        source_exact_zero=sum(v['bounds'] == (ZERO, ZERO) for v in cache.values()),
        source_outer_crossing=sum(v['bounds'][0] < 0 < v['bounds'][1] for v in cache.values()),
        actual_sign_reachability_not_measured=True,
        max_slots=max_slots, raw_side_consumers=len(packet['side_consumers']))
    return dict(frame_identity=(packet['raw_model_sha256'], packet['input_name']),
                first_shape=first_shape, first_post_affine=post, source_forms=cache,
                pair_bounds=pair_cache, receivers=receivers, windows=windows, summary=stats)


def one_model(source, index, k, helper, extract_model, compile_capacities, budget,
              completed, evidence_meter, evidence):
    model_start = budget.used
    budget.limit = min(CAP, model_start + MODEL_CAP)
    model_path, spec_path = Path(source['model_path']), Path(source['spec_path'])
    budget.charge(4096 + model_path.stat().st_size + 8 * spec_path.stat().st_size)
    raw, spec = model_path.read_bytes(), spec_path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != source['model_sha256']
            or hashlib.sha256(spec).hexdigest() != source['spec_sha256']):
        raise ValueError('raw source differs from frozen bytes')
    packet = extract_model(raw, enabled=True, input_batch=1)
    box = helper.input_box(spec, packet['input_shape'])
    emit(dict(event='source_parsed', model=source['model_relative_path'],
              input_shape=packet['input_shape'], whole_work_used=budget.used))
    result = source_census(packet, box, helper, k, compile_capacities, budget)
    emit(dict(event='source_numerical_completed', model=source['model_relative_path'],
              summary=result['summary'], whole_work_used=budget.used,
              qualification=False, evidence_serialization_not_completed=True))
    roots = dict(source=source, model_raw=raw, spec_raw=spec, packet=packet, box=box, result=result)
    evidence_start = evidence_meter.used
    held = evidence.bounded_ledger((roots, completed), evidence_meter)
    # This covers bounded transient row arrays/indices in addition to retained
    # evidence objects. Actual allocator and temporary peaks are measured below.
    entry_upper = held['retained_entries'] + 64 * result['summary']['max_slots'] + 4096
    if entry_upper > 64_000_000:
        raise ValueError('retained numeric-entry cap exceeded')
    name = 'complete_' + str(index) + '.json'
    partial = RUN / (name + '.partial')
    written = evidence.write_evidence(partial, roots, evidence_meter,
        {id(raw): source['model_sha256'], id(spec): source['spec_sha256']})
    if (RUN / name).exists():
        raise ValueError('complete evidence target already exists')
    partial.rename(RUN / name)
    answer = dict(model=source['model_relative_path'], evidence_file=name,
        evidence_sha256=written['sha256'], evidence_bytes=written['bytes'],
        evidence_work=evidence_meter.used - evidence_start, held_ledger=held, retained_entry_upper=entry_upper,
        numerical_model_work=budget.used - model_start, summary=result['summary'])
    emit(dict(event='source_completed', model=source['model_relative_path'],
              summary=result['summary'], whole_work_used=budget.used))
    return answer


def main():
    if sys.argv[1:] != ['--enabled']:
        raise RuntimeError('source census requires explicit --enabled')
    start = time.monotonic()
    initial = None
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            initial = int(line.split()[1]) * 1024
            break
    if initial is None:
        raise RuntimeError('initial RSS missing')
    budget, model_start, evidence_meter = None, None, None
    report = dict(source_census_completed=False, memory_gate_passed=False, formal_gain=0,
        diagnostic_solver_calls=0, model_forward_calls=0, new_benchmark_solves=0,
        native_HZ_admitted=False, gpu_computation_completed=False,
        complete_physical_qualification=False, models=[], evidence_work_used=0)
    branch_used, entries = 0, 0
    tracemalloc.start()
    try:
        from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2 import census_worker_v2 as helper
        if (len(os.sched_getaffinity(0)) != 1
                or resource.getrlimit(resource.RLIMIT_AS) != (16 * 1024**3,) * 2
                or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not __debug__):
            raise RuntimeError('worker requires CPU1 AS16GiB assertions CUDA-disabled')
        from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
        from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2.source_binding_v2 import extract_model
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930.interval_capacity import compile_capacities
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence
        budget = k.WorkBudget(enabled=True)
        budget.charge(EVIDENCE_WORK + RESERVE)
        evidence_meter = evidence.Meter(limit=EVIDENCE_WORK)
        frozen = json.loads((RUN / 'preregistered.json').read_text())
        sources = frozen['selected_sources']
        if len(sources) != 3 or len({v['model_sha256'] for v in sources}) != 3:
            raise ValueError('three original models required')
        for index, source in enumerate(sources):
            model_start = budget.used
            answer = one_model(source, index, k, helper, extract_model, compile_capacities,
                               budget, report['models'], evidence_meter, evidence)
            report['models'].append(answer)
            report['evidence_work_used'] = evidence_meter.used
            entries = max(entries, answer['retained_entry_upper'])
            branch_used = max(branch_used, budget.used - model_start)
            model_start = None
        report['source_census_completed'] = len(report['models']) == 3
    except Exception as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc))
        # Do not perform unbudgeted recovery serialization of an incomplete root.
    finally:
        if budget is not None and model_start is not None:
            branch_used = max(branch_used, budget.used - model_start)
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        growth = max(0, hwm - initial) if initial is not None else hwm
        wall = time.monotonic() - start
        report.update(wall_s=wall, initial_rss_bytes=initial, rss_highwater_growth_bytes=growth,
            evidence_work_used=evidence_meter.used if evidence_meter else 0,
            traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
            final_summary_reserve_bytes=RESERVE, whole_work_used=budget.used if budget else 0,
            branch_work_used=branch_used, retained_entries=entries,
            actual_cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
            scope='all fixed first-bank windows and channels of three original models; CPU source certificates only',
            evidence_scope='all rows with shared premises and deterministic formula; no native solver storage claim')
        report['memory_gate_passed'] = (report['source_census_completed']
            and growth + RESERVE <= MEMORY_CAP and peak + metadata + RESERVE <= MEMORY_CAP
            and entries <= 64_000_000 and wall <= 240
            and report['whole_work_used'] <= CAP and branch_used <= MODEL_CAP
            and report['evidence_work_used'] <= EVIDENCE_WORK and 'failure' not in report)
        data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        if len(data) > RESERVE:
            raise ValueError('final summary reserve exceeded')
        with (RUN / 'diagnostic.json').open('xb') as stream:
            stream.write(data)
        print(data.decode(), end='', flush=True)
        tracemalloc.stop()
    return 0 if report['source_census_completed'] and report['memory_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
