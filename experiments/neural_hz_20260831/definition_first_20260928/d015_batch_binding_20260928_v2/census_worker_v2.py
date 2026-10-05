"""Opt-in local source-certificate pilot. Not a verifier or HZ admission path."""
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
ROOT = HERE.parents[3]
RUN = EXP / 'results/d015_source_shielding_20260928_v2'
sys.path.insert(0, str(ROOT))
RESERVE = 65536


def sexprs(raw):
    text = re.sub(r';[^\n]*', '', raw.decode('utf-8'))
    stack, out = [], []
    for token in re.findall(r'\(|\)|[^\s()]+', text):
        if token == '(':
            stack.append([])
        elif token == ')':
            if not stack:
                raise ValueError('unbalanced spec')
            value = stack.pop()
            (stack[-1] if stack else out).append(value)
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError('unparenthesized spec')
    if stack:
        raise ValueError('unbalanced spec')
    return out


def number(value):
    if isinstance(value, str):
        answer = F(value)
    elif isinstance(value, list) and len(value) == 2 and value[0] == '-':
        answer = -number(value[1])
    elif isinstance(value, list) and len(value) == 3 and value[0] == '/':
        answer = number(value[1]) / number(value[2])
    else:
        raise ValueError('unsupported exact number')
    if max(answer.numerator.bit_length(), answer.denominator.bit_length()) > 512:
        raise ValueError('spec rational cap')
    return answer


def atoms(value):
    if isinstance(value, str):
        return [value]
    return [token for child in value for token in atoms(child)]


def input_box(raw, shape):
    if len(shape) != 4 or shape[0] != 1 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('expected one NCHW input')
    count = shape[1] * shape[2] * shape[3]
    declarations, bounds = {}, {}
    for expr in sexprs(raw):
        if len(expr) == 3 and expr[0] == 'declare-const' and expr[2] == 'Real':
            name = expr[1]
            if name in declarations or not re.fullmatch(r'[XY]_\d+', name):
                raise ValueError('duplicate/unsupported declaration')
            declarations[name] = True
        elif len(expr) == 2 and expr[0] == 'assert':
            claim = expr[1]
            mentions = [a for a in atoms(claim) if a.startswith('X_')]
            if not mentions:
                # Output-only clauses are retained as raw bytes but are not queried.
                if any(a.startswith('Y_') and a not in declarations for a in atoms(claim)):
                    raise ValueError('undeclared output symbol')
                continue
            if (not isinstance(claim, list) or len(claim) != 3 or claim[0] not in ('<=', '>=')
                    or not isinstance(claim[1], str) or not re.fullmatch(r'X_\d+', claim[1])
                    or mentions != [claim[1]] or claim[1] not in declarations):
                raise ValueError('unsupported non-box input assertion')
            idx = int(claim[1][2:])
            if idx >= count:
                raise ValueError('input index outside tensor')
            side = 1 if claim[0] == '<=' else 0
            key = (idx, side)
            if key in bounds:
                raise ValueError('duplicate input endpoint')
            bounds[key] = number(claim[2])
        else:
            raise ValueError('unsupported spec command')
    if {n for n in declarations if n.startswith('X_')} != {'X_' + str(i) for i in range(count)}:
        raise ValueError('input declaration population differs')
    if len(bounds) != 2 * count:
        raise ValueError('missing input endpoint')
    result = {i: (bounds[i, 0], bounds[i, 1]) for i in range(count)}
    if any(lo > hi for lo, hi in result.values()):
        raise ValueError('inverted input box')
    return result


def conv_shape(conv, incoming):
    n, ci, height, width = incoming
    co, wc, kh, kw = conv['weight_shape']
    if conv['group'] != 1 or ci != wc or n != 1:
        raise ValueError('unsupported grouped or mismatched conv')
    sy, sx = conv['strides']; dy, dx = conv['dilations']
    top, left, bottom, right = conv['pads']
    if min(sy, sx, dy, dx) <= 0 or min(top, left, bottom, right) < 0:
        raise ValueError('invalid convolution geometry')
    out = (1, co, (height + top + bottom - dy * (kh - 1) - 1) // sy + 1,
           (width + left + right - dx * (kw - 1) - 1) // sx + 1)
    if min(out) <= 0:
        raise ValueError('empty convolution output')
    return out


def anchors(height, width):
    return tuple(sorted({(0, 0), (0, width - 1), (height - 1, 0),
                         (height - 1, width - 1), (height // 2, width // 2)}))


def receptive(conv, incoming, row, col):
    _, ci, height, width = incoming
    _, _, kh, kw = conv['weight_shape']
    sy, sx = conv['strides']; dy, dx = conv['dilations']
    top, left, _, _ = conv['pads']
    result = []
    for channel in range(ci):
        for ky in range(kh):
            for kx in range(kw):
                iy, ix = row * sy - top + ky * dy, col * sx - left + kx * dx
                # ONNX padding is zero AFTER input channel affine preprocessing.
                if 0 <= iy < height and 0 <= ix < width:
                    result.append((channel, iy, ix, (channel * kh + ky) * kw + kx))
    return result


def post_affine(ops, channel, k, budget):
    scale, bias = k.point(F(1), budget), k.point(F(0), budget)
    for op in ops:
        if op['kind'] == 'affine':
            a, b = k.point(op['scale'][channel], budget), k.point(op['bias'][channel], budget)
        elif op['kind'] == 'batchnorm':
            budget.charge(4)  # Raw variance addition and positivity comparisons.
            variance = op['variance'][channel] + op['epsilon']
            if variance <= 0:
                raise ValueError('nonpositive BN variance plus epsilon')
            root = k.sqrt_interval(variance, budget)
            if root[0] <= 0:
                raise ValueError('BN sqrt enclosure crosses zero')
            a = k.dyadic_enclose(k.div(k.point(op['gamma'][channel], budget), root, budget), budget)
            b = k.add(k.point(op['beta'][channel], budget),
                      k.neg(k.mul(a, k.point(op['mean'][channel], budget), budget), budget), budget)
        else:
            raise ValueError('unrecognized affine source operation')
        scale, bias = k.mul(a, scale, budget), k.add(k.mul(a, bias, budget), b, budget)
    return scale, bias


def first_form(packet, channel, row, col, k, budget):
    conv, shape = packet['first_conv'], packet['input_shape']
    weights = conv['weights']; fanin = len(weights) // conv['weight_shape'][0]
    constant, terms = k.point(conv['bias'][channel], budget), {}
    budget.charge(16 + 8 * fanin)
    for ci, iy, ix, offset in receptive(conv, shape, row, col):
        w = k.point(weights[channel * fanin + offset], budget)
        pre = packet['pre_affine']
        coefficient = k.mul(w, k.point(pre['scale'][ci], budget), budget)
        constant = k.add(constant, k.mul(w, k.point(pre['bias'][ci], budget), budget), budget)
        idx = (ci * shape[2] + iy) * shape[3] + ix
        terms[idx] = k.add(terms.get(idx, k.point(F(0), budget)), coefficient, budget)
    scale, bias = post_affine(packet['first_post_ops'], channel, k, budget)
    form = k.affine_scale((constant, terms), scale, budget)
    return k.affine_add(form, (bias, {}), budget)


def census(packet, box, k, budget, progress=None):
    first_shape = conv_shape(packet['first_conv'], packet['input_shape'])
    cache, windows = {}, []
    for branch_index, branch in enumerate(packet['branches']):
        if branch['target_relu'] is None:
            continue  # Complete side-branch metadata remains in the original packet.
        conv = branch['conv']
        next_shape = conv_shape(conv, first_shape)
        co = next_shape[1]; weights = conv['weights']; fanin = len(weights) // co
        for row, col in anchors(next_shape[2], next_shape[3]):
            budget.charge(32 + 8 * fanin)
            patch = receptive(conv, first_shape, row, col)
            budget.charge(64 + 20 * len(patch) * co)
            if progress is not None:
                progress(dict(event='window_started', branch=branch_index,
                              position=(row, col), whole_work_used=budget.used))
            for ci, iy, ix, _ in patch:
                key = (ci, iy, ix)
                if key not in cache:
                    form = first_form(packet, ci, iy, ix, k, budget)
                    bounds = k.source_box_bounds(form, box, budget)
                    tau, oriented, cap = k.orient_error(form, box, budget)
                    cache[key] = dict(form=form, bounds=bounds, tau=tau, oriented=oriented,
                                      cap=cap[1], original_phase=(packet['first_relu']['output'], *key))
            keys = [tuple(term[:3]) for term in patch]
            rows = []
            for oc in range(co):
                alpha, shift = post_affine(branch['post_ops'], oc, k, budget)
                base = k.add(k.mul(alpha, k.point(conv['bias'][oc], budget), budget), shift, budget)
                baseline_form, full = (base, {}), base
                positive, negative, uncertain = [], [], []
                for pos, (ci, iy, ix, offset) in enumerate(patch):
                    datum = cache[(ci, iy, ix)]
                    w = k.mul(alpha, k.point(weights[oc * fanin + offset], budget), budget)
                    lo, hi = datum['bounds']
                    full = k.add(full, k.mul(w, (max(F(0), lo), max(F(0), hi)), budget), budget)
                    if datum['tau']:
                        baseline_form = k.affine_add(baseline_form,
                            k.affine_scale(datum['form'], w, budget), budget)
                    if datum['cap'] == 0 or w == (F(0), F(0)):
                        continue  # Value-only zero; no original phase is deleted.
                    if w[0] > 0:
                        positive.append((pos, w))
                    elif w[1] < 0:
                        negative.append((pos, w))
                    else:
                        uncertain.append(pos)
                baseline = k.source_box_bounds(baseline_form, box, budget)
                rows.append(dict(channel=oc, baseline_form=baseline_form,
                    baseline_bounds=baseline, full_bounds=full,
                    positive=positive, negative=negative, ambiguous_signs=uncertain,
                    stable_by_outer_box=(full[0] >= 0 or full[1] <= 0)))
            requested_pairs = set()
            for entry in rows:
                if entry['baseline_bounds'][1] <= 0 and not entry['ambiguous_signs']:
                    # Covers Cartesian generation, dedup/hash, and the later
                    # proof loop INCLUDING conflicting pairs that immediately skip.
                    visits = len(entry['positive']) * len(entry['negative'])
                    budget.charge(32 + 24 * visits + 8 * len(entry['negative']))
                    requested_pairs.update(tuple(sorted((pi, ni))) for pi, _ in entry['positive']
                                           for ni, _ in entry['negative'])
            pair_evidence = {}
            budget.charge(len(requested_pairs) * (len(requested_pairs).bit_length() + 16))
            for pi, ni in sorted(requested_pairs):
                summed = k.affine_add(cache[keys[pi]]['oriented'], cache[keys[ni]]['oriented'], budget)
                bound = k.source_box_bounds(summed, box, budget)
                pair_evidence[(pi, ni)] = bound
            for entry in rows:
                tests, selected = [], []
                if entry['ambiguous_signs']:
                    reason = 'unproved_coefficient_sign'
                elif entry['baseline_bounds'][1] > 0:
                    reason = 'baseline_upper_positive'
                else:
                    reason = 'certificate_evaluated'
                    for ni, _ in entry['negative']:
                        capsum = k.point(F(0), budget)
                        for pi, w in entry['positive']:
                            pair = tuple(sorted((pi, ni)))
                            if pair_evidence[pair][1] <= 0:
                                continue
                            capsum = k.add(capsum, k.mul(w, (F(0), cache[keys[pi]]['cap']), budget), budget)
                        proof_bound = k.add((entry['baseline_bounds'][1],) * 2, capsum, budget)[1]
                        tests.append((ni, proof_bound))
                        if proof_bound <= 0:
                            selected.append(ni)
                entry.update(reason=reason, negative_tests=tests, shielded_positions=selected)
                # Weights are recoverable from original packet/BN scale; no duplicate dense row kept.
                entry['positive_positions'] = [i for i, _ in entry.pop('positive')]
                entry['negative_positions'] = [i for i, _ in entry.pop('negative')]
            windows.append(dict(branch_index=branch_index, position=(row, col),
                source_keys=keys, source_offsets=[term[3] for term in patch],
                output_shape=next_shape, rows=rows, pair_bounds=pair_evidence))
            if progress is not None:
                progress(dict(event='window_completed', branch=branch_index,
                    position=(row, col), rows=len(rows), tested_pairs=len(pair_evidence),
                    shielded_terms=sum(len(r['shielded_positions']) for r in rows),
                    whole_work_used=budget.used))
    if not windows:
        raise ValueError('no admitted direct first-bank next-ReLU consumer')
    return dict(first_shape=first_shape, source_forms=cache, windows=windows,
        rows=sum(len(w['rows']) for w in windows),
        shielded_terms=sum(len(r['shielded_positions']) for w in windows for r in w['rows']),
        outer_unstable_rows_shielded=sum(bool(r['shielded_positions']) and not r['stable_by_outer_box']
                                   for w in windows for r in w['rows']),
        actual_sign_reachability_not_measured=True,
        certified_pairs=sum(b[1] <= 0 for w in windows for b in w['pair_bounds'].values()),
        raw_consumers_unchanged=True, original_bits_deleted=0, native_row_savings_not_measured=True)


def encoded(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is bytes:
        return dict(byte_count=len(value), sha256=hashlib.sha256(value).hexdigest())
    if type(value) in (tuple, list):
        return [encoded(v) for v in value]
    if type(value) is dict:
        return {str(key): encoded(v) for key, v in value.items()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError('unaccounted evidence type: ' + str(type(value)))


def ledger(root):
    seen, pending, total, entries = set(), [root], 0, 0
    while pending:
        value = pending.pop()
        if type(value) in (int, bool, float, F):
            entries += 1
        if id(value) in seen:
            continue
        seen.add(id(value)); total += sys.getsizeof(value)
        if type(value) is F:
            pending.extend((value.numerator, value.denominator))
        elif type(value) in (tuple, list):
            pending.extend(value)
        elif type(value) is dict:
            for key, item in value.items():
                pending.extend((key, item))
        elif type(value) is bytes:
            # Byte payload storage is fully paid; serialized evidence is not a numeric tensor.
            pass
        elif value is not None and type(value) not in (str, int, float, bool):
            raise TypeError('unaccounted held root')
    return dict(held_instance_bytes=total, retained_entries=entries, unique_objects=len(seen),
                bytes_in_storage_not_numeric_tensor_entries=True)


def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('RSS missing')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise RuntimeError('D015 census is default-off; explicit --enabled required')
    start, initial = time.monotonic(), rss()
    roots, budget, branch_used, branch_start = [], None, 0, None
    report = dict(source_census_completed=False, memory_gate_passed=False, formal_gain=0,
                  diagnostic_solver_calls=0, model_forward_calls=0, new_benchmark_solves=0)
    tracemalloc.start()
    try:
        if len(os.sched_getaffinity(0)) != 1 or resource.getrlimit(resource.RLIMIT_AS) != (16 * 1024**3,) * 2:
            raise RuntimeError('worker requires inherited CPU1/AS16GiB')
        from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
        from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2.source_binding_v2 import extract_model
        budget = k.WorkBudget(enabled=True)
        frozen = json.loads((RUN / 'preregistered.json').read_text())
        for source in frozen['selected_sources']:
            branch_start = budget.used
            budget.limit = min(256000000, branch_start + 200000000)
            raw, spec = Path(source['model_path']).read_bytes(), Path(source['spec_path']).read_bytes()
            roots.append(dict(source=source, model_raw=raw, spec_raw=spec))
            if hashlib.sha256(raw).hexdigest() != source['model_sha256'] or hashlib.sha256(spec).hexdigest() != source['spec_sha256']:
                raise ValueError('consumed original bytes differ from freeze')
            budget.charge(4096 + len(raw) + 8 * len(spec))
            packet = extract_model(raw, enabled=True, input_batch=1)
            roots[-1]['packet'] = packet
            box = input_box(spec, packet['input_shape'])
            roots[-1]['box'] = box
            print(json.dumps(dict(event='source_parsed', model=source['model_relative_path'],
                                  whole_work_used=budget.used)), flush=True)
            result = census(packet, box, k, budget,
                            progress=lambda event: print(json.dumps(event), flush=True))
            roots[-1]['result'] = result
            branch_used = max(branch_used, budget.used - branch_start)
            if branch_used > 200000000:
                raise ValueError('nested source work cap')
        budget.limit = 256000000
        branch_start = None
        if len(roots) != 3:
            raise ValueError('expected all three distinct original models')
        # Fixed evidence/ledger reservation, checked against complete measured roots below.
        budget.charge(40000000 + 65536)
        payload = encoded(roots)
        raw_evidence = (json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        with (RUN / 'complete_source_evidence.json').open('xb') as stream:
            stream.write(raw_evidence)
        held = ledger((roots, payload, raw_evidence))
        if 8 * (held['retained_entries'] + held['unique_objects']) + len(raw_evidence) > 40000000:
            raise ValueError('complete evidence exceeds prepaid work reservation')
        report.update(source_census_completed=True, evidence_sha256=hashlib.sha256(raw_evidence).hexdigest(),
            evidence_bytes=len(raw_evidence), held_ledger=held, retained_entries=held['retained_entries'],
            models=[dict(model=r['source']['model_relative_path'], rows=r['result']['rows'],
                         shielded_terms=r['result']['shielded_terms'],
                         outer_unstable_rows_shielded=r['result']['outer_unstable_rows_shielded'],
                         certified_pairs=r['result']['certified_pairs']) for r in roots])
    except Exception as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc))
        # Preserve parsed metadata/provisional evidence, never treat it as qualification.
        partial = encoded(roots)
        with (RUN / 'partial_source_evidence.json').open('x') as stream:
            json.dump(partial, stream, sort_keys=True, allow_nan=False)
    finally:
        if budget is not None and branch_start is not None:
            branch_used = max(branch_used, budget.used - branch_start)
        current, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial)
        wall = time.monotonic() - start
        report.update(wall_s=wall, initial_rss_bytes=initial, rss_highwater_growth_bytes=growth,
            traced_peak_bytes=peak, tracer_metadata_bytes=metadata, final_summary_reserve_bytes=RESERVE,
            whole_work_used=budget.used if budget is not None else 0, branch_work_used=branch_used,
            retained_entries=report.get('retained_entries', 0), actual_cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
            scope='three-model one-spec each fixed spatial pilot, not dataset prevalence or native admission')
        report['memory_gate_passed'] = (report['source_census_completed']
            and growth + RESERVE <= 1024**3 and peak + metadata + RESERVE <= 1024**3
            and report['retained_entries'] <= 64000000 and wall <= 240
            and report['whole_work_used'] <= 256000000 and branch_used <= 200000000)
        serialized = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        if len(serialized) > RESERVE:
            raise ValueError('summary exceeds reserved bytes')
        with (RUN / 'diagnostic.json').open('xb') as stream:
            stream.write(serialized)
        print(serialized.decode(), end='', flush=True)
        tracemalloc.stop()
    return 0 if report['source_census_completed'] and report['memory_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

