"""Single-use original-source mixed-query worker; no verifier or native HZ.

The supervisor authenticates its complete inherited dependency population.
This process binds its manifest and every actual project import before import,
then authenticates and parses original sources through the frozen D015 reader.
"""
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d120_mixed_consumer_source_20261002_v1'
CAP, MODEL_CAP, EVIDENCE_CAP = 256_000_000, 200_000_000, 40_000_000
MEMORY_CAP, RESERVE, ENTRY_CAP = 1024**3, 65536, 64_000_000
ZERO, ONE = F(0), F(1)
ARMS = ('original', 'independent', 'common')


class Bootstrap:
    def __init__(self):
        self.used = 0

    def charge(self, amount):
        if type(amount) is not int or amount < 0 or amount > MODEL_CAP - self.used:
            raise ValueError('pre-import work exceeded')
        self.used += amount


def emit(value, budget):
    # Fixed small progress records, including the complete scalar summary.
    budget.charge(8192)
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def read_bytes(path, budget, digest=None, maximum=8_000_000, prepaid=False):
    if path.is_symlink() or not path.is_file():
        raise ValueError('ordinary original file required')
    size = path.stat().st_size
    if not 0 <= size <= maximum:
        raise ValueError('file size outside authenticated bound')
    if not prepaid:
        budget.charge(4096 + size)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if len(raw) != size or (digest is not None and actual != digest):
        raise ValueError('original file identity mismatch: ' + str(path))
    return raw, actual


def metadata(raw, budget):
    def integer(text):
        budget.charge(3 + len(text))
        if len(text) > 160:
            raise ValueError('oversized metadata integer')
        value = int(text)
        if abs(value).bit_length() > 512:
            raise ValueError('oversized metadata integer')
        return value

    def floating(text):
        budget.charge(3 + len(text))
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('nonfinite metadata')
        return value

    def pairs(items):
        budget.charge(5 + 5 * len(items))
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError('duplicate metadata key')
            result[name] = value
        return result

    def invalid(text):
        raise ValueError('nonstandard JSON number')

    return json.loads(raw, parse_int=integer, parse_float=floating,
                      object_pairs_hook=pairs, parse_constant=invalid)


def authenticate(budget):
    manifest_path = RUN / 'preregistered.json'
    expected = os.environ.get('NEURAL_HZ_SOURCE_MANIFEST_SHA256')
    if (os.environ.get('NEURAL_HZ_SOURCE_MANIFEST') != str(manifest_path)
            or type(expected) is not str or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError('supervisor-bound source manifest required')
    raw, digest = read_bytes(manifest_path, budget, expected)
    frozen = metadata(raw, budget)
    caps = dict(worker_wall_cap_s=240, whole_work_cap=CAP, branch_work_cap=MODEL_CAP,
        evidence_prepaid_work=EVIDENCE_CAP, retained_entry_cap=ENTRY_CAP,
        rational_bit_cap=512, summary_reserve_bytes=RESERVE,
        host_memory_cap_bytes=MEMORY_CAP, address_space_bytes=16 * 1024**3)
    if any(type(frozen.get(key)) is not int or frozen[key] != value for key, value in caps.items()):
        raise ValueError('frozen source limits changed')
    identities, inputs = frozen.get('source_sha256'), frozen.get('input_sha256')
    selected = frozen.get('selected_sources')
    if (type(identities) is not dict or type(inputs) is not dict
            or type(selected) is not list or len(selected) != 3
            or len(identities) > 50_000 or len(inputs) > 50_000):
        raise ValueError('complete inherited source/input records required')
    imports = (
        HERE / 'census.py', HERE / 'source_query.py',
        HERE.parent / 'd119_curvature_component_20261002/curvature_transfer.py',
        HERE.parent / 'd112_shared_endpoint_forward_20261002/endpoint_forward.py',
        HERE.parent / 'd015_source_shielding_20260928/shield_kernel_v1.py',
        HERE.parent / 'd015_source_shielding_20260928/source_packet_v1.py',
        HERE.parent / 'd015_batch_binding_20260928_v2/source_binding_v2.py',
        HERE.parent / 'd015_batch_binding_20260928_v2/census_worker_v2.py',
        HERE.parent / 'd025_interval_capacity_20260930/census.py',
        HERE.parent / 'd025_interval_capacity_20260930/evidence.py',
    )
    checked = {}
    budget.charge(64 + 16 * len(imports) + 16 * len(selected))
    for path in imports:
        package = path.parent
        while package != ROOT:
            budget.charge(8)
            init = package / '__init__.py'
            if init.exists() and str(init) not in checked:
                authority = identities.get(str(init))
                if type(authority) is not str:
                    raise ValueError('package initializer lacks binding')
                read_bytes(init, budget, authority)
                checked[str(init)] = authority
            package = package.parent
        authority = identities.get(str(path))
        if type(authority) is not str:
            raise ValueError('actual project import lacks frozen identity')
        read_bytes(path, budget, authority)
        checked[str(path)] = authority
    for source in selected:
        if type(source) is not dict:
            raise ValueError('invalid selected source')
        for name in ('model', 'spec'):
            if inputs.get(source[name + '_path']) != source[name + '_sha256']:
                raise ValueError('original source not bound to inherited inputs')
    return frozen, digest, (raw, frozen, checked)


def source_census(packet, box, helper, k, first_form, query, budget):
    first_shape = helper.conv_shape(packet['first_conv'], packet['input_shape'])
    if first_shape[1] != 64:
        raise ValueError('registered complete first channel population differs')
    budget.charge(128 + 32 * len(packet['branches']) + 16 * first_shape[1])
    post = tuple(helper.post_affine(packet['first_post_ops'], c, k, budget)
                 for c in range(first_shape[1]))
    sources, groups, windows, branches = {}, {}, [], []
    summary = dict(rows=0, expected_rows=0, completed_windows=0, expected_windows=0,
        canonical_slots=0, valid_slots=0, padding_slots=0, max_slots=0,
        source_forms=0, residual_groups=0, original_bits_deleted=0,
        preactivation_lower_improved=0, preactivation_upper_improved=0,
        relu_lower_improved=0, relu_upper_improved=0, no_gain_rows=0,
        outer_crossing_reference_rows=0, actual_sign_reachability_not_measured=True,
        raw_side_consumers=len(packet['side_consumers']))
    last_templates, last_query, last_forms = [], None, []
    for branch_index, branch in enumerate(packet['branches']):
        conv = branch['conv']
        shape = helper.conv_shape(conv, first_shape)
        co, ci, kh, kw = conv['weight_shape']
        fanin, spatial = ci * kh * kw, kh * kw
        budget.charge(64 + 8 * fanin)
        admitted = branch['target_relu'] is not None
        branches.append(dict(index=branch_index, admitted=admitted,
            conv_node=conv['node'], input_aliases=branch['input_aliases'],
            output_shape=shape, weight_shape=conv['weight_shape'],
            strides=conv['strides'], pads=conv['pads'], dilations=conv['dilations'],
            group=conv['group'], target_relu=branch['target_relu'], stop=branch['stop']))
        if not admitted:
            continue
        if (ci != 64 or (kh, kw) != (3, 3) or fanin != 576 or conv['group'] != 1
                or len(conv['weights']) != co * fanin or len(conv['bias']) != co):
            raise ValueError('complete registered direct Conv geometry differs')
        positions = helper.anchors(shape[2], shape[3])
        if len(positions) != 5:
            raise ValueError('complete five-position population differs')
        summary['expected_rows'] += co * len(positions)
        summary['expected_windows'] += len(positions)
        summary['max_slots'] = max(summary['max_slots'], fanin)
        branch_windows = []
        for row, col in positions:
            budget.charge(128 + 48 * fanin)
            slots = [None] * fanin
            for channel, iy, ix, offset in helper.receptive(conv, first_shape, row, col):
                if slots[offset] is not None:
                    raise ValueError('duplicate canonical source slot')
                slots[offset] = (channel, iy, ix)
            group_keys = []
            for offset in range(spatial):
                for begin in range(0, ci, 5):
                    end = min(begin + 5, ci)
                    budget.charge(48 + 16 * (end - begin))
                    keys = tuple(slots[channel * spatial + offset] for channel in range(begin, end))
                    if any(key is None for key in keys):
                        if not all(key is None for key in keys):
                            raise ValueError('mixed padding inside common spatial group')
                        group_keys.append(None)
                        continue
                    if keys not in groups:
                        forms, bounds = [], []
                        for key in keys:
                            form = first_form(packet, key, post, helper, k, budget)
                            bound = k.source_box_bounds(form, box, budget)
                            if key in sources and sources[key]['bounds'] != bound:
                                raise ValueError('same source obtained different certified bounds')
                            sources[key] = dict(coordinate=key, bounds=bound,
                                original_phase=(packet['first_relu']['output'], *key))
                            forms.append(form); bounds.append(bound)
                        n = len(keys)
                        budget.charge(64 + 32 * n)
                        knots = tuple(F(i, n - 1) for i in range(n))
                        residuals = []
                        for t, form in zip(knots[1:-1], forms[1:-1]):
                            line = k.affine_add(k.affine_scale(forms[0], k.point(ONE - t, budget), budget),
                                k.affine_scale(forms[-1], k.point(t, budget), budget), budget)
                            residual = k.affine_add(form, k.affine_scale(line, k.point(-1, budget), budget), budget)
                            residuals.append(k.source_box_bounds(residual, box, budget))
                        groups[keys] = dict(source_coordinates=keys, knots=knots,
                            source_bounds=tuple(bounds), residual_bounds=tuple(residuals))
                        last_forms = forms
                    group_keys.append(keys)
            valid = sum(key is not None for key in slots)
            if len(set(key for key in slots if key is not None)) != valid:
                raise ValueError('original source repeated in one Conv window')
            window = dict(branch=branch_index, position=(row, col), canonical_slots=tuple(slots),
                group_keys=tuple(group_keys), rows=[])
            windows.append(window); branch_windows.append(window)
            summary['canonical_slots'] += co * fanin
            summary['valid_slots'] += co * valid
            summary['padding_slots'] += co * (fanin - valid)
            summary['completed_windows'] += 1
        # A channel's complete templates are reused across all five fixed
        # windows, then released. Serialized references reconstruct them from
        # the original raw weights and this fixed template rule, without loss.
        for oc in range(co):
            budget.charge(128 + 24 * fanin)
            plans = []
            for offset in range(spatial):
                for begin in range(0, ci, 5):
                    end = min(begin + 5, ci)
                    weights = tuple(conv['weights'][oc * fanin + channel * spatial + offset]
                                    for channel in range(begin, end))
                    plans.append(query.template(weights, enabled=True, budget=budget))
            alpha, shift = helper.post_affine(branch['post_ops'], oc, k, budget)
            bias = k.point(conv['bias'][oc], budget)
            for window in branch_windows:
                budget.charge(128 + 16 * len(plans))
                totals = {arm: bias for arm in ARMS}
                for plan, group_key in zip(plans, window['group_keys']):
                    budget.charge(12)
                    if group_key is None:
                        # Structural ONNX padding is exactly zero in all arms;
                        # all original slots and this group's position remain.
                        continue
                    group = groups[group_key]
                    last_query = query._compiled_group_bounds(plan, group['source_bounds'],
                        group['residual_bounds'], budget)
                    for arm in ARMS:
                        totals[arm] = k.add(totals[arm], last_query[arm], budget)
                pre = {arm: query.post_affine(totals[arm], alpha, shift, budget) for arm in ARMS}
                relu = {arm: k.nonnegative_part(pre[arm], budget) for arm in ARMS}
                budget.charge(96)
                for name, values in (('preactivation', pre), ('relu', relu)):
                    if not (values['original'][0] <= values['independent'][0] <= values['common'][0]
                            <= values['common'][1] <= values['independent'][1] <= values['original'][1]):
                        raise ValueError('same-information envelope containment failed')
                    summary[name + '_lower_improved'] += int(values['common'][0] > values['independent'][0])
                    summary[name + '_upper_improved'] += int(values['common'][1] < values['independent'][1])
                improved = pre['common'] != pre['independent']
                summary['no_gain_rows'] += int(not improved)
                summary['outer_crossing_reference_rows'] += int(pre['independent'][0] < ZERO < pre['independent'][1])
                window['rows'].append(dict(channel=oc, raw_weight_slice_ref=(branch_index, oc),
                    template_rule='source_query.template; spatial-offset/channel-groups-5; tail-4; equal-knots',
                    raw_conv_bounds=totals, preactivation=pre, relu=relu,
                    common_strictly_improves_independent=improved))
                summary['rows'] += 1
            last_templates = plans
        emit(dict(event='branch_completed', branch=branch_index, completed_rows=summary['rows'],
                  whole_work_used=budget.used), budget)
    budget.charge(64 + 16 * (len(sources) + len(groups)))
    summary['source_forms'], summary['residual_groups'] = len(sources), len(groups)
    if (summary['rows'] != summary['expected_rows'] or summary['rows'] == 0
            or summary['completed_windows'] != summary['expected_windows']):
        raise ValueError('complete direct-branch population not established')
    result = dict(frame_identity=(packet['raw_model_sha256'], packet['input_name']),
        first_shape=first_shape, original_source_relu=packet['first_relu'], branches=branches,
        original_input_box=box, source_bounds=sources, residual_groups=groups,
        windows=windows, side_consumers=packet['side_consumers'], summary=summary)
    held = dict(first_post_affine=post, last_templates=last_templates,
                last_query=last_query, last_source_forms=last_forms,
                final_slots=slots, final_group_keys=group_keys)
    return result, held


def one_model(source, index, frozen, authroots, report, k, helper, extractor,
              packet_base, first_form, query, evidence, budget, meter):
    start = budget.used
    budget.limit = min(CAP, start + MODEL_CAP)
    model_path, spec_path = Path(source['model_path']), Path(source['spec_path'])
    parse_work = 4096 + model_path.stat().st_size + 8 * spec_path.stat().st_size
    budget.charge(parse_work)
    raw, _ = read_bytes(model_path, budget, source['model_sha256'], packet_base.MAX_RAW_BYTES, prepaid=True)
    spec, _ = read_bytes(spec_path, budget, source['spec_sha256'], prepaid=True)
    packet = extractor.extract_model(raw, enabled=True, input_batch=1)
    if packet['raw_model_sha256'] != source['model_sha256']:
        raise ValueError('decoded original identity changed')
    box = helper.input_box(spec, packet['input_shape'])
    emit(dict(event='source_parsed', model=source['model_relative_path'],
              input_shape=packet['input_shape'], whole_work_used=budget.used), budget)
    result, retained = source_census(packet, box, helper, k, first_form, query, budget)
    if result['summary']['rows'] != (320, 640, 640)[index]:
        raise ValueError('registered source receiver count differs')
    extractor_path = HERE.parent / 'd015_batch_binding_20260928_v2/source_binding_v2.py'
    payload = dict(schema='d120_mixed_consumer_source_v1', source=source,
        source_packet_ref=dict(model_sha256=source['model_sha256'],
            extractor_path=str(extractor_path), extractor_sha256=frozen['source_sha256'][str(extractor_path)],
            input_batch=1, scope='full original packet reconstructed; parameters not duplicated'),
        source_query_sha256=frozen['source_sha256'][str(HERE / 'source_query.py')],
        binding_mathematical_only=True, actual_phase_column_binding_verified=False, **result)
    roots = dict(model_raw=raw, spec_raw=spec, packet=packet, box=box, retained=retained,
        payload=payload, authentication=authroots, report=report,
        budget_state=vars(budget), evidence_state=vars(meter))
    initial_evidence = meter.used
    ledger = evidence.bounded_ledger(roots, meter)
    first_fanin = len(packet['first_conv']['weights']) // packet['first_conv']['weight_shape'][0]
    transient = (query.TRANSIENT_ENTRIES_PER_SLOT * result['summary']['max_slots']
        + query.TRANSIENT_ENTRY_BASE + 128 * first_fanin
        + 8 * packet['decoded_scalar_count'] + 32 * packet['graph']['node_count'])
    entries = ledger['retained_entries'] + transient
    if entries > ENTRY_CAP:
        raise ValueError('retained plus transient entry limit exceeded')
    name = 'complete_' + str(index) + '.json'
    partial, final = RUN / (name + '.partial'), RUN / name
    written = evidence.write_evidence(partial, payload, meter, {})
    if final.exists():
        raise ValueError('complete source evidence already exists')
    partial.rename(final)
    answer = dict(model=source['model_relative_path'], evidence_file=name,
        evidence_bytes=written['bytes'], evidence_sha256=written['sha256'],
        evidence_work=meter.used - initial_evidence, held_ledger=ledger,
        retained_entry_upper=entries, numerical_model_work=budget.used - start,
        extractor_prepaid_work=parse_work, summary=result['summary'])
    emit(dict(event='source_completed', model=source['model_relative_path'],
              summary=result['summary'], whole_work_used=budget.used), budget)
    return answer


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    started = time.monotonic()
    initial = None
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            initial = int(line.split()[1]) * 1024
            break
    bootstrap = Bootstrap()
    budget = meter = model_start = authroots = None
    branch_used = entries = 0
    report = dict(source_census_completed=False, source_census_qualified=False,
        memory_gate_passed=False, formal_gain=0, diagnostic_solver_calls=0,
        model_forward_calls=0, new_benchmark_solves=0, models=[], native_HZ_admitted=False,
        actual_phase_column_binding_verified=False, gpu_computation_completed=False,
        complete_physical_qualification=False, binding_mathematical_only=True)
    tracemalloc.start()
    try:
        if (initial is None or len(os.sched_getaffinity(0)) != 1
                or resource.getrlimit(resource.RLIMIT_AS) != (16 * 1024**3,) * 2
                or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not __debug__
                or not sys.dont_write_bytecode
                or any(os.environ.get(key) != '1' for key in
                       ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                or os.environ.get('TMPDIR') != str(RUN / 'tmp')):
            raise ValueError('CPU1 AS16GiB assertions isolated single-thread worker required')
        frozen, digest, authroots = authenticate(bootstrap)
        report.update(manifest_sha256=digest, selected_sources=frozen['selected_sources'])
        sys.path.insert(0, str(ROOT))
        from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
        from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import source_packet_v1 as packet_base
        from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2 import census_worker_v2 as helper
        from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2 import source_binding_v2 as extractor
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930.census import first_form
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence
        from experiments.neural_hz_20260831.definition_first_20260928.d120_mixed_consumer_source_20261002 import source_query as query
        budget = k.WorkBudget(enabled=True)
        budget.charge(bootstrap.used)
        budget.charge(EVIDENCE_CAP + RESERVE)
        meter = evidence.Meter(limit=EVIDENCE_CAP)
        for index, source in enumerate(frozen['selected_sources']):
            model_start = budget.used
            answer = one_model(source, index, frozen, authroots, report, k, helper, extractor,
                packet_base, first_form, query, evidence, budget, meter)
            report['models'].append(answer)
            branch_used = max(branch_used, budget.used - model_start)
            entries = max(entries, answer['retained_entry_upper'])
            model_start = None
        budget.limit = CAP
        meter_start = meter.used
        terminal = evidence.bounded_ledger((authroots, report, vars(budget), vars(meter)), meter)
        entries = max(entries, terminal['retained_entries'])
        report['terminal_ledger'] = terminal
        report['terminal_evidence_work'] = meter.used - meter_start
        report['rows'] = sum(model['summary']['rows'] for model in report['models'])
        report['canonical_slots'] = sum(model['summary']['canonical_slots'] for model in report['models'])
        report['source_census_completed'] = (len(report['models']) == 3
            and report['rows'] == 1600 and report['canonical_slots'] == 921600)
    except Exception as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        if budget is not None and model_start is not None:
            branch_used = max(branch_used, budget.used - model_start)
        _, peak = tracemalloc.get_traced_memory()
        metadata_bytes = tracemalloc.get_tracemalloc_memory()
        growth = (max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial)
                  if initial is not None else None)
        wall = time.monotonic() - started
        report.update(wall_s=wall, initial_rss_bytes=initial,
            rss_highwater_growth_bytes=growth, traced_peak_bytes=peak,
            tracer_metadata_bytes=metadata_bytes, final_summary_reserve_bytes=RESERVE,
            summary_reserve_bytes=RESERVE, whole_work_used=budget.used if budget else bootstrap.used,
            preimport_work_used=bootstrap.used, branch_work_used=branch_used,
            evidence_work_used=meter.used if meter else 0, retained_entries=entries,
            actual_cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
            scope='three original sources; five fixed windows; all channels and canonical Conv slots')
        report['memory_gate_passed'] = (report['source_census_completed'] and 'failure' not in report
            and growth is not None and growth + RESERVE <= MEMORY_CAP
            and peak + metadata_bytes + RESERVE <= MEMORY_CAP
            and entries <= ENTRY_CAP and wall <= 240 and report['whole_work_used'] <= CAP
            and branch_used <= MODEL_CAP and report['evidence_work_used'] <= EVIDENCE_CAP)
        report['source_census_qualified'] = report['source_census_completed'] and report['memory_gate_passed']
        data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        if len(data) > RESERVE:
            raise ValueError('summary exceeds prepaid reserve')
        with (RUN / 'diagnostic.json').open('xb') as stream:
            stream.write(data)
        print(data.decode(), end='', flush=True)
        tracemalloc.stop()
    return 0 if report['source_census_qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
