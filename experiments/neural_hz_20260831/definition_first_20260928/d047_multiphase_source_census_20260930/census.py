"""Single-use, source-only D047 census. No native HZ or GPU qualification.

Only stdlib code runs before authenticating the inherited manifest and every
actual project import. The supervisor checks the complete decoder/GPU/source
inventory before and after this worker; non-imported GPU libraries are not
loaded or hashed a second time here. Old entry points and globals are unused.
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
RUN = EXP / 'results/d047_multiphase_source_census_20260930_v1'
PRIOR = EXP / 'results/d046_multiphase_envelopes_20260930_v1'
PRIOR_SHA = 'a4a5eef3249b0161b0be4322370faba4c42bdfd79d77f4bde3720f5757b99904'
NEW_FILES = ('PREREG.md', 'THEORY.md', 'seed_relation.py',
             'test_seed_relation.py', 'census.py', 'run_census.py')
TEST_NAMES = ('test_interval_seed_soundness', 'test_stable_padding_and_identity',
              'test_multiphase_transfer_and_metrics', 'test_disabled_and_rejected_premises')
CAP, MODEL_CAP, EVIDENCE_CAP = 256_000_000, 200_000_000, 40_000_000
MEMORY_CAP, RESERVE, ENTRY_CAP = 1024**3, 65536, 64_000_000
AFTER = ('difference_lower', 'difference_upper', 'companion_lower', 'companion_upper')
ZERO = F(0)


class PreflightBudget:
    def __init__(self):
        self.used = 0

    def charge(self, amount):
        if (type(amount) is not int or amount < 0
                or amount > min(MODEL_CAP, CAP - EVIDENCE_CAP - RESERVE) - self.used):
            raise ValueError('pre-import work budget exceeded')
        self.used += amount


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def read_bytes(path, budget, expected=None, maximum=8_000_000, prepaid=False):
    if path.is_symlink() or not path.is_file():
        raise ValueError('ordinary authenticated file required')
    size = path.stat().st_size
    if not 0 <= size <= maximum:
        raise ValueError('authenticated file exceeds size bound')
    if not prepaid:
        budget.charge(4096 + size)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != size or (expected is not None and digest != expected):
        raise ValueError('authenticated source changed: ' + str(path))
    return raw, digest


def parse_json(raw, budget):
    def integer(text):
        budget.charge(3 + len(text))
        if len(text) > 160:
            raise ValueError('oversized metadata integer')
        value = int(text)
        if abs(value).bit_length() > 512:
            raise ValueError('metadata integer exceeds bit cap')
        return value

    def real(text):
        budget.charge(3 + len(text))
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('nonfinite metadata')
        return value

    def pairs(items):
        budget.charge(5 + 5 * len(items))
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate metadata key')
            result[key] = value
        return result

    def invalid(text):
        raise ValueError('nonstandard metadata number')

    return json.loads(raw, parse_int=integer, parse_float=real,
                      parse_constant=invalid, object_pairs_hook=pairs)


def digest_map(value, budget):
    if type(value) is not dict or len(value) > 50_000:
        raise ValueError('bounded digest mapping required')
    budget.charge(8 * len(value))
    for path, digest in value.items():
        if (type(path) is not str or len(path) > 4096 or not Path(path).is_absolute()
                or type(digest) is not str or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid frozen digest')
    return value


def authenticate(budget):
    prior_raw, _ = read_bytes(PRIOR / 'preregistered.json', budget, PRIOR_SHA)
    prior = parse_json(prior_raw, budget)
    raw, _ = read_bytes(RUN / 'preregistered.json', budget)
    frozen = parse_json(raw, budget)
    freeze_raw, freeze_sha = read_bytes(HERE / 'freeze.json', budget)
    freeze = parse_json(freeze_raw, budget)
    identities = digest_map(frozen.get('source_sha256'), budget)
    inputs = digest_map(frozen.get('input_sha256'), budget)
    old_sources = digest_map(prior.get('source_sha256'), budget)
    old_inputs = digest_map(prior.get('input_sha256'), budget)
    new_sources = digest_map(freeze.get('source_sha256'), budget)
    budget.charge(8 * (len(old_sources) + len(old_inputs)))
    if (any(identities.get(p) != d for p, d in old_sources.items())
            or any(inputs.get(p) != d for p, d in old_inputs.items())
            or identities.get(str(PRIOR / 'preregistered.json')) != PRIOR_SHA
            or identities.get(str(HERE / 'freeze.json')) != freeze_sha
            or frozen.get('freeze_path') != str(HERE / 'freeze.json')
            or frozen.get('freeze_sha256') != freeze_sha
            or frozen.get('provenance') != prior.get('provenance')
            or frozen.get('selected_sources') != prior.get('selected_sources')
            or len(frozen.get('selected_sources', ())) != 3
            or frozen.get('decoder_dependency_files') != prior.get('decoder_dependency_files')
            or frozen.get('gpu_dependency_files') != prior.get('gpu_dependency_files')):
        raise ValueError('inherited source/input/decoder/provenance binding differs')
    expected = dict(required_tests=3781, required_test_files=172, expected_models=3,
        address_space_bytes=16 * 1024**3, tests_combined_wall_cap_s=60,
        worker_wall_cap_s=240, whole_work_cap=CAP, branch_work_cap=MODEL_CAP,
        host_memory_cap_bytes=MEMORY_CAP, retained_entry_cap=ENTRY_CAP,
        rational_bit_cap=512, summary_reserve_bytes=RESERVE, evidence_prepaid_work=EVIDENCE_CAP)
    if any(type(frozen.get(key)) is not int or frozen[key] != value for key, value in expected.items()):
        raise ValueError('fixed resource or population limits differ')
    if (freeze.get('schema') != 'd047_frozen_v1'
            or freeze.get('required_tests') != 3781 or freeze.get('required_test_files') != 172
            or freeze.get('new_test_names') != list(TEST_NAMES)
            or set(new_sources) != {str(HERE / name) for name in NEW_FILES}):
        raise ValueError('frozen six-file/four-test contract differs')
    old_ids, ids = prior.get('expected_nodeids'), frozen.get('expected_nodeids')
    budget.charge(12 * (3777 + 3781))
    prefix = str((HERE / 'test_seed_relation.py').relative_to(ROOT)) + '::'
    if (type(old_ids) is not list or len(old_ids) != 3777 or len(set(old_ids)) != 3777
            or type(ids) is not list or len(ids) != 3781 or len(set(ids)) != 3781
            or set(ids) != set(old_ids) | {prefix + name for name in TEST_NAMES}
            or frozen.get('tests') != [*prior['tests'], str(HERE / 'test_seed_relation.py')]
            or len(frozen['tests']) != 172):
        raise ValueError('complete inherited test population differs')
    for path, digest in new_sources.items():
        if identities.get(path) != digest:
            raise ValueError('new source lacks supervisor binding')
        read_bytes(Path(path), budget, digest)
    project_imports = (
        HERE / 'seed_relation.py',
        HERE.parent / 'd046_multiphase_envelopes_20260930/multiphase.py',
        HERE.parent / 'd015_source_shielding_20260928/shield_kernel_v1.py',
        HERE.parent / 'd015_source_shielding_20260928/source_packet_v1.py',
        HERE.parent / 'd015_batch_binding_20260928_v2/source_binding_v2.py',
        HERE.parent / 'd015_batch_binding_20260928_v2/census_worker_v2.py',
        HERE.parent / 'd025_interval_capacity_20260930/census.py',
        HERE.parent / 'd025_interval_capacity_20260930/evidence.py',
    )
    checked = {}
    for module in project_imports:
        package = module.parent
        while package != ROOT:
            initializer = package / '__init__.py'
            if initializer.exists() and str(initializer) not in checked:
                digest = old_sources.get(str(initializer))
                if digest is None or identities.get(str(initializer)) != digest:
                    raise ValueError('package initializer not inherited')
                read_bytes(initializer, budget, digest)
                checked[str(initializer)] = digest
            package = package.parent
        authority = new_sources if module.parent == HERE else old_sources
        digest = authority.get(str(module))
        if digest is None or identities.get(str(module)) != digest:
            raise ValueError('actual project import not authenticated')
        read_bytes(module, budget, digest)
        checked[str(module)] = digest
    for source in frozen['selected_sources']:
        if any(inputs.get(source[k + '_path']) != source[k + '_sha256'] for k in ('model', 'spec')):
            raise ValueError('selected original input not bound')
    return frozen, (prior_raw, prior, raw, frozen, freeze_raw, freeze, checked)


def source_census(packet, box, helper, k, first_form, compile_pair, budget):
    first_shape = helper.conv_shape(packet['first_conv'], packet['input_shape'])
    budget.charge(128 + 32 * len(packet['branches']) + 16 * first_shape[1])
    post = tuple(helper.post_affine(packet['first_post_ops'], c, k, budget)
                 for c in range(first_shape[1]))
    cache, receivers, windows, branches = {}, {}, [], []
    summary = dict(expected_windows=0, completed_windows=0,
        expected_receiver_rows=0, completed_receiver_rows=0, expected_pairs=0, completed_pairs=0,
        canonical_slots=0, valid_slots=0, padding_slots=0, pair_canonical_slots=0,
        unique_source_count=0, strict_crossing_source_count=0, nonconstant_relations=0,
        hinge_crossing_relations=0, last_anchor_strict_relations=0,
        physical_coordinate_nnz_upper=0, ordinary_crossing_receiver_rows=0,
        max_slots=0, support_totals={name: 0 for name in AFTER},
        original_bits_deleted=0, actual_phase_column_binding_verified=False)
    for branch_index, branch in enumerate(packet['branches']):
        conv = branch['conv']
        shape = helper.conv_shape(conv, first_shape)
        co = shape[1]
        fanin = conv['weight_shape'][1] * conv['weight_shape'][2] * conv['weight_shape'][3]
        if not 1 <= fanin <= 65536 or len(conv['weights']) != co * fanin:
            raise ValueError('complete Conv coefficient population unsupported')
        budget.charge(96 + 8 * co)
        channels = tuple((i, min(i + 1, co - 1)) for i in range(0, co, 2))
        positions = helper.anchors(shape[2], shape[3])
        admitted = branch['target_relu'] is not None
        branches.append(dict(index=branch_index, admitted=admitted,
            conv_output=conv['output'], consumer_relu=branch['target_relu'],
            output_shape=shape, weight_shape=conv['weight_shape'], strides=conv['strides'],
            pads=conv['pads'], dilations=conv['dilations'], group=conv['group'],
            receiver_count=co, channel_pairs=channels, positions=positions,
            stop=branch['stop'], input_aliases=branch['input_aliases']))
        if not admitted:
            continue
        summary['expected_windows'] += len(positions)
        summary['expected_receiver_rows'] += co * len(positions)
        summary['expected_pairs'] += len(channels) * len(positions)
        summary['max_slots'] = max(summary['max_slots'], fanin)
        budget.charge(128 + co * (32 + 8 * fanin))
        receiver_bank = []
        for channel in range(co):
            alpha, shift = helper.post_affine(branch['post_ops'], channel, k, budget)
            bias = k.add(k.mul(alpha, k.point(conv['bias'][channel], budget), budget), shift, budget)
            weights = tuple(k.mul(alpha, k.point(value, budget), budget)
                            for value in conv['weights'][channel * fanin:(channel + 1) * fanin])
            receiver_bank.append(dict(channel=channel, weights=weights, bias=bias,
                                      alpha=alpha, shift=shift))
        receivers[branch_index] = receiver_bank
        for row, col in positions:
            budget.charge(128 + 48 * fanin + 32 * co)
            emit(dict(event='window_started', branch=branch_index, position=(row, col),
                      whole_work_used=budget.used))
            slots = [None] * fanin
            for ci, iy, ix, offset in helper.receptive(conv, first_shape, row, col):
                coordinate = (ci, iy, ix)
                if slots[offset] is not None:
                    raise ValueError('duplicate canonical source slot')
                slots[offset] = coordinate
                if coordinate not in cache:
                    form = first_form(packet, coordinate, post, helper, k, budget)
                    bound = k.source_box_bounds(form, box, budget)
                    del form  # Real construction/support happened; only its bound is retained.
                    ordinal = (ci * first_shape[2] + iy) * first_shape[3] + ix
                    cache[coordinate] = dict(coordinate=coordinate, ordinal=ordinal, bounds=bound)
            bounds = tuple((ZERO, ZERO) if key is None else cache[key]['bounds'] for key in slots)
            ordinals = tuple(None if key is None else cache[key]['ordinal'] for key in slots)
            real_count = sum(key is not None for key in slots)
            if len({v for v in ordinals if v is not None}) != real_count:
                raise ValueError('canonical window repeats an original source')
            window = dict(branch=branch_index, position=(row, col), source_slots=tuple(slots),
                          source_ordinals=ordinals, pairs=[])
            windows.append(window)
            for left, right in channels:
                budget.charge(128)
                a, b = receiver_bank[left], receiver_bank[right]
                relation = compile_pair(a['weights'], a['bias'], b['weights'], b['bias'],
                    bounds, ordinals, same_consumer=left == right, enabled=True, budget=budget)
                window['pairs'].append(dict(channels=(left, right), relation=relation,
                    receiver_coefficients_ref=((branch_index, left), (branch_index, right))))
                summary['completed_pairs'] += 1
                summary['completed_receiver_rows'] += 1 + int(left != right)
                summary['pair_canonical_slots'] += fanin
                for key in ('nonconstant_relations', 'hinge_crossing_relations', 'last_anchor_strict_relations'):
                    summary[key] += relation[key]
                summary['physical_coordinate_nnz_upper'] += relation['potential_coordinate_nnz']
                for name in AFTER:
                    summary['support_totals'][name] += relation['supports'][name]
                summary['ordinary_crossing_receiver_rows'] += int(relation['ordinary_h'][0] < 0 < relation['ordinary_h'][1])
                if left != right:
                    summary['ordinary_crossing_receiver_rows'] += int(relation['ordinary_w'][0] < 0 < relation['ordinary_w'][1])
            summary['canonical_slots'] += co * fanin
            summary['valid_slots'] += co * real_count
            summary['padding_slots'] += co * (fanin - real_count)
            summary['completed_windows'] += 1
            emit(dict(event='window_completed', branch=branch_index, position=(row, col),
                completed_pairs=summary['completed_pairs'], whole_work_used=budget.used))
    if (not windows or any(summary['expected_' + kind] != summary['completed_' + kind]
                          for kind in ('windows', 'receiver_rows', 'pairs'))):
        raise ValueError('complete direct-branch population not established')
    budget.charge(64 + 24 * len(cache) + 8 * len(cache) * len(cache).bit_length())
    summary['unique_source_count'] = len(cache)
    summary['strict_crossing_source_count'] = sum(v['bounds'][0] < 0 < v['bounds'][1] for v in cache.values())
    result = dict(frame_identity=(packet['raw_model_sha256'], packet['input_name']),
        first_shape=first_shape, original_source_relu=packet['first_relu'], branches=branches,
        source_bounds=tuple(cache[key] for key in sorted(cache)), windows=windows,
        side_consumers=packet['side_consumers'], summary=summary)
    # Keep all numerically retained roots until evidence sealing. bounds/ordinals
    # are the final window's actual temporary arrays, not regenerated proxies.
    held = dict(cache=cache, receivers=receivers, first_post_affine=post,
                final_bounds=bounds, final_ordinals=ordinals, final_slots=slots)
    return result, held


def one_model(source, index, frozen, authroots, report, k, helper, extractor,
              packet_base, first_form, kernel, evidence, budget, meter):
    model_start = budget.used
    budget.limit = min(CAP, model_start + MODEL_CAP)
    model_path, spec_path = Path(source['model_path']), Path(source['spec_path'])
    # Unchanged D025 reader contract: an amortized parsing charge, NOT a CPU
    # instruction count. New wrapper/authentication/kernel work is additional.
    parse_work = 4096 + model_path.stat().st_size + 8 * spec_path.stat().st_size
    budget.charge(parse_work)
    raw, _ = read_bytes(model_path, budget, source['model_sha256'],
                        packet_base.MAX_RAW_BYTES, prepaid=True)
    spec, _ = read_bytes(spec_path, budget, source['spec_sha256'], prepaid=True)
    packet = extractor.extract_model(raw, enabled=True, input_batch=1)
    if packet['raw_model_sha256'] != source['model_sha256']:
        raise ValueError('extractor original byte identity differs')
    box = helper.input_box(spec, packet['input_shape'])
    emit(dict(event='source_parsed', model=source['model_relative_path'],
              input_shape=packet['input_shape'], whole_work_used=budget.used))
    result, retained = source_census(packet, box, helper, k, first_form, kernel.compile_pair, budget)
    extractor_path = HERE.parent / 'd015_batch_binding_20260928_v2/source_binding_v2.py'
    payload = dict(schema='d047_multiphase_source_census_v1', source=source,
        source_packet_ref=dict(model_sha256=source['model_sha256'],
            extractor_path=str(extractor_path), extractor_sha256=frozen['source_sha256'][str(extractor_path)],
            input_batch=1, scope='complete packet reconstructed from original bytes; parameters not duplicated'),
        binding_mathematical_only=True, actual_phase_column_binding_verified=False, **result)
    roots = dict(model_raw=raw, spec_raw=spec, packet=packet, box=box,
        retained=retained, payload=payload, authentication=authroots, report=report,
        budget_state=vars(budget), evidence_state=vars(meter))
    evidence_start = meter.used
    held = evidence.bounded_ledger(roots, meter)
    first_fanin = len(packet['first_conv']['weights']) // packet['first_conv']['weight_shape'][0]
    temporary = (kernel.TRANSIENT_ENTRIES_PER_SLOT * result['summary']['max_slots']
                 + kernel.TRANSIENT_ENTRY_BASE + 128 * first_fanin
                 + 8 * packet['decoded_scalar_count'] + 32 * packet['graph']['node_count'])
    entries = held['retained_entries'] + temporary
    if entries > ENTRY_CAP:
        raise ValueError('complete retained plus temporary entry cap exceeded')
    name = 'complete_' + str(index) + '.json'
    partial, final = RUN / (name + '.partial'), RUN / name
    written = evidence.write_evidence(partial, payload, meter, {})
    if final.exists():
        raise ValueError('complete evidence already exists')
    partial.rename(final)
    answer = dict(model=source['model_relative_path'], evidence_file=name,
        evidence_sha256=written['sha256'], evidence_bytes=written['bytes'],
        evidence_work=meter.used - evidence_start, held_ledger=held,
        retained_entry_upper=entries, numerical_model_work=budget.used - model_start,
        extractor_prepaid_work=parse_work, summary=result['summary'])
    emit(dict(event='source_completed', model=source['model_relative_path'],
              summary=result['summary'], whole_work_used=budget.used))
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
    bootstrap = PreflightBudget()
    budget, meter, model_start, authroots = None, None, None, None
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
        frozen, authroots = authenticate(bootstrap)
        sys.path.insert(0, str(ROOT))
        from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
        from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import source_packet_v1 as packet_base
        from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2 import census_worker_v2 as helper
        from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2 import source_binding_v2 as extractor
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930.census import first_form
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence
        from experiments.neural_hz_20260831.definition_first_20260928.d047_multiphase_source_census_20260930 import seed_relation as kernel
        budget = k.WorkBudget(enabled=True)
        budget.charge(bootstrap.used)
        budget.charge(EVIDENCE_CAP + RESERVE)
        meter = evidence.Meter(limit=EVIDENCE_CAP)
        for index, source in enumerate(frozen['selected_sources']):
            model_start = budget.used
            answer = one_model(source, index, frozen, authroots, report, k, helper,
                extractor, packet_base, first_form, kernel, evidence, budget, meter)
            report['models'].append(answer)
            entries = max(entries, answer['retained_entry_upper'])
            branch_used = max(branch_used, budget.used - model_start)
            model_start = None
        # Include final authentication/report roots in the same global meter.
        terminal = evidence.bounded_ledger((authroots, report, vars(budget), vars(meter)), meter)
        entries = max(entries, terminal['retained_entries'])
        report['terminal_ledger'] = terminal
        report['source_census_completed'] = len(report['models']) == 3
    except Exception as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
        # No unbudgeted recovery traversal of incomplete model/evidence roots.
    finally:
        if budget is not None and model_start is not None:
            branch_used = max(branch_used, budget.used - model_start)
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = (max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial)
                  if initial is not None else None)
        wall = time.monotonic() - started
        report.update(wall_s=wall, initial_rss_bytes=initial,
            rss_highwater_growth_bytes=growth, traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
            final_summary_reserve_bytes=RESERVE, summary_reserve_bytes=RESERVE,
            whole_work_used=budget.used if budget is not None else bootstrap.used,
            preimport_work_used=bootstrap.used, branch_work_used=branch_used,
            evidence_work_used=meter.used if meter is not None else 0, retained_entries=entries,
            actual_cpu_affinity=list(os.sched_getaffinity(0)),
            address_space_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
            scope='complete fixed direct branches, windows and consumer pairs of all three original sources')
        report['memory_gate_passed'] = (report['source_census_completed'] and 'failure' not in report
            and growth is not None
            and growth + RESERVE <= MEMORY_CAP and peak + metadata + RESERVE <= MEMORY_CAP
            and entries <= ENTRY_CAP and wall <= 240 and report['whole_work_used'] <= CAP
            and branch_used <= MODEL_CAP and report['evidence_work_used'] <= EVIDENCE_CAP)
        report['source_census_qualified'] = report['source_census_completed'] and report['memory_gate_passed']
        data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        if len(data) > RESERVE:
            raise ValueError('final summary exceeds prepaid reserve')
        with (RUN / 'diagnostic.json').open('xb') as stream:
            stream.write(data)
        print(data.decode(), end='', flush=True)
        tracemalloc.stop()
    return 0 if report['source_census_qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
