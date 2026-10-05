"""One-use, default-off D036 classification of a complete authenticated archive.

This is CPU rational arithmetic, not native phase-column binding or a verifier.
The supervisor checks COMPLETE inherited source/input drift before and after
this worker.  Before importing any project module this worker independently
authenticates the old manifest anchor, all six frozen new files and every
actual project import (including any existing package __init__).  It does not
rehash non-imported GPU libraries a second time or reduce the supervisor's
inherited test/source population.  No old main, model loader or solver runs.
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
RUN = EXP / 'results/d037_descendant_census_20260930_v1'
PRIOR = EXP / 'results/d025_interval_capacity_20260930_v1'
ARCHIVE = PRIOR / 'complete_0.json'
ARCHIVE_SHA = 'fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0'
ARCHIVE_BYTES = 12_062_002
PRIOR_MANIFEST_SHA = 'c23b0541c696375224a6aae6b530ecce9609cece8806b16d53e1bda517860a4a'
MODEL_SHA = '5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16'
SPEC_SHA = 'caca9ef2245019dd70883661bd5c102544f64152ddc0ea8f967eb8ed0882e996'
CAP, MODEL_CAP, EVIDENCE_CAP = 256_000_000, 200_000_000, 40_000_000
MEMORY_CAP, RESERVE, ENTRY_CAP = 1024**3, 65_536, 64_000_000
POSITIONS = ((0, 0), (0, 31), (16, 16), (31, 0), (31, 31))
NEW_FILES = ('PREREG.md', 'THEORY.md', 'amplitude.py', 'test_amplitude.py',
             'archive_worker.py', 'run_reference.py')
FORMULA = 'amplitude.compile_receiver:d037_interval_lowercuts_v1'
MASK_NNZ = (2, 3, 3, 4)
ZERO = Fraction(0)


class PreflightBudget:
    """Bound stdlib-only authentication before importing the real ledger."""

    def __init__(self):
        self.used = 0
        self.max_bits = 512

    def charge(self, amount):
        if (type(amount) is not int or amount < 0
                or amount > min(MODEL_CAP, CAP - EVIDENCE_CAP - RESERVE) - self.used):
            raise ValueError('pre-import authentication work limit exceeded')
        self.used += amount


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def parse_json(raw, budget):
    def integer(text):
        budget.charge(3 + len(text))
        if len(text) > 160:
            raise ValueError('JSON integer text exceeds bound')
        answer = int(text)
        if abs(answer).bit_length() > 512:
            raise ValueError('JSON integer exceeds 512 bits')
        return answer

    def real(text):
        budget.charge(3 + len(text))
        answer = float(text)
        if not math.isfinite(answer):
            raise ValueError('nonfinite JSON float')
        return answer

    def constant(text):
        raise ValueError('nonstandard JSON number')

    def object_pairs(pairs):
        budget.charge(5 + 5 * len(pairs))
        answer = {}
        for key, value in pairs:
            if key in answer:
                raise ValueError('duplicate JSON key')
            answer[key] = value
        return answer

    return json.loads(raw, parse_int=integer, parse_float=real,
                      parse_constant=constant, object_pairs_hook=object_pairs)


def read_bytes(path, budget, maximum=8_000_000, expected=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError('authentication requires an ordinary file')
    size = path.stat().st_size
    if not 0 < size <= maximum:
        raise ValueError('authenticated file size outside bound')
    budget.charge(4096 + size)
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError('file size changed while authenticating')
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('authenticated file digest differs: ' + str(path))
    return raw, digest


def digest_map(value, budget):
    if type(value) is not dict or len(value) > 50_000:
        raise ValueError('invalid manifest digest mapping')
    budget.charge(8 * len(value))
    for key, digest in value.items():
        if (type(key) is not str or len(key) > 4096 or not Path(key).is_absolute()
                or type(digest) is not str or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid manifest digest binding')
    return value


def authenticate(budget):
    prior_raw, _ = read_bytes(PRIOR / 'preregistered.json', budget,
                              expected=PRIOR_MANIFEST_SHA)
    prior = parse_json(prior_raw, budget)
    manifest_raw, _ = read_bytes(RUN / 'preregistered.json', budget)
    frozen = parse_json(manifest_raw, budget)
    freeze_raw, freeze_sha = read_bytes(HERE / 'freeze.json', budget)
    freeze = parse_json(freeze_raw, budget)
    if any(type(value) is not dict for value in (prior, frozen, freeze)):
        raise ValueError('manifest object required')
    identities = digest_map(frozen.get('source_sha256'), budget)
    inputs = digest_map(frozen.get('input_sha256'), budget)
    prior_sources = digest_map(prior.get('source_sha256'), budget)
    prior_inputs = digest_map(prior.get('input_sha256'), budget)
    budget.charge(4 * (len(prior_sources) + len(prior_inputs)))
    if (any(identities.get(path) != digest for path, digest in prior_sources.items())
            or any(inputs.get(path) != digest for path, digest in prior_inputs.items())
            or identities.get(str(PRIOR / 'preregistered.json')) != PRIOR_MANIFEST_SHA
            or identities.get(str(HERE / 'freeze.json')) != freeze_sha
            or frozen.get('freeze_path') != str(HERE / 'freeze.json')
            or frozen.get('freeze_sha256') != freeze_sha
            or frozen.get('selected_sources') != prior.get('selected_sources')
            or type(frozen.get('selected_sources')) is not list
            or len(frozen['selected_sources']) != 3
            or inputs.get(str(ARCHIVE)) != ARCHIVE_SHA
            or frozen.get('archive_path') != str(ARCHIVE)
            or frozen.get('archive_sha256') != ARCHIVE_SHA
            or frozen.get('archive_bytes') != ARCHIVE_BYTES
            or frozen.get('positions') != [list(p) for p in POSITIONS]):
        raise ValueError('manifest inherited identity, archive or freeze binding differs')
    expected_caps = dict(address_space_bytes=16 * 1024**3, tests_combined_wall_cap_s=60,
        worker_wall_cap_s=240, whole_work_cap=CAP, branch_work_cap=MODEL_CAP,
        host_memory_cap_bytes=MEMORY_CAP, retained_entry_cap=ENTRY_CAP,
        rational_bit_cap=512, summary_reserve_bytes=RESERVE, evidence_prepaid_work=EVIDENCE_CAP,
        expected_rows=320, canonical_edges=184_320, valid_edges=102_400, padding_edges=81_920)
    budget.charge(8 * len(expected_caps))
    if any(type(frozen.get(key)) is not int or frozen[key] != value
           for key, value in expected_caps.items()):
        raise ValueError('manifest fixed resource or population limits differ')
    new_sources = digest_map(freeze.get('source_sha256'), budget)
    if (freeze.get('schema') != 'd037_frozen_v1'
            or set(new_sources) != {str(HERE / name) for name in NEW_FILES}
            or freeze.get('required_tests') != 3759 or freeze.get('required_test_files') != 169
            or frozen.get('required_tests') != 3759 or frozen.get('required_test_files') != 169):
        raise ValueError('new frozen source or test population differs')
    names, nodeids = freeze.get('new_test_names'), frozen.get('expected_nodeids')
    old_nodeids = prior.get('expected_nodeids')
    if (type(names) is not list or len(names) != 6
            or any(type(name) is not str or not re.fullmatch(r'test_[A-Za-z0-9_]+', name) for name in names)
            or len(set(names)) != 6 or type(nodeids) is not list or len(nodeids) != 3759
            or type(old_nodeids) is not list or len(old_nodeids) != 3753
            or any(type(name) is not str or len(name) > 4096 for name in nodeids + old_nodeids)):
        raise ValueError('complete inherited test nodeids required')
    budget.charge(12 * (len(nodeids) + len(old_nodeids)))
    prefix = str((HERE / 'test_amplitude.py').relative_to(ROOT)) + '::'
    if (len(set(nodeids)) != 3759 or len(set(old_nodeids)) != 3753
            or set(nodeids) != set(old_nodeids) | {prefix + name for name in names}):
        raise ValueError('inherited nodeids omitted or new test population changed')
    for path, digest in new_sources.items():
        if identities.get(path) != digest:
            raise ValueError('new source is not bound by supervisor manifest')
        read_bytes(Path(path), budget, expected=digest)
    project_imports = (
        HERE / 'amplitude.py',
        HERE.parent / 'd025_interval_capacity_20260930/interval_capacity.py',
        HERE.parent / 'd025_interval_capacity_20260930/evidence.py',
        HERE.parent / 'd015_source_shielding_20260928/shield_kernel_v1.py',
    )
    checked_imports = {}
    for module in project_imports:
        package = module.parent
        while package != ROOT:
            initializer = package / '__init__.py'
            if initializer.exists() and str(initializer) not in checked_imports:
                digest = identities.get(str(initializer))
                if digest is None or prior_sources.get(str(initializer)) != digest:
                    raise ValueError('project package initializer lacks inherited authentication')
                read_bytes(initializer, budget, expected=digest)
                checked_imports[str(initializer)] = digest
            package = package.parent
        digest = identities.get(str(module))
        authority = new_sources if module.parent == HERE else prior_sources
        if digest is None or authority.get(str(module)) != digest:
            raise ValueError('actual import lacks authenticated source identity')
        read_bytes(module, budget, expected=digest)
        checked_imports[str(module)] = digest
    return frozen, (prior_raw, prior, manifest_raw, frozen, freeze_raw, freeze, checked_imports)


def fraction(value, budget):
    budget.charge(10)
    if (type(value) is not list or len(value) != 2
            or any(type(v) is not int for v in value) or value[1] <= 0
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


def form(value, budget):
    budget.charge(6)
    if type(value) is not list or len(value) != 2 or type(value[1]) is not dict:
        raise ValueError('invalid archive affine form')
    constant, coefficients = interval(value[0], budget), {}
    budget.charge(5 * len(value[1]))
    for key, coefficient in value[1].items():
        if (type(key) is not str or not key.isdecimal() or len(key) > 7
                or str(int(key)) != key or not 0 <= int(key) < 3072):
            raise ValueError('invalid original input coordinate')
        coefficients[int(key)] = interval(coefficient, budget)
    return constant, coefficients


def decode(data, frozen, budget):
    budget.charge(1024)
    if type(data) is not dict or set(data) != {'box', 'model_raw', 'packet', 'result', 'source', 'spec_raw'}:
        raise ValueError('complete source archive schema differs')
    source, packet, result = data['source'], data['packet'], data['result']
    if (source != frozen['selected_sources'][0]
            or source['model_sha256'] != MODEL_SHA or source['spec_sha256'] != SPEC_SHA
            or data['model_raw']['sha256'] != MODEL_SHA or data['spec_raw']['sha256'] != SPEC_SHA
            or packet['raw_model_sha256'] != MODEL_SHA
            or packet['schema'] != 'd015_raw_first_bank_v1' or packet['input_name'] != 'modelInput'
            or result['frame_identity'] != [MODEL_SHA, 'modelInput']
            or result['first_shape'] != [1, 64, 32, 32] or packet['first_relu']['output'] != '127'):
        raise ValueError('original source/frame archive identity differs')
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
            raise ValueError('duplicate or incorrectly archived source phase')
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
                or len(window['source_slots']) != 576 or len(window['original_phases']) != 576
                or len(window['source_bounds']) != 576 or len(window['rows']) != 64):
            raise ValueError('canonical window shape differs')
        row, col = position
        slot_forms, slot_bounds, real_slots, original_phases = [], [], [], []
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
                raise ValueError('canonical archived original phase differs')
            bound = interval(window['source_bounds'][offset], budget)
            if bound != (bounds[index] if index is not None else (ZERO, ZERO)):
                raise ValueError('saved source premise differs')
            slot_forms.append(forms[index] if index is not None else zero_form)
            slot_bounds.append(bound)
            real_slots.append(index is not None)
            original_phases.append(phases[index] if index is not None else None)
            if index is not None:
                seen_sources.add(index)
        old_rows = []
        for channel, old in enumerate(window['rows']):
            if (old['channel'] != channel or old['receiver_coefficients_ref'] != [0, channel]
                    or old['all_slot_and_pair_premises_ref'] != [0, row, col]):
                raise ValueError('saved receiver premise binding differs')
            old_rows.append(interval(old['ordinary_bounds'], budget))
        decoded_windows.append(dict(position=position, slot_forms=tuple(slot_forms),
            slot_bounds=tuple(slot_bounds), real_slots=tuple(real_slots),
            original_phases=tuple(original_phases), old_rows=tuple(old_rows)))
    if seen_sources != set(forms):
        raise ValueError('not all saved source forms are consumed')
    return box, forms, bounds, phases, tuple(decoded_receivers), tuple(decoded_windows)


def check_result(result, window, ordinary, budget):
    budget.charge(64 + 24 * 576)
    if (type(result) is not dict or result.get('ordinary_bounds') != ordinary
            or any(type(result.get(key)) is not tuple or len(result[key]) != 576
                   for key in ('source_caps', 'residual_bounds', 'row_masks'))):
        raise ValueError('complete kernel population or old ordinary bound differs')
    counts = [0, 0, 0, 0]
    potential_edges = potential_rows = potential_nnz = valid = 0
    for real, mask in zip(window['real_slots'], result['row_masks']):
        if type(mask) is not int or not 0 <= mask <= 15 or (not real and mask):
            raise ValueError('invalid or nonzero padding mask')
        bits = tuple((mask >> bit) & 1 for bit in range(4))
        number, nnz = sum(bits), sum(bits[i] * MASK_NNZ[i] for i in range(4))
        if number > 2 or nnz > 6:
            raise ValueError('uniform lowercut population bound exceeded')
        for bit in range(4):
            counts[bit] += bits[bit]
        potential_edges += int(mask != 0)
        potential_rows += number
        potential_nnz += nnz
        valid += int(real)
    expected = dict(canonical_slots=576, valid_slots=valid, padding_slots=576-valid,
        potential_edges=potential_edges, potential_rows=potential_rows,
        potential_coordinate_nnz=potential_nnz)
    if (any(type(result.get(key)) is not int or result[key] != value for key, value in expected.items())
            or result.get('row_unresolved_counts') != tuple(counts)
            or any(type(result.get(key)) is not int or not 0 <= result[key] <= 576
                   for key in ('zero_coeff_slots', 'interval_cross_zero_slots'))):
        raise ValueError('kernel masks and count ledger disagree')
    return dict(canonical_edges=576, valid_edges=valid, padding_edges=576-valid,
        potential_edges=potential_edges, potential_rows=potential_rows, potential_nnz=potential_nnz,
        row_unresolved_counts=tuple(counts), zero_coeff_slots=result['zero_coeff_slots'],
        interval_cross_zero_slots=result['interval_cross_zero_slots'])


def main():
    start, initial = time.monotonic(), None
    budget = meter = None
    bootstrap = PreflightBudget()
    entries, numerical_start = 0, EVIDENCE_CAP + RESERVE
    report = dict(archive_completed=False, memory_gate_passed=False, expected_rows=320,
        completed_rows=0, archive_sha256=ARCHIVE_SHA, formal_gain=0, diagnostic_solver_calls=0,
        model_forward_calls=0, new_benchmark_solves=0, native_HZ_admitted=False,
        gpu_computation_completed=False, complete_physical_qualification=False,
        source_census_qualified=False, binding_mathematical_only=True,
        actual_phase_column_binding_verified=False, binding_math_verified=False)
    tracemalloc.start()
    try:
        if sys.argv[1:] != ['--enabled']:
            raise ValueError('archive component requires explicit --enabled')
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                initial = int(line.split()[1]) * 1024
                break
        if initial is None:
            raise RuntimeError('initial RSS missing')
        if (len(os.sched_getaffinity(0)) != 1
                or resource.getrlimit(resource.RLIMIT_AS) != (16 * 1024**3,) * 2
                or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not __debug__):
            raise RuntimeError('worker requires CPU1 AS16GiB assertions CUDA-disabled')
        frozen, authentication_roots = authenticate(bootstrap)
        # ROOT must precede package imports when this file is launched by path.
        sys.path.insert(0, str(ROOT))
        sys.dont_write_bytecode = True
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence
        from experiments.neural_hz_20260831.definition_first_20260928.d037_descendant_census_20260930 import amplitude
        budget = amplitude.WorkBudget(enabled=True)
        budget.charge(EVIDENCE_CAP + RESERVE + bootstrap.used)
        budget.limit = min(CAP, numerical_start + MODEL_CAP)
        meter = evidence.Meter(limit=EVIDENCE_CAP)
        raw, _ = read_bytes(ARCHIVE, budget, maximum=ARCHIVE_BYTES, expected=ARCHIVE_SHA)
        if len(raw) != ARCHIVE_BYTES:
            raise ValueError('complete archived input byte count differs')
        data = parse_json(raw, budget)
        decoded = decode(data, frozen, budget)
        box, forms, bounds, phases, receivers, windows = decoded
        rows = []
        stats = dict(expected_rows=320, completed_rows=0, canonical_edges=0,
            valid_edges=0, padding_edges=0, potential_edges=0, potential_rows=0, potential_nnz=0,
            row_unresolved_counts=[0, 0, 0, 0], zero_coeff_slots=0, interval_cross_zero_slots=0)
        result = None
        for window in windows:
            position = window['position']
            emit(dict(event='window_started', position=position, completed_rows=len(rows),
                      whole_work_used=budget.used))
            for channel, (weights, bias) in enumerate(receivers):
                budget.charge(128)
                result = amplitude.compile_receiver(weights, bias, window['slot_bounds'],
                    window['real_slots'], enabled=True, budget=budget)
                row_summary = check_result(result, window, window['old_rows'][channel], budget)
                for key, value in row_summary.items():
                    if key == 'row_unresolved_counts':
                        for index in range(4):
                            stats[key][index] += value[index]
                    else:
                        stats[key] += value
                rows.append(dict(branch=0, position=position, channel=channel,
                    input_window_ref=(0, *position), receiver_coefficients_ref=(0, channel),
                    old_ordinary_bounds_ref=(0, *position, channel, 'ordinary_bounds'),
                    ordinary_bounds_match=True, formula_ref=FORMULA, row_masks=result['row_masks'],
                    summary=row_summary))
                stats['completed_rows'] = report['completed_rows'] = len(rows)
                if (channel + 1) % 16 == 0:
                    emit(dict(event='rows_completed', completed_rows=len(rows),
                              whole_work_used=budget.used))
        if (len(rows) != 320 or stats['canonical_edges'] != 184_320
                or stats['valid_edges'] != 102_400 or stats['padding_edges'] != 81_920):
            raise ValueError('complete archived edge population differs')
        payload = dict(schema='d037_descendant_census_v1', archive_sha256=ARCHIVE_SHA,
            archive_path=str(ARCHIVE), kernel_sha256=frozen['source_sha256'][str(HERE / 'amplitude.py')],
            frame_identity=data['result']['frame_identity'], source=data['source'],
            rule='d036_interval_lowercuts_all_canonical_slots', formula_ref=FORMULA,
            expected_rows=320, completed_rows=320, rows=rows, summary=stats,
            scope='complete archival mathematical classification; not native HZ qualification',
            classification='UNRESOLVED versus proven redundant under q<=Q*alpha; not nonredundancy',
            binding_mathematical_only=True, binding_math_verified=True,
            actual_phase_column_binding_verified=False, original_bits_deleted=0, formal_gain=0)
        # All raw/decoded inputs, all masks, manifests and the FINAL FULL kernel
        # result remain roots.  Earlier per-row Fraction results are temporary;
        # their lossless reconstruction uses authenticated archive + kernel.
        physical = evidence.bounded_ledger((raw, data, decoded, payload, result,
            authentication_roots, bootstrap.__dict__, budget.__dict__, meter.__dict__, report), meter)
        # Complete bounded receiver/decode numeric transients, not a tensor-only
        # count.  Process peaks also include imports, parsing and ledger stacks.
        entries = physical['retained_entries'] + 64 * 576 + 4096
        if entries > ENTRY_CAP:
            raise ValueError('retained plus transient numeric-entry cap exceeded')
        partial, destination = RUN / 'complete.json.partial', RUN / 'complete.json'
        if destination.exists():
            raise ValueError('complete archive output already exists')
        written = evidence.write_evidence(partial, payload, meter, {})
        partial.rename(destination)
        report.update(archive_completed=True, binding_math_verified=True,
            evidence_file='complete.json', evidence_sha256=written['sha256'],
            evidence_bytes=written['bytes'], held_ledger=physical, summary=stats)
        report.update({key: stats[key] for key in ('canonical_edges', 'valid_edges', 'padding_edges',
            'potential_edges', 'potential_rows', 'potential_nnz')})
    except Exception as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        _, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        growth = (max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial)
                  if initial is not None else 0)
        wall = time.monotonic() - start
        whole = budget.used if budget is not None else EVIDENCE_CAP + RESERVE + bootstrap.used
        branch = whole - numerical_start
        report.update(wall_s=wall, whole_work_used=whole, branch_work_used=branch,
            bootstrap_work_used=bootstrap.used, evidence_work_used=meter.used if meter is not None else 0,
            retained_entries=entries, rss_highwater_growth_bytes=growth, traced_peak_bytes=peak,
            tracer_metadata_bytes=metadata, initial_rss_bytes=initial,
            summary_reserve_bytes=RESERVE, final_summary_reserve_bytes=RESERVE)
        report['memory_gate_passed'] = (report['archive_completed'] and 'failure' not in report
            and initial is not None and growth + RESERVE <= MEMORY_CAP
            and peak + metadata + RESERVE <= MEMORY_CAP and wall <= 240
            and whole <= CAP and branch <= MODEL_CAP and entries <= ENTRY_CAP
            and report['evidence_work_used'] <= EVIDENCE_CAP)
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
