"""One-use, default-off audit of ALL frozen source forms and canonical pairs.

No project module, candidate, model or solver is imported. This diagnostic
checks archived bounds, not extraction correctness or native phase columns.
"""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP, ROOT = HERE.parent.parent, HERE.parent.parent.parent.parent
RUN = EXP / 'results/d067_fixed_pair_applicability_20261001_v1'
FREEZE = HERE / 'freeze.json'
ARCHIVE = EXP / 'results/d025_interval_capacity_20260930_v1/complete_0.json'
MANIFEST = EXP / 'results/d025_interval_capacity_20260930_v1/preregistered.json'
ARCHIVE_BYTES = 12_062_002
ARCHIVE_SHA = 'fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0'
INPUTS = {
    str(ARCHIVE): ARCHIVE_SHA,
    str(MANIFEST): 'c23b0541c696375224a6aae6b530ecce9609cece8806b16d53e1bda517860a4a',
    str(HERE.parent / 'd025_interval_capacity_20260930/census.py'): '21e2c45664888f7b4059b0bc35a6c64e07b18b9096b9d8f5b36c65f99c501fbc',
    str(HERE.parent / 'd015_source_shielding_20260928/shield_kernel_v1.py'): 'f7f91ea83a62087d32966cea89927119b4b9f71448ef99823e6527635a486ba6',
    str(HERE.parent / 'd066_wide_phase_interface_20261001/phase_interface.py'): 'f3747300524ec4856c937e405c66ed37ef99b1c4b6ef4fc7f13ed3dc534d874d',
}
NEW_FILES = ('PREREG.md', 'THEORY.md', 'audit.py')
PROVENANCE = dict(branch='redu-hz', commit='f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac')
POSITIONS = ((0, 0), (0, 31), (16, 16), (31, 0), (31, 31))
STATES = ('strict_active', 'strict_inactive', 'crossing', 'touch_lower_zero', 'touch_upper_zero', 'exact_zero')
KINDS = ('padding_both', 'padding_one', 'both_fixed', 'one_unfixed', 'both_unfixed')
CAP, BRANCH_CAP, EVIDENCE_CAP, RESERVE = 256_000_000, 200_000_000, 40_000_000, 65536
AS_CAP, MEMORY_CAP, ENTRY_CAP = 16 * 1024**3, 1024**3, 64_000_000


class Budget:
    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def charge(self, amount):
        if type(amount) is not int or amount < 0 or amount > self.limit - self.used:
            raise ValueError('work budget exhausted before operation')
        self.used += amount


class DigestMismatch(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_bound(path, maximum, budget, expected=None):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'ordinary authenticated file required')
    size = path.stat().st_size
    require(0 < size <= maximum, 'authenticated file exceeds bound')
    budget.charge(4097 + size)
    with path.open('rb') as stream:
        raw = stream.read(size + 1)
    require(len(raw) == size, 'file changed while reading')
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise DigestMismatch('source/input hash differs: ' + str(path))
    return raw, digest


def parse(raw, budget):
    budget.charge(4096 + len(raw))  # Parser byte work, separate from read/hash.
    def integer(text):
        budget.charge(3 + len(text))
        require(len(text) <= 160, 'oversized JSON integer')
        value = int(text)
        require(abs(value).bit_length() <= 512, 'JSON integer exceeds 512 bits')
        return value
    def real(text):
        budget.charge(3 + len(text))
        value = float(text)
        require(math.isfinite(value), 'nonfinite JSON float')
        return value
    def pairs(items):
        budget.charge(5 + 5 * len(items))
        answer = {}
        for key, value in items:
            require(key not in answer, 'duplicate JSON key')
            answer[key] = value
        return answer
    def constant(_):
        raise ValueError('nonstandard JSON constant')
    return json.loads(raw, parse_int=integer, parse_float=real,
                      parse_constant=constant, object_pairs_hook=pairs)


def checked(value, budget):
    budget.charge(3)
    require(type(value) is Fraction and max(abs(value.numerator).bit_length(),
            value.denominator.bit_length()) <= 512, 'rational exceeds 512 bits')
    return value


def fraction(value, budget):
    budget.charge(10)
    require(type(value) is list and len(value) == 2
            and all(type(x) is int and abs(x).bit_length() <= 512 for x in value)
            and value[1] > 0, 'invalid archived Fraction')
    return checked(Fraction(*value), budget)


def interval(value, budget):
    budget.charge(5)
    require(type(value) is list and len(value) == 2, 'invalid archived interval')
    lower, upper = (fraction(x, budget) for x in value)
    require(lower <= upper, 'reversed interval')
    return lower, upper


def state(bound):
    lower, upper = bound
    if lower > 0:
        return 'strict_active'
    if upper < 0:
        return 'strict_inactive'
    if lower < 0 < upper:
        return 'crossing'
    if lower == upper == 0:
        return 'exact_zero'
    return 'touch_lower_zero' if lower == 0 else 'touch_upper_zero'


def provenance(budget):
    budget.charge(8192)
    commands = (('branch', ['rev-parse', '--abbrev-ref', 'HEAD']),
                ('commit', ['rev-parse', 'HEAD']))
    return {key: subprocess.check_output(['git', '-C', str(ROOT), *args],
            text=True, timeout=5).strip() for key, args in commands}


def population(data, original, budget):
    budget.charge(4096)
    require(type(data) is dict and set(data) == {'box', 'model_raw', 'packet', 'result', 'source', 'spec_raw'},
            'complete source archive schema differs')
    packet, result, source = data['packet'], data['result'], data['source']
    require(source == original['selected_sources'][0]
            and data['model_raw']['sha256'] == source['model_sha256']
            and data['spec_raw']['sha256'] == source['spec_sha256']
            and packet['raw_model_sha256'] == source['model_sha256']
            and packet['schema'] == 'd015_raw_first_bank_v1'
            and packet['input_name'] == 'modelInput'
            and result['frame_identity'] == [source['model_sha256'], 'modelInput']
            and result['first_shape'] == [1, 64, 32, 32]
            and packet['first_relu']['output'] == '127', 'original frame/source binding differs')
    require(len(packet['branches']) == 1 and set(result['receivers']) == {'0'}, 'branch population differs')
    conv = packet['branches'][0]['conv']
    require(conv['weight_shape'] == [64, 64, 3, 3] and conv['group'] == 1
            and conv['strides'] == conv['dilations'] == [1, 1] and conv['pads'] == [1, 1, 1, 1]
            and conv['input_shape'] == conv['output_shape'] == [1, 64, 32, 32], 'canonical Conv differs')
    budget.charge(64 * 16 + 3072 * 8)
    require(len(result['receivers']['0']) == 64 and len(data['box']) == 3072, 'box/consumer population differs')
    for channel, receiver in enumerate(result['receivers']['0']):
        require(receiver['channel'] == channel and len(receiver['weights']) == 576, 'receiver identity differs')
    require(set(data['box']) == {str(i) for i in range(3072)}, 'full original input coordinates required')
    box = {i: interval(data['box'][str(i)], budget) for i in range(3072)}
    records, source_ids, bound_map, phase_map = [], {}, {}, {}
    counts, term_count = dict.fromkeys(STATES, 0), 0
    require(len(result['source_forms']) == 1600, 'all 1600 source forms required')
    for key, record in result['source_forms'].items():
        budget.charge(128)
        require(type(key) is str and len(key) <= 40, 'invalid source key')
        match = re.fullmatch(r'\((\d+), (\d+), (\d+)\)', key)
        require(match is not None, 'noncanonical source tuple')
        coord = tuple(int(v) for v in match.groups())
        require(0 <= coord[0] < 64 and all(0 <= x < 32 for x in coord[1:])
                and coord not in source_ids and record['original_phase'] == ['127', *coord], 'source/phase identity differs')
        value = record['form']
        require(type(value) is list and len(value) == 2 and type(value[1]) is dict, 'invalid source affine form')
        lower, upper = interval(value[0], budget)
        budget.charge(8 * len(value[1]))
        for index, coefficient in value[1].items():
            budget.charge(32)
            require(type(index) is str and index.isdecimal() and len(index) <= 4
                    and str(int(index)) == index and 0 <= int(index) < 3072, 'source coordinate differs')
            coefficient, domain = interval(coefficient, budget), box[int(index)]
            products = []
            for a in coefficient:
                for b in domain:
                    budget.charge(2)
                    products.append(checked(a * b, budget))
            budget.charge(8)
            lower = checked(lower + min(products), budget)
            upper = checked(upper + max(products), budget)
            term_count += 1
        bound = (lower, upper)
        require(bound == interval(record['bounds'], budget), 'recomputed whole-box source bound differs')
        kind, ordinal = state(bound), (coord[0] * 32 + coord[1]) * 32 + coord[2]
        budget.charge(32)
        source_ids[coord], bound_map[coord], phase_map[coord] = ordinal, bound, tuple(record['original_phase'])
        counts[kind] += 1
        records.append(dict(ordinal=ordinal, coordinate=coord, phase=phase_map[coord], state=kind,
            bounds=((lower.numerator, lower.denominator), (upper.numerator, upper.denominator)),
            archive_form_key=key, bound_recomputed_equal=True))
    require(term_count == 34752, 'full affine coefficient occurrence count differs')
    pairs, seen, pair_counts = [], set(), dict.fromkeys(KINDS, 0)
    windows = result['windows']
    require(len(windows) == 5 and result['summary']['rows'] == 320, 'five windows/all receivers required')
    for window_index, (position, window) in enumerate(zip(POSITIONS, windows)):
        budget.charge(256 + 576 * 48 + 64 * 16)
        require(window['branch'] == 0 and window['position'] == list(position)
                and all(len(window[k]) == 576 for k in ('source_slots', 'source_bounds', 'original_phases'))
                and len(window['rows']) == 64 and len(window['pair_keys']) == 288,
                'complete canonical window differs')
        for channel, row in enumerate(window['rows']):
            require(row['channel'] == channel and row['receiver_coefficients_ref'] == [0, channel]
                    and row['all_slot_and_pair_premises_ref'] == [0, *position], 'receiver premise binding differs')
        slots = []
        for offset in range(576):
            channel, location = divmod(offset, 9)
            dy, dx = divmod(location, 3)
            row, col = position[0] + dy - 1, position[1] + dx - 1
            coord = (channel, row, col) if 0 <= row < 32 and 0 <= col < 32 else None
            expected_phase = phase_map[coord] if coord is not None else None
            require(window['source_slots'][offset] == (list(coord) if coord is not None else None)
                    and window['original_phases'][offset] == (list(expected_phase) if expected_phase else None),
                    'canonical slot or original phase differs')
            bound = interval(window['source_bounds'][offset], budget)
            require(bound == (bound_map[coord] if coord is not None else (Fraction(0), Fraction(0))),
                    'window source premise differs')
            slots.append(coord)
            if coord is not None:
                seen.add(coord)
        for offset in range(0, 576, 2):
            budget.charge(80)
            require(window['pair_keys'][offset // 2] == window['source_slots'][offset:offset + 2],
                    'archived global-even pair population differs')
            left, right = slots[offset:offset + 2]
            padding = int(left is None) + int(right is None)
            fixed = sum(state(bound_map[c]) in ('strict_active', 'strict_inactive') for c in (left, right) if c is not None)
            kind = ('padding_both' if padding == 2 else 'padding_one' if padding == 1
                    else 'both_fixed' if fixed == 2 else 'one_unfixed' if fixed == 1 else 'both_unfixed')
            pair_counts[kind] += 1
            ids = tuple(source_ids[c] if c is not None else None for c in (left, right))
            pairs.append(dict(window=window_index, position=position, slots=(offset, offset + 1),
                sources=ids, phase_refs=tuple(phase_map[c] if c is not None else None for c in (left, right)),
                bounds_refs=ids, classification=kind, consumer_count=64))
    require(seen == set(source_ids) and len(pairs) == 1440 and sum(pair_counts.values()) == 1440,
            'complete source/pair coverage not established')
    summary = dict(source_count=1600, source_term_occurrences=term_count, source_states=counts,
        windows=5, receiver_rows=320, pair_count=1440, pair_counts=pair_counts,
        consumer_record_count=92160, consumer_counts={k: 64 * v for k, v in pair_counts.items()},
        kernel_tables_evaluated=0, all_source_bounds_equal=True,
        all_real_pairs_have_fixed_anchor=(pair_counts['both_unfixed'] == 0))
    payload = dict(schema='d067_fixed_pair_applicability_v1', archive_sha256=ARCHIVE_SHA,
        archive_path=str(ARCHIVE), frame_identity=result['frame_identity'], source_relu_port='127',
        positions=POSITIONS, source_records=records, pair_records=pairs, summary=summary,
        scope='B1/B2 both explicitly contain the recomputed stable-bit facts; not a native LP claim',
        original_bits_deleted=0, kernel_tables_evaluated=0, formal_gain=0)
    return payload, (box, source_ids, bound_map, phase_map, seen, slots)


def ledger(roots, meter):
    seen, stack, entries, total = set(), [iter((roots,))], 0, 0
    while stack:
        meter.charge(1)
        try:
            value = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        meter.charge(8)
        kind = type(value)
        require(kind in (dict, list, tuple, set, str, bytes, int, bool, float, Fraction, type(None)), 'non-plain ledger root')
        if kind in (int, bool, float, Fraction):
            entries += 1
            require(entries <= ENTRY_CAP, 'retained entry cap exceeded')
        if id(value) in seen:
            continue
        meter.charge(8)
        seen.add(id(value))
        total += sys.getsizeof(value)
        if kind is Fraction:
            children = (value.numerator, value.denominator)
        elif kind is dict:
            children = (item for pair in value.items() for item in pair)
        elif kind in (tuple, list, set):
            children = value
        else:
            continue
        require(len(stack) < 128, 'ledger nesting bound exceeded')
        stack.append(iter(children))
    return dict(held_bytes=total, retained_entries=entries, unique_objects=len(seen))


def write_evidence(payload, meter):
    digest, byte_count = hashlib.sha256(), 0
    partial, destination = RUN / 'complete.json.partial', RUN / 'complete.json'
    require(not destination.exists(), 'complete evidence already exists')
    with partial.open('xb') as stream:
        def emit(text):
            nonlocal byte_count
            meter.charge(len(text))
            data = text.encode('ascii')
            stream.write(data)
            digest.update(data)
            byte_count += len(data)
        def visit(value, depth=0):
            meter.charge(8)
            require(depth < 128, 'JSON nesting bound exceeded')
            if type(value) in (dict, tuple, list):
                meter.charge(8 * len(value))
                mapping = type(value) is dict
                emit('{' if mapping else '[')
                for index, (key, item) in enumerate(value.items() if mapping else enumerate(value)):
                    if index:
                        emit(',')
                    if mapping:
                        require(type(key) is str, 'output keys must be strings')
                        visit(key, depth + 1)
                        emit(':')
                    visit(item, depth + 1)
                emit('}' if mapping else ']')
            else:
                require(type(value) in (str, int, bool, type(None)), 'plain bounded output scalar required')
                require(not isinstance(value, str) or len(value) <= 4096, 'output string too long')
                require(type(value) is not int or abs(value).bit_length() <= 512, 'output integer exceeds 512 bits')
                maximum = 12 * len(value) + 2 if type(value) is str else 158
                require(maximum <= meter.limit - meter.used, 'evidence exhausted before scalar encoding')
                emit(json.dumps(value, ensure_ascii=True, allow_nan=False))
        visit(payload)
        emit('\n')
    return dict(file='complete.json.partial', sha256=digest.hexdigest(), bytes=byte_count)


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required; no default audit')
    RUN.mkdir(exist_ok=False)
    started, initial_rss = time.monotonic(), None
    work, meter = Budget(min(BRANCH_CAP, CAP - EVIDENCE_CAP - RESERVE)), Budget(EVIDENCE_CAP)
    identities, checks, held, payload = {}, {}, [], None
    report = dict(audit_completed=False, memory_gate_passed=False, applicability_audit_qualified=False,
        kernel_tables_evaluated=0, formal_gain=0, candidate_admitted=False,
        actual_model_binding_qualified=False, native_nonredundancy_verified=False,
        actual_phase_column_binding_verified=False, native_HZ_admitted=False,
        source_census_completed=False, source_census_qualified=False,
        gpu_computation_completed=False, complete_physical_qualification=False)
    def timeout(_signum, _frame):
        raise TimeoutError('240-second audit limit exceeded')
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TMPDIR=str(RUN / 'tmp'))
        (RUN / 'tmp').mkdir()
        tracemalloc.start()
        initial_rss = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        require(__debug__, 'assertions required')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(240)
        raw_freeze, freeze_sha = read_bound(FREEZE, RESERVE, work)
        frozen = parse(raw_freeze, work)
        require(frozen['schema'] == 'd067_fixed_pair_audit_v1'
                and set(frozen['source_sha256']) == {str(HERE / n) for n in NEW_FILES}
                and frozen['input_sha256'] == INPUTS and frozen['provenance'] == PROVENANCE,
                'frozen three-source/five-input/provenance contract differs')
        identities = {**frozen['source_sha256'], **INPUTS, str(FREEZE): freeze_sha}
        held.extend((raw_freeze, frozen))
        for path, digest in identities.items():
            maximum = ARCHIVE_BYTES if path == str(ARCHIVE) else 8 * 1024**2
            raw, _ = read_bound(path, maximum, work, digest)
            checks[path] = digest
            if path in (str(ARCHIVE), str(MANIFEST)):
                held.append(raw)
                if path == str(ARCHIVE):
                    require(len(raw) == ARCHIVE_BYTES, 'complete archive byte count differs')
                    data = parse(raw, work)
                    held.append(data)
                else:
                    original = parse(raw, work)
                    held.append(original)
        report['provenance_before'] = provenance(work)
        require(report['provenance_before'] == PROVENANCE, 'production branch/commit differs')
        payload, numerical_roots = population(data, original, work)
        held.extend((payload, numerical_roots, raw, checks, identities, work.__dict__, meter.__dict__, report))
        physical = ledger(held, meter)
        # JSON parse pair-lists and source arithmetic can coexist transiently;
        # two occurrences per archive byte plus a small arithmetic reserve bound
        # numeric entries without pretending process peaks are tensor storage.
        transient = 2 * ARCHIVE_BYTES + 4096
        require(physical['retained_entries'] + transient <= ENTRY_CAP, 'retained plus parser transient entry cap exceeded')
        report.update(held_ledger=physical, transient_entry_reserve=transient,
            retained_entry_upper=physical['retained_entries'] + transient, summary=payload['summary'])
        report['evidence'] = write_evidence(payload, meter)
        report['audit_completed'] = True
    except BaseException as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        try:
            report['source_drift'], report['input_drift'] = [], []
            report['identity_unchecked_paths'] = []
            work.charge(64 + 16 * len(identities))
            pending = list(identities)
            for path, digest in identities.items():
                try:
                    read_bound(path, ARCHIVE_BYTES if path == str(ARCHIVE) else 8 * 1024**2, work, digest)
                except DigestMismatch:
                    report['input_drift' if path in INPUTS else 'source_drift'].append(path)
                except BaseException:
                    report['identity_unchecked_paths'] = pending
                    raise
                pending.pop(0)
            report['provenance_after'] = provenance(work)
            require(identities and not report['source_drift'] and not report['input_drift']
                    and report['provenance_after'] == PROVENANCE, 'final provenance unavailable/different')
            report['identity_checks_completed'] = True
        except BaseException as exc:
            report['identity_checks_completed'] = False
            report['final_check_failure'] = str(exc)[:4096]
            report.setdefault('failure', dict(type=type(exc).__name__, reason='final source/input/provenance checks failed'))
        signal.alarm(0)
        peak, metadata = (tracemalloc.get_traced_memory()[1], tracemalloc.get_tracemalloc_memory()) if tracemalloc.is_tracing() else (None, None)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        growth = max(0, rss - initial_rss) if initial_rss is not None else None
        wall = time.monotonic() - started
        report.update(wall_s=wall, initial_rss_bytes=initial_rss, rss_highwater_bytes=rss,
            rss_highwater_growth_bytes=growth, traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
            branch_work_used=work.used, whole_work_used=EVIDENCE_CAP + RESERVE + work.used,
            evidence_work_used=meter.used, summary_reserve_bytes=RESERVE, final_summary_reserve_bytes=RESERVE)
        report['memory_gate_passed'] = (report['audit_completed'] and 'failure' not in report
            and growth is not None and growth + RESERVE <= MEMORY_CAP
            and rss + RESERVE <= MEMORY_CAP
            and peak is not None and peak + metadata + RESERVE <= MEMORY_CAP
            and wall <= 240 and work.used <= BRANCH_CAP
            and report['whole_work_used'] <= CAP and meter.used <= EVIDENCE_CAP)
        report['applicability_audit_qualified'] = report['memory_gate_passed'] and report['identity_checks_completed']
        if report['applicability_audit_qualified']:
            try:
                require(not (RUN / 'complete.json').exists(), 'complete evidence already exists')
                (RUN / 'complete.json.partial').rename(RUN / 'complete.json')
                report['evidence']['file'] = 'complete.json'
            except BaseException as exc:
                report['applicability_audit_qualified'] = False
                report['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
        report['supervisor_exit'] = 0 if report['applicability_audit_qualified'] else 1
        encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        require(len(encoded) <= RESERVE, 'diagnostic exceeds reserved boundary')
        with (RUN / 'diagnostic.json').open('xb') as stream:
            stream.write(encoded)
        print(encoded.decode(), end='', flush=True)
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    return report['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
