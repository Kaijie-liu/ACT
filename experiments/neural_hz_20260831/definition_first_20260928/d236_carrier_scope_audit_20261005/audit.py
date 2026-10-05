"""Frozen read-only arithmetic audit of cached premises, not a verifier candidate."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d236_carrier_scope_audit_20261005_v1'
INPUT = EXP / 'results/d025_interval_capacity_20260930_v1/complete_0.json'
POSITIONS = [[0, 0], [0, 31], [16, 16], [31, 0], [31, 31]]
WORK = 0
PEAK_BITS = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def charge(count=1):
    global WORK
    WORK += count
    require(WORK <= 256_000_000, 'diagnostic counter cap exceeded')


def checked(value):
    global PEAK_BITS
    charge()
    bits = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
    PEAK_BITS = max(PEAK_BITS, bits)
    require(bits <= 512, 'rational result exceeds 512 bits')
    return value


def rational(value):
    require(type(value) is list and len(value) == 2, 'rational encoding differs')
    require(all(type(x) is int for x in value) and value[1] > 0, 'invalid rational')
    require(max(abs(x).bit_length() for x in value) <= 512, 'rational input too wide')
    return checked(Fraction(*value))


def interval(value):
    require(type(value) is list and len(value) == 2, 'interval encoding differs')
    lo, hi = map(rational, value)
    require(lo <= hi, 'reversed interval')
    return lo, hi


def pack(value):
    return [value.numerator, value.denominator]


def product(left, right):
    values = [checked(x * y) for x in left for y in right]
    return min(values), max(values)


def total(values, start):
    for value in values:
        start = checked(start + value)
    return start


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing or linked frozen file')
    charge(path.stat().st_size)
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    charge(path.stat().st_size)
    return json.loads(path.read_text())


def provenance():
    def git(*args):
        value = subprocess.check_output(['git', *args], cwd=ROOT)
        charge(len(value))
        return value
    return dict(branch=git('branch', '--show-current').decode().strip(),
                commit=git('rev-parse', 'HEAD').decode().strip(),
                tracked_diff_sha256=hashlib.sha256(git('diff', '--binary')).hexdigest())


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def timeout(_signum, _frame):
    raise TimeoutError('frozen 60 second diagnostic alarm')


def analyze(data, report):
    result = data['result']
    require(data['source']['model_relative_path'] == 'onnx/CIFAR100_resnet_large.onnx',
            'archival population differs')
    require(set(result['receivers']) == {'0'}, 'receiver branches differ')
    receivers = result['receivers']['0']
    require(len(receivers) == 64 and len(result['windows']) == 5, 'row population differs')
    parsed = []
    for channel, receiver in enumerate(receivers):
        require(receiver['channel'] == channel and len(receiver['weights']) == 576,
                'receiver channel or complete fan-in differs')
        parsed.append(([interval(w) for w in receiver['weights']], interval(receiver['bias'])))
    crossing_slots = 0
    crossing_rows = 0
    for number, window in enumerate(result['windows']):
        require(window['branch'] == 0 and window['position'] == POSITIONS[number],
                'window order differs')
        slots, phases, raw_bounds = (window[key] for key in
                                    ('source_slots', 'original_phases', 'source_bounds'))
        require(len(slots) == len(phases) == len(raw_bounds) == 576, 'slot population differs')
        bounds, owners, seen = [], [], set()
        for index, (slot, phase, raw) in enumerate(zip(slots, phases, raw_bounds)):
            bound = interval(raw)
            if slot is None:
                require(phase is None and bound == (0, 0), 'padding changed')
            else:
                require(type(slot) is list and len(slot) == 3 and
                        all(type(x) is int for x in slot), 'slot identity invalid')
                require(tuple(slot) not in seen, 'duplicate source slot')
                seen.add(tuple(slot))
                require(phase == ['127', *slot], 'original phase identity differs')
                form = result['source_forms'][str(tuple(slot))]
                require(form['bounds'] == raw and form['original_phase'] == phase,
                        'cached source binding differs')
                if bound[0] < 0 < bound[1]:
                    owners.append(index)
            bounds.append((max(Fraction(0), bound[0]), max(Fraction(0), bound[1])))
        crossing_slots += len(owners)
        require(len(window['rows']) == 64, 'window receiver population differs')
        for channel, row in enumerate(window['rows']):
            require(row['channel'] == channel and row['receiver_coefficients_ref'] == [0, channel]
                    and row['all_slot_and_pair_premises_ref'] == [0, *window['position']],
                    'row reference differs')
            weights, bias = parsed[channel]
            terms = [product(w, b) for w, b in zip(weights, bounds)]
            full = (total((t[0] for t in terms), bias[0]),
                    total((t[1] for t in terms), bias[1]))
            require(full == interval(row['ordinary_bounds']), 'complete interval replay differs')
            crossing = full[0] < 0 < full[1]
            require(type(row['outer_crossing']) is bool and row['outer_crossing'] == crossing,
                    'cached crossing classification differs')
            crossing_rows += int(crossing)
            report['replayed_rows'] += 1
            kind = ('crossing' if crossing else 'strict_negative' if full[1] < 0
                    else 'zero_upper' if full[1] == 0 else 'other')
            for owner in owners:
                # Re-sum the full complement. Do not subtract interval bounds from a row.
                upper = total((t[1] for k, t in enumerate(terms) if k != owner), bias[1])
                report['checks'].append(dict(window=window['position'], channel=channel,
                    owner_slot=owner, owner_phase=phases[owner], child_kind=kind,
                    off_upper=pack(upper), strict_bit_support=upper < 0,
                    amplitude_support=upper <= 0))
    require(crossing_slots == 10 and crossing_rows == 2 and report['replayed_rows'] == 320
            and len(report['checks']) == 640, 'complete diagnostic population differs')
    report['summary'] = {kind: {
        'checks': sum(item['child_kind'] == kind for item in report['checks']),
        'strict_bit_support': sum(item['child_kind'] == kind and item['strict_bit_support']
                                  for item in report['checks']),
        'amplitude_support': sum(item['child_kind'] == kind and item['amplitude_support']
                                 for item in report['checks'])}
        for kind in ('crossing', 'strict_negative', 'zero_upper', 'other')}
    report['crossing_parent_slots'] = crossing_slots
    report['crossing_child_rows'] = crossing_rows


def main():
    require(sys.argv[1:] == ['--enabled'], 'explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(schema='d236_cached_carrier_audit_v1', diagnostic_completed=False,
                  mathematical_component_gate_passed=False, native_binding_qualified=False,
                  new_domain_qualified=False, formal_gain=0, independent_e0_gain=0,
                  replayed_rows=0, checks=[])
    frozen = None
    try:
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        require(0 in os.sched_getaffinity(0), 'CPU 0 unavailable')
        os.sched_setaffinity(0, {0})
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(60)
        frozen = read_json(HERE / 'freeze.json')
        require(frozen['schema'] == report['schema'], 'freeze schema differs')
        require(str(Path(sys.executable).resolve()) == frozen['interpreter'], 'interpreter differs')
        for path, expected in frozen['sha256'].items():
            require(digest(path) == expected, 'frozen identity differs: ' + path)
        report['provenance_before'] = provenance()
        require(report['provenance_before'] == frozen['provenance'], 'provenance differs')
        report['freeze_sha256'] = digest(HERE / 'freeze.json')
        save('preregistered.json', frozen)
        require(INPUT.stat().st_size <= 16 * 1024**2, 'cached input too large')
        data = read_json(INPUT)
        analyze(data, report)
        report['diagnostic_completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if frozen is not None:
            try:
                report['identity_drift'] = [path for path, expected in frozen['sha256'].items()
                                            if digest(path) != expected]
                report['provenance_after'] = provenance()
                require(not report['identity_drift'] and report['provenance_after'] == frozen['provenance'],
                        'end-of-run drift')
                require(time.monotonic() - started <= 60, 'complete diagnostic deadline exceeded')
            except Exception as exc:
                report['failure'] = type(exc).__name__ + ': ' + str(exc)
                report['diagnostic_completed'] = False
        signal.alarm(0)
        report.update(elapsed_s=time.monotonic() - started, work_counter=WORK,
                      peak_rational_bits=PEAK_BITS,
                      max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save('report.json', report)
        print(json.dumps({key: report[key] for key in
              ('diagnostic_completed', 'replayed_rows', 'elapsed_s', 'work_counter', 'peak_rational_bits')},
              sort_keys=True), flush=True)
    return 0 if report['diagnostic_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
