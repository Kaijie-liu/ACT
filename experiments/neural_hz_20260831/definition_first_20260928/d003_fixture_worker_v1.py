"""Complete small-fixture exact evidence and memory diagnostic; not ACT admission."""
from dataclasses import fields, is_dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE.parent / 'results/d003_reference_semantics_20260928_v1'
sys.path.insert(0, str(ROOT))
SUMMARY_RESERVE = 65536


def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('RSS unavailable')


def children(value):
    """Complete reachable instance state of this reference schema, not modules."""
    if type(value) is Fraction:
        return (value.numerator, value.denominator)
    if type(value) in (tuple, list):
        return value
    if type(value) is dict:
        return tuple(item for pair in value.items() for item in pair)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(getattr(value, field.name) for field in fields(value))
    if type(value).__name__ == 'Builder' and type(value).__module__.endswith('.d003_domain_v1'):
        return (value.__dict__,)
    if value is None or type(value) in (int, bool, str, bytes, float, object):
        return ()
    raise TypeError('unaccounted held instance: ' + str(type(value)))


def ledger(root):
    seen, sizes = set(), {}
    scalar_occurrences = 0
    pending = [root]
    while pending:
        value = pending.pop()
        if type(value) in (int, bool, float, Fraction):
            scalar_occurrences += 1
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        key = type(value).__name__
        group = sizes.setdefault(key, dict(objects=0, bytes=0))
        group['objects'] += 1
        group['bytes'] += sys.getsizeof(value)
        pending.extend(children(value))
    return dict(by_type=sizes, unique_objects=len(seen),
                held_instance_bytes=sum(group['bytes'] for group in sizes.values()),
                numeric_occurrences=scalar_occurrences,
                module_and_class_singletons_not_part_of_instance_ledger=True)


def encoded(value):
    if type(value) is Fraction:
        return dict(numerator=value.numerator, denominator=value.denominator)
    if type(value) in (tuple, list):
        return [encoded(item) for item in value]
    if type(value) is dict:
        return {str(key): encoded(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return dict(record_type=type(value).__name__,
                    fields={f.name: encoded(getattr(value, f.name)) for f in fields(value)})
    if type(value).__name__ == 'Builder' and type(value).__module__.endswith('.d003_domain_v1'):
        return dict(record_type='Builder', fields=encoded(value.__dict__))
    if type(value) is object:
        return dict(frame_identity=str(id(value)))
    if value is None or type(value) in (int, bool, str, float):
        return value
    raise TypeError('unencoded evidence: ' + str(type(value)))


def main():
    start, initial_rss = time.monotonic(), rss()
    if len(os.sched_getaffinity(0)) != 1 or resource.getrlimit(resource.RLIMIT_AS) != (16 * 1024**3,) * 2:
        raise RuntimeError('worker requires frozen CPU1/AS16GiB controls')
    tracemalloc.start()
    from experiments.neural_hz_20260831.definition_first_20260928.test_d003_domain_v1 import (
        _mixed_fixture, _direct_mixed, _direct_bits, _direct_feasible,
        _independent_native, _independent_accepts, _apply_form)
    builder, element = _mixed_fixture()
    native, independent = element.lower(), _independent_native()
    checks = dict(counts=native.counts == independent['counts'],
        all_node_forms=tuple((f.terms, f.constant) for f in native.node_forms) == independent['forms'],
        all_output_forms=tuple((f.terms, f.constant) for f in native.output_forms) == independent['output_forms'],
        all_rows=tuple((r.terms, r.rhs, r.relation) for r in native.rows) == independent['rows'],
        continuous_bounds=native.continuous_bounds == independent['continuous_bounds'],
        binary_ids=native.binary_ids == independent['binary_ids'],
        binary_columns=native.binary_columns == independent['binary_columns'],
        gates=tuple(tuple(getattr(g, f.name) for f in fields(g)) for g in native.gates) == independent['gates'])
    # Four ordinary points, plus the other legal zero phase at the same point.
    fixtures = (((0, 0), 0, 0), ((0, 0), 0, 1),
                ((Fraction(1, 2), Fraction(1, 2)), 1, 0),
                ((Fraction(-1, 2), Fraction(1, 2)), 0, 0), ((1, -1), 0, 0))
    points = []
    for inputs, free_bit, zero_phase in fixtures:
        ordinary = _direct_mixed(inputs, free_bit)
        bits = _direct_bits(ordinary, free_bit, zero_phase)
        evaluation = element.evaluate(inputs, bits)
        assignment = native.assignment(evaluation, bits)
        accept = _independent_accepts(independent, assignment)
        point_checks = dict(all_nodes=evaluation.node_values == ordinary,
            all_reconstructed_nodes=tuple(_apply_form(f, assignment) for f in independent['forms']) == ordinary,
            all_outputs=evaluation.outputs == tuple(ordinary[i] for i in independent['outputs']),
            input_inverse=assignment[:2] == evaluation.inputs == tuple(map(Fraction, inputs)),
            full_feasibility=evaluation.feasible == native.satisfies(assignment) == accept == _direct_feasible(ordinary))
        points.append(dict(inputs=inputs, free_bit=free_bit, zero_phase=zero_phase,
            ordinary_nodes=ordinary, bits=bits, evaluation=evaluation,
            assignment=assignment, independent_accepts=accept, checks=point_checks))
    complete_root = dict(builder=builder, element=element, native=native,
                         independent=independent, fixtures=fixtures, points=points, checks=checks)
    payload = encoded(complete_root)
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with (RUN / 'complete_reference_evidence.json').open('xb') as stream:
        stream.write(raw)
    held = ledger((complete_root, payload, raw))
    ledger_raw = (json.dumps(held, sort_keys=True, indent=2) + '\n').encode()
    with (RUN / 'held_instance_ledger.json').open('xb') as stream:
        stream.write(ledger_raw)
    current, peak = tracemalloc.get_traced_memory()
    metadata = tracemalloc.get_tracemalloc_memory()
    rss_growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial_rss)
    wall = time.monotonic() - start
    passed = (all(checks.values()) and all(all(p['checks'].values()) for p in points)
              and held['numeric_occurrences'] <= 64000000 and wall <= 240
              and rss_growth + SUMMARY_RESERVE <= 1024**3
              and peak + metadata + SUMMARY_RESERVE <= 1024**3)
    result = dict(prototype_fixture_memory_diagnostic_passed=passed,
        complete_source_qualification=False, native_HZ_admitted=False, formal_gain=0,
        new_benchmark_solves=0, diagnostic_solver_calls=0, ordinary_points=4,
        complete_input_phase_assignments=len(points), independent_checks=checks,
        all_point_checks_passed=all(all(p['checks'].values()) for p in points),
        counts=native.counts, ledger=held, actual_cpu_affinity=list(os.sched_getaffinity(0)),
        address_space_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
        initial_rss_bytes=initial_rss, rss_highwater_growth_bytes=rss_growth,
        traced_current_bytes=current, traced_peak_bytes=peak, tracer_metadata_bytes=metadata,
        final_summary_reserve_bytes=SUMMARY_RESERVE, wall_s=wall,
        evidence_bytes=len(raw), evidence_sha256=hashlib.sha256(raw).hexdigest(),
        measurement_scope='import+complete_fixture+independent_evidence+full_encoding+held_ledger',
        instance_ledger_excludes_module_class_singletons=True,
        module_import_allocations_in_transient_trace=True)
    report = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    if len(report) > SUMMARY_RESERVE:
        raise ValueError('final summary exceeds retained reserve')
    with (RUN / 'diagnostic.json').open('xb') as stream:
        stream.write(report)
    tracemalloc.stop()
    print(report.decode(), end='')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
