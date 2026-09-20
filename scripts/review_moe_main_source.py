"""Separate arithmetic review of saved inputs; not a third-party SAFE audit.

Does NOT import the applicability analyzer, ACT or a solver. Float values are
decoded from IEEE bits, and each midpoint/radius operation is separately rounded
from its exact rational result. Only request.pt, never models/data, is loaded.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]


def binary(value):
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    sign, exponent, mantissa = bits >> 63, (bits >> 52) & 2047, bits & (2**52-1)
    if exponent == 2047:
        raise ValueError('nonfinite')
    if exponent:
        mantissa += 2**52
    power = (exponent-1023 if exponent else -1022)-52
    return (-1 if sign else 1) * Fraction(mantissa) * Fraction(2)**power


def run():
    import torch
    torch.set_num_threads(1)
    start = time.monotonic()
    path = ROOT/'docs/main_table_source_applicability_20260921.json'
    report = json.loads(path.read_bytes())
    edges = ('requested_rational_to_box', 'requested_binary64_to_box', 'box_to_hz_formula')
    lookup = {}
    summaries = []
    for box in report['boxes']:
        reference = box['representative_saved_request']
        p = ROOT/reference['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == reference['sha256']
        saved = torch.load(p, weights_only=True, map_location='cpu')
        for k in ('center', 'lower', 'upper'):
            t = saved[k].contiguous()
            h = hashlib.sha256(str(t.dtype).encode() + json.dumps(list(t.shape), separators=(',', ':')).encode() + t.numpy().tobytes()).hexdigest()
            assert box['identities'][k]['sha256'] == h
        counts, lower_counts, upper_counts = [0]*3, [0]*3, [0]*3
        maximum = [Fraction(0)]*3
        for values in zip(*(saved[k].reshape(-1).tolist() for k in ('center', 'lower', 'upper'))):
            if values not in lookup:
                x, lo, hi = map(binary, values)
                # Round each frozen operation independently, using exact operands.
                c = binary(float(binary(float(lo+hi))/2))
                r = binary(float(binary(float(hi-lo))/2))
                if abs(r) <= binary(1e-12): r = Fraction(0)
                intervals = []
                for eps in (Fraction(2, 255), binary(2/255)):
                    intervals.append(((lo, hi), (max(0, x-eps), min(1, x+eps))))
                intervals.append(((c-r, c+r), (lo, hi)))
                lookup[values] = tuple((max(0, outer[0]-inner[0]), max(0, inner[1]-outer[1]))
                                        for outer, inner in intervals)
            for k, (gl, gu) in enumerate(lookup[values]):
                counts[k] += bool(gl or gu); lower_counts[k] += gl > 0; upper_counts[k] += gu > 0
                maximum[k] = max(maximum[k], gl, gu)
        for k, edge in enumerate(edges):
            s = box['comparisons'][edge]
            assert (counts[k], lower_counts[k], upper_counts[k], maximum[k]) == (
                s['inward_coordinates'], s['lower_inward'], s['upper_inward'], Fraction(s['max_gap_exact']))
        summaries.append(counts)
    # Every reference is reread, not just the representative boxes.
    artifact_files = 0
    for item in report['artifact_inventory']:
        for key in ('terminal', 'manifest', 'package', 'request'):
            r = item.get(key)
            if r:
                assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() == r['sha256']
                artifact_files += 1
    for edge in edges:
        assert report['summary'][edge]['gain_requests_with_inward_coordinates'] == sum(
            not g['input_scope'][edge]['contains'] for g in report['gain_records'])
    # Scope inventory must remain consistent with individual saved gain packages.
    decision_tiers = Counter()
    for gain in report['gain_records']:
        p = ROOT/gain['evidence_refs']['package']['path']
        e = json.loads(p.read_bytes())
        assert e['identity'] == gain['identity'] and e['request_id'] == gain['request_id']
        assert e['verdict']['status'] == 'SAFE'
        assert e['route_coverage']['feasible_route_sets'] == gain['acceptance_basis']['legal_pairs_recorded']
        decision_tiers[e['verdict']['decision_tier']] += 1
        for row in gain['acceptance_basis'].get('obligations', []):
            pair = next(p for p in e['tier2']['pairs'] if p['pair'] == row['pair'])
            original = next(v for v in pair['property_rows'] if v['property_index'] == row['property_index'])
            canonical = json.dumps(original, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            assert hashlib.sha256(canonical).hexdigest() == row['record_sha256']
    receipt = {'schema': 'main_source_arithmetic_review_v1', 'status': 'PASS',
        'review_kind': 'SEPARATE_IMPLEMENTATION_SAME_AGENT_NOT_THIRD_PARTY',
        'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'boxes': len(summaries), 'distinct_coordinate_triples': len(lookup),
        'artifact_files_rehashed': artifact_files, 'gain_tiers': dict(decision_tiers),
        'no_solver_or_model_calls': True, 'seconds': time.monotonic()-start,
        'source_complete_SAFE_reproof': False}
    out = ROOT/'docs/main_table_source_applicability_20260921_review.json'
    with out.open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    run()
