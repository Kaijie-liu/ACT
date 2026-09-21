"""Append-only saved-input audit: requested real box versus frozen input formula.

No ACT/model/solver execution. Torch only decodes 100 hash-bound saved requests.
This does not recover historical HZ traces or certify downstream transformations.
"""
import argparse
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '282e0507b94ac2f6105096e43ea7c0f7a0d38bc5'
PARENT = 'docs/main_table_source_applicability_20260921.json'
OUTPUT = ROOT / 'docs/main_table_input_composition_20260921.json'
KINDS = ('rational_epsilon', 'binary64_epsilon')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pinned_bytes(path, sha):
    p = (ROOT / path).resolve()
    require(p.is_relative_to(ROOT), 'outside project reference')
    data = p.read_bytes()
    require(digest(data) == sha, 'source hash mismatch')
    return data


def endpoint_gaps(requested, represented):
    """Return coverage loss and outward excess separately, in exact arithmetic."""
    a, b = map(Q, requested)
    c, d = map(Q, represented)
    require(a <= b and c <= d, 'reversed interval')
    return (max(Q(0), c-a), max(Q(0), b-d)), (max(Q(0), a-c), max(Q(0), d-b))


@lru_cache(maxsize=None)
def coordinate(x, lo, hi, epsilon):
    require(all(math.isfinite(v) for v in (x, lo, hi, epsilon)), 'nonfinite input')
    require(0 <= x <= 1 and 0 <= lo <= hi <= 1 and epsilon >= 0, 'invalid box')
    require(lo == max(0., x-epsilon) and hi == min(1., x+epsilon), 'materialization mismatch')
    center, raw_radius = (lo+hi)*.5, (hi-lo)*.5
    radius = raw_radius if abs(raw_radius) > 1e-12 else 0.
    represented = Q(center)-Q(radius), Q(center)+Q(radius)
    return {kind: endpoint_gaps(
        (max(Q(0), Q(x)-eps), min(Q(1), Q(x)+eps)), represented)
        for kind, eps in zip(KINDS, (Q(2, 255), Q(epsilon)))}


def tensor_identity(t):
    require(str(t.dtype) == 'torch.float64' and list(t.shape) == [1, 3, 32, 32],
            'unexpected tensor type/shape')
    data = t.detach().cpu().contiguous().numpy().tobytes()
    return {'dtype': str(t.dtype), 'shape': list(t.shape), 'sha256': digest(
        str(t.dtype).encode('ascii') + json.dumps(list(t.shape), separators=(',', ':')).encode() + data)}


def box_record(box, values, epsilon):
    require(set(values) == {'center', 'lower', 'upper'}, 'missing tensor')
    require({len(v) for v in values.values()} == {box['coordinates']}, 'coordinate count mismatch')
    stats = {kind: {'inward_coordinates': 0, 'outward_coordinates': 0,
                   'max_inward_gap_exact': Q(0), 'max_outward_gap_exact': Q(0),
                   'first_inward_witness': None, 'first_outward_witness': None}
             for kind in KINDS}
    for i, (x, lo, hi) in enumerate(zip(values['center'], values['lower'], values['upper'])):
        for kind, gaps in coordinate(x, lo, hi, epsilon).items():
            for direction, pair in zip(('inward', 'outward'), gaps):
                stat = stats[kind]
                gap = max(pair)
                stat[direction+'_coordinates'] += int(gap > 0)
                key = 'max_'+direction+'_gap_exact'
                stat[key] = max(stat[key], gap)
                key = 'first_'+direction+'_witness'
                if gap and stat[key] is None:
                    stat[key] = {'flat_coordinate': i, 'center': x.hex(),
                                 'lower': lo.hex(), 'upper': hi.hex(),
                                 'lower_gap_exact': str(pair[0]), 'upper_gap_exact': str(pair[1])}
    for stat in stats.values():
        for direction in ('inward', 'outward'):
            key = 'max_'+direction+'_gap_exact'
            stat[key] = str(stat[key])
        stat['requested_subset_of_hz_formula'] = stat['inward_coordinates'] == 0
        stat['hz_formula_subset_of_requested'] = stat['outward_coordinates'] == 0
    return {k: box[k] for k in ('rank', 'dataset_index', 'coordinates', 'representative_saved_request')} | {
        'comparisons': stats}


def analyze():
    old = subprocess.check_output(['git', 'show', BASELINE+':'+PARENT], cwd=ROOT)
    parent = json.loads(pinned_bytes(PARENT, digest(old)))
    require([b['rank'] for b in parent['boxes']] == list(range(100)), 'parent roster changed')
    # Saved tensor decoding ONLY; never load checkpoint, dataset or ACT modules.
    import torch
    torch.set_num_threads(1)
    rows = []
    for box in parent['boxes']:
        ref = box['representative_saved_request']
        pinned_bytes(ref['path'], ref['sha256'])
        request = torch.load(ROOT/ref['path'], map_location='cpu', weights_only=True)
        require(set(request) == {'center', 'lower', 'upper', 'request_id'}, 'request schema mismatch')
        for key in ('center', 'lower', 'upper'):
            require(tensor_identity(request[key]) == box['identities'][key], 'tensor identity mismatch')
        values = {k: request[k].reshape(-1).tolist() for k in ('center', 'lower', 'upper')}
        rows.append(box_record(box, values, 2/255))
    summary = {}
    for kind in KINDS:
        stats = [b['comparisons'][kind] for b in rows]
        out = {}
        for direction in ('inward', 'outward'):
            counts = [v[direction+'_coordinates'] for v in stats]
            out.update({
                'inputs_with_'+direction+'_coordinates': sum(n > 0 for n in counts),
                direction+'_coordinates': sum(counts),
                direction+'_coordinates_per_input_min': min(counts),
                direction+'_coordinates_per_input_max': max(counts),
                'max_'+direction+'_gap_exact': str(max(Q(v['max_'+direction+'_gap_exact']) for v in stats))})
        affected = [r for r in parent['gain_records'] if
                    rows[r['rank']]['comparisons'][kind]['inward_coordinates']]
        out.update(gain_model_input_pairs_with_coverage_loss=len(affected),
                   distinct_gain_inputs_with_coverage_loss=len({r['rank'] for r in affected}))
        summary[kind] = out
    return {'schema': 'MAIN_TABLE_INPUT_COMPOSITION_V1', 'parent': {'path': PARENT, 'sha256': digest(old)},
            'starting_head': BASELINE, 'scope': 'Exact saved-input arithmetic; not historical intermediate traces or network reproof.',
            'historical_statuses_preserved': True, 'new_solver_calls': 0, 'new_model_forwards': 0,
            'new_source_propagations': 0, 'new_output_bounds': 0,
            'input98_followup': 'STOP_INPUT98_FOLLOWUP', 'source_complete_positive_proof': False,
            'saved_representatives_decoded': len(rows), 'coordinates_checked': sum(b['coordinates'] for b in rows),
            'all_739_package_identity_checks': 'Inherited from hash-bound parent; this addendum rereads 100 representatives.',
            'center_semantics': parent['center_semantics'],
            'requested_epsilon_rational': '2/255', 'requested_epsilon_binary64': str(Q(2/255)),
            'reconstruction_semantics': parent['reconstruction_semantics'],
            'summary': summary, 'boxes': rows,
            'parent_schema_note': 'V1 summary.coordinate_count_min/max count inward coordinates per box, NOT tensor dimensions; parent unchanged.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = analyze()
    if args.check:
        require(json.loads(OUTPUT.read_bytes()) == result, 'fresh composition reread differs')
    else:
        with OUTPUT.open('x') as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
    print(json.dumps({'fresh_reread_matches': args.check, 'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
