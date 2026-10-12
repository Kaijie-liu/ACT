"""Read-only, stdlib capacity arithmetic; no source lowering or bound query.

This is a conditional cost model of the documented factor-scale-only proposal,
not a lower bound on every possible verifier. It hashes the reviewed code and
uses the already archived source identity. No raw tensors or forest are opened.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANCHORS = {
    'scoped_source/tensor_row_controls.py': '262bd2d3511ce5e10641a542d99c9b57ceb34d18285a8f14568deb820066abd2',
    'scoped_source/source_dag.py': '2a65d7191f22c38cfe177d92b054811401a48de774e481870dbd0b74a93fe18c',
    'scoped_source/check_source_dag.py': 'de57076bcade7d428ffa075f34ad8923f2a004ddc1b3893a58eb8e0ce7cf44e7',
    'scoped_source/affine_references.py': '0f0d10033640210e6ce79642fa340a61c6e4293cb76779e4851a1f0f5cf582d8',
    'scoped_source/check_affine_references.py': 'a0d688ee82bf88c42945edfd3abc9591a46998c01cf236cbf752ad5b549e9c62',
    'scoped_source/forest_endpoints.py': '86396e9c662ba8c91f6ac3921dc7f410d9b977937df38a9fb782f82991c50a4c',
    'act/back_end/moe/forest_endpoints.py': 'd0ca9c0fd9f159397b5f668e7081b7c1fe4aaef5a0bc05fc8c5a14e21e014334',
    'scoped_source/hz_request_store.py': '7cb2be7647039f59d3c896b7856f634f43be9c532d9d2fd43632be09ab3230e0',
    'docs/hz_source_dag_20261013.json': 'ed61995ca9719f223fe642b58a1188d4fcdf5e8e80f13cd1bdbe4b9dae2953f3',
}
SOURCE = '0745027b58f00c608a9db921204cb2feae58273600fee658d4c9ccc3d8c19dbc'


def analyze():
    for name, expected in ANCHORS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('reviewed-code/archive identity changed: '+name)
    archive = json.loads((ROOT/'docs/hz_source_dag_20261013.json').read_bytes())
    assert archive['source_anchors']['dense'] == SOURCE
    dense = next(c for c in archive['cases'] if c['source'] == SOURCE)
    assert dense['check']['cut'] == ['expert0', 2]
    assert sum(dense['work'].values()) == 7202670

    # All 31 residues occur per complete period. Only THREE tail numerators
    # remain; their range 3..93 gives a rigorous row-independent norm bracket.
    width, hidden = 3072, 256
    periods, tail = divmod(width, 31)
    assert (periods, tail) == (99, 3)
    radius = F(1, 255)
    rounded = float(radius)
    if F(rounded) < radius:
        rounded = math.nextafter(rounded, math.inf)
    scale = F(rounded)
    low = scale * F(periods*sum(range(1, 32))+tail, 32)
    high = scale * F(periods*sum(range(1, 32))+31*tail, 32)
    # Centers (i-2)/16 are exactly binary64. No center error in this source.
    assert F(98-2, 16) < low <= high < F(99-2, 16)
    assert F(-2, 16) > -low
    active, unstable = hidden-99, 99
    # Router rows have opposite signs at each coordinate. Their difference
    # norm is at least width*(2/32)*scale, exceeding the 1/16 center gap.
    assert width*F(2, 32)*scale > F(1, 16)
    # Expert output rows also have opposite signs: nonzero margin coefficients.
    # Constraints reference only unstable preactivation rows. The active
    # ancestors (t=1), and expert1 output roots (t=0), cannot be dual-cancelled
    # at the node level in this fixed graph. This is NOT true for arbitrary DAGs.
    first_weights = (2+hidden+2)*width
    forced_entries = active*width+2*width
    unstable_entries = unstable*width
    source_floor = 5*first_weights
    three_pass_floor = 3*3*forced_entries
    next_major = source_floor+first_weights+3*forced_entries+5*unstable_entries
    return {
        'schema': 'FACTOR_SCALE_STATIC_CAPACITY_ANALYSIS_V1',
        'analyzed_head': '1b171c97fc5b88eab148d618f34a4ddf5b852f46',
        'source': SOURCE,
        'code_and_archive': ANCHORS,
        'scope': 'conditional static accounting; no generated source or evaluated candidate',
        'source_assumptions': 'unscaled first-weight local bank; independent producer/checker weighted norms; same V2 ReLU relation structure',
        'consumer_assumptions': 'current scalar pullback: one coefficient visit plus two exact calls; proposed, zero and final selected passes',
        'counts_are': 'current Work policy calls, not FLOPs, elapsed time or a verifier-wide complexity lower bound',
        'input_scale': str(scale),
        'first_row_radius_bracket': [str(low), str(high)],
        'active_rows': active,
        'unstable_rows': unstable,
        'legal_pairs': 1,
        'output_properties': 1,
        'gate_interval': ['0', '1'],
        'required_endpoints': 2,
        'first_layer_weights': first_weights,
        'source_first_bank_count_floor': source_floor,
        'forced_endpoint_entries_per_pass': forced_entries,
        'three_exact_pass_count_floor': three_pass_floor,
        'partial_count_floor_without_native': source_floor+three_pass_floor,
        'native_first_bank_count_floor': first_weights,
        'partial_count_floor_with_native': source_floor+three_pass_floor+first_weights,
        'work_cap_unchanged': 8000000,
        'decision': 'FACTOR_SCALE_ALONE_NOT_ADMITTED',
        'not_counted': ['metadata and tensor intake', 'later layers and references',
                        'input and auxiliary rows', 'gate planning', 'weighted residual norms',
                        'candidate iteration', 'serialization and terminal handling'],
        'unimplemented_check_once_batch_candidate': {
            'source_and_native_first_banks': source_floor+first_weights,
            'objective_pullbacks_once': 3*forced_entries,
            'two_dual_columns_unstable_bank': 5*unstable_entries,
            'major_terms_only': next_major,
            'remaining_for_all_omitted_work': 8000000-next_major,
            'complete_cost_upper_bound_established': False,
            'execution_admitted': False,
        },
        'new_source_runs': 0, 'new_bound_queries': 0, 'gpu_calls': 0,
        'complete_new_output_certificates': 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args()
    result = analyze()
    assert result['partial_count_floor_without_native'] > result['work_cap_unchanged']
    if args.check_report:
        if json.loads(args.check_report.read_bytes()) != result:
            raise ValueError('static report differs from the reviewed arithmetic')
        print('PASS: static report and reviewed identities; no numerical experiment')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
