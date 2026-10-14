"""Static elimination test for two consumer-only ideas; no consumer execution.

The free-access counterfactual is deliberately more optimistic than a proposed
cursor. It retains all three logical factor passes and exact check points.
Zero-role sizes describe a new format draft, not the implemented frame format.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = {
    'scoped_source/scaled_references.py': '534761cad44cbdf62ed31ee4d999fce0a5bd51aed4461b98d8f986b75d50d75d',
    'scoped_source/check_scaled_references.py': '244dc7ee5cf7fe46ab7097b2494dfec64d818e926b4726eb19d59a53c5074f9e',
    'scoped_source/check_source_dag.py': '58e5e2158843631c9e6ed324cfeeb2eb38e51df1205d2e14d1ed8ac8ce634458',
    'scoped_source/hz_request_store.py': '7cb2be7647039f59d3c896b7856f634f43be9c532d9d2fd43632be09ab3230e0',
    'scoped_source/direct_candidates.py': 'cb9b791b9a1b828c9d4fd6b351361c6fe59072b40caf0c91e55beefeaf2665a4',
    'scoped_source/pair_workspace.py': '18df56b27f952189c52e268450c0e7dd53e58da5e4f2060a4eb098b0278a2829',
    'scoped_source/scaled_endpoints.py': '48cf0be6fd9fe1776089f982a69d4100fdbd5867e207579286422bfbc45ad044',
    'scoped_source/bounded_runtime.py': 'c9c1d295e6a23bb92042f1060f99a5ee14e2e448d46e3f35af739663e750f7af',
    'act/back_end/moe/scaled_forest_endpoints.py': '118e88e3484ff4f4977f3b2535fa3aba8bdf7b1b1b7598da8232c0fe0ea57af1',
    'docs/hz_pair_scoped_capacity_20261015.json': '401820ca62f29977119746010034791ffad83bc4833efb54cd0c159cc4a3e099',
    'docs/hz_full_resource_ledger_20261014.json': '04c51e4765ada9445dcdd80f63f745459366154037443a68dd565c093acbe9d2',
}


def positive(*values):
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError('positive integer dimension')


def body_floor(inputs, endpoints):
    positive(inputs, endpoints)
    # Merge and both norms: each visits a factor and checks an exact sum.
    # All physical accesses, extra scoped setter checks and final group sums
    # are given away. This is NOT a proposed Work counter implementation.
    return 6 * inputs * endpoints


def zero_format(rows, scalar, pairs=28):
    positive(rows, scalar, pairs)
    if scalar not in (8, 1040):
        raise ValueError('native or general lossless scalar profile')
    widths = (8, 8, 2)
    stride = 72 + 2 * 1040 + scalar * rows
    frames = [128 + width * stride for width in widths]
    return dict(profile=('native' if scalar == 8 else 'general'),
                pair_rows=rows, scalar_bytes=scalar, stride=stride,
                frame_upper=max(frames), stream_upper=pairs * sum(frames),
                proposed_role='complete dense ordered values',
                zero_role='explicit full-roster symbolic zero descriptor',
                frame_upper_fits=max(frames) <= 4 * 2**20,
                stream_upper_fits=pairs * sum(frames) <= 16 * 2**20 - 8192,
                actual_bytes=None, implemented=False)


def analyze():
    for name, expected in ANCHORS.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('consumer option source drift: ' + name)
    v = json.loads((ROOT / 'docs/hz_full_resource_ledger_20261014.json').read_bytes())['inventory']
    n, lo, hi = v['inputs'], v['endpoints_min'], v['endpoints_upper']
    if (n, lo, hi, v['pairs'], v['pair_dual_rows_upper']) != (3072, 252, 504, 28, 2700):
        raise ValueError('registered conservative inventory changed')
    source_floor = 2 * v['first_weight_positions']
    return dict(
        schema='CONSUMER_ONLY_ELIMINATION_STATIC_V1', date='2026-10-15',
        based_on='17c982185a4b24b7788036069ca340585adbaa9f', anchors=ANCHORS,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        free_access=dict(
            assumptions=['unchanged merge and two norm passes',
                         'unchanged logical coefficient visits and exact sums',
                         'all endpoint candidate pairs present',
                         'all storage/mapping/native/pullback costs granted free'],
            logical_visits_per_factor=3, exact_sums_per_factor=3,
            consumer_lower_at_min_endpoints=body_floor(n, lo),
            consumer_lower_at_max_endpoints=body_floor(n, hi),
            full_max_endpoint_envelope_exceeds_8M=body_floor(n, hi) > 8000000,
            minimum_endpoint_consumer_alone_refuses=False,
            lower_with_unchanged_source_at_min_endpoints=source_floor + body_floor(n, lo),
            algorithm_universal_lower_bound=False, actual_endpoint_count=None),
        source_unchanged=dict(first_weight_visits_lower=source_floor,
                              root_memberships_lower=v['mandatory_roots'],
                              root_cap=65536, work_cap=8000000),
        zero_format_draft={name:zero_format(v['pair_dual_rows_upper'], scalar)
                           for name, scalar in (('native', 8), ('general', 1040))},
        unchanged=dict(candidate_width=8, check_width=2, seconds=300,
                       stored_bits=4096, working_bits=8192,
                       threshold='1/10000000', duties=252, max_endpoints=504,
                       max_candidate_roles=1008, work=8000000),
        decision='DO_NOT_IMPLEMENT_CONSUMER_ONLY_OPTIONS_AS_FULL_ADMISSION_FIX',
        representation_draft_only=True, production_changed=False,
        complete_cost_closed=False, full_work_upper=None, full_rss_upper=None,
        new_quota=None, execution_admitted=False,
        source_builds=0, model_runs=0, solver_calls=0, gpu_runs=0,
        next='Joint source/identity/resource contract before another isolated consumer implementation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args()
    report = analyze()
    if args.check_report:
        if json.loads(args.check_report.read_bytes()) != report:
            raise ValueError('consumer option report drift')
        print('PASS: static option elimination; no implementation or execution admission')
    else:
        print(json.dumps(report, sort_keys=True, indent=2))
