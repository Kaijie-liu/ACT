"""Saved-only applicability audit; never run ACT, a model, or a solver.

Torch is used ONLY to decode hash-bound request.pt tensors (weights_only).
All containment comparisons use exact binary rationals. The HZ endpoints are
an explicit reconstruction of the frozen input formula, NOT recovered traces.
This produces a new scope ledger, not replacements for historical outcomes.
"""
from collections import Counter
from fractions import Fraction as Q
from functools import lru_cache
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HEAD = 'bc0791976b00879c28c268692aecc5854b3bc091'
RAW = ROOT / 'data/moe/results/schedule_confirmation_100_full_20260912_r1'
ARCHIVE = ROOT / 'act/pipeline/moe/results/schedule_confirmation_100_review_20260914_r1.json'
SELECTION = ROOT / 'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json'
OUTPUT = ROOT / 'docs/main_table_source_applicability_20260921.json'
PATHS = (
    'act/pipeline/moe/schedule_confirmation.py',
    'act/pipeline/moe/staged_verifier.py',
    'act/pipeline/moe/route_complexity_schedule.py',
    'act/pipeline/moe/experiment1.py',
    'act/back_end/moe/factory.py',
    'act/back_end/verifier.py',
    'act/back_end/hybridz_tf/hybridz_tf.py',
    'act/back_end/solver/solver_hz.py',
)
EDGES = ('requested_rational_to_box', 'requested_binary64_to_box', 'box_to_hz_formula')


def require(value, why):
    if not value:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    p = Path(path).resolve()
    require(p.is_relative_to(ROOT), 'reference outside project')
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p.read_bytes())}


def bound_ref(path, digest):
    r = ref(path)
    require(r['sha256'] == digest, f'hash mismatch: {path}')
    return r


def git_blob(path):
    return subprocess.check_output(['git', 'show', f'{HEAD}:{path}'], cwd=ROOT)


def frozen_sources(expected):
    names = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', HEAD, 'act'], cwd=ROOT, text=True).splitlines()
    entries = {p: sha(git_blob(p)) for p in sorted(names) if p.endswith('.py')}
    digest = sha(''.join(f'{p}:{h}\n' for p, h in entries.items()).encode())
    require(digest == expected, 'runtime source digest is not frozen Git source')
    return {'git_head': HEAD, 'all_act_python_files': len(entries),
            'all_act_python_sha256': digest,
            'reviewed_call_path': {p: {'frozen_sha256': entries[p],
                'current_sha256': sha((ROOT / p).read_bytes()),
                'current_equals_frozen': sha((ROOT / p).read_bytes()) == entries[p]}
                for p in PATHS}}


def tensor_identity(t):
    require(str(t.dtype) == 'torch.float64' and list(t.shape) == [1, 3, 32, 32],
            'unexpected saved input type/shape')
    t = t.detach().cpu().contiguous()
    b = t.numpy().tobytes()
    digest = sha(str(t.dtype).encode('ascii') +
                 json.dumps(list(t.shape), separators=(',', ':')).encode() + b)
    return {'dtype': str(t.dtype), 'shape': list(t.shape), 'sha256': digest}


def inclusion(outer, inner):
    """Positive gaps disprove containment, not safety of a network."""
    require(outer[0] <= outer[1] and inner[0] <= inner[1], 'reversed box')
    return max(Q(0), outer[0] - inner[0]), max(Q(0), inner[1] - outer[1])


@lru_cache(maxsize=None)
def coordinate(x, lo, hi, epsilon):
    require(all(math.isfinite(v) for v in (x, lo, hi, epsilon)), 'nonfinite input')
    require(0 <= lo <= hi <= 1 and 0 <= x <= 1 and epsilon > 0, 'invalid input box')
    require(lo == max(0.0, x - epsilon) and hi == min(1.0, x + epsilon),
            'saved box differs from frozen binary64 materialization')
    # Scalar operations match the frozen NumPy float64 midpoint/radius formula.
    c, raw_r = (lo + hi) * .5, (hi - lo) * .5
    r = raw_r if abs(raw_r) > 1e-12 else 0.0
    box = (Q(lo), Q(hi))
    hz = (Q(c) - Q(r), Q(c) + Q(r))
    requested = lambda e: (max(Q(0), Q(x) - e), min(Q(1), Q(x) + e))
    return {'gaps': tuple(inclusion(o, i) for o, i in (
                (box, requested(Q(2, 255))), (box, requested(Q(epsilon))), (hz, box))),
            'hz': [str(hz[0]), str(hz[1])],
            'c': c, 'r': r, 'positive_radius_dropped': raw_r > 0 and r == 0}


def summarize_box(sample, values, epsilon):
    require(set(values) == {'center', 'lower', 'upper'}, 'missing box tensor')
    require(len({len(v) for v in values.values()}) == 1, 'box length mismatch')
    stats = {e: {'inward_coordinates': 0, 'lower_inward': 0, 'upper_inward': 0,
                 'max_gap_exact': Q(0), 'first_witness': None} for e in EDGES}
    dropped = 0
    for j, (x, lo, hi) in enumerate(zip(values['center'], values['lower'], values['upper'])):
        v = coordinate(x, lo, hi, epsilon)
        dropped += v['positive_radius_dropped']
        for e, (gl, gu) in zip(EDGES, v['gaps']):
            s = stats[e]
            s['inward_coordinates'] += bool(gl or gu)
            s['lower_inward'] += gl > 0
            s['upper_inward'] += gu > 0
            s['max_gap_exact'] = max(s['max_gap_exact'], gl, gu)
            if (gl or gu) and s['first_witness'] is None:
                s['first_witness'] = {'flat_coordinate': j,
                    'center': x.hex(), 'lower': lo.hex(), 'upper': hi.hex(),
                    'hz_center': v['c'].hex(), 'hz_radius': v['r'].hex(),
                    'hz_exact_endpoints': v['hz'],
                    'lower_gap_exact': str(gl), 'upper_gap_exact': str(gu)}
    for s in stats.values():
        s['max_gap_float'] = float(s['max_gap_exact'])
        s['max_gap_exact'] = str(s['max_gap_exact'])
        s['contains'] = s['inward_coordinates'] == 0
    return {'rank': sample['sample_rank'], 'dataset_index': sample['dataset_index'],
            'identities': {k: sample[k] for k in values},
            'coordinates': len(values['center']), 'positive_radii_dropped': dropped,
            'comparisons': stats}


def expected_identity(sample, model, cfg, epsilon):
    return {'model_state': model['model_state'],
            'checkpoint': {'path': model['checkpoint'], 'sha256': model['checkpoint_sha256']},
            **{k: sample[k] for k in ('center', 'lower', 'upper')},
            'epsilon': epsilon,
            'property': {'kind': 'TOP1_ROBUST', 'clean_prediction': sample['label'], 'classes': 10},
            'config_sha256': sha(canonical(cfg))}


def bind_identity(e, sample, model, cfg, epsilon):
    expected = expected_identity(sample, model, cfg, epsilon)
    require(e['identity'] == expected, 'request/selection/config binding mismatch')
    require(e['request_id'] == sha(canonical(expected)), 'request id mismatch')


def gain_basis(e):
    """Describe and validate RECORDED acceptance fields, not their source bounds."""
    coverage = e['route_coverage']
    pairs = coverage['feasible_route_sets']
    require(coverage['coverage_complete'] and coverage['route_sets_exact'] and len(pairs) > 1,
            'gain lacks recorded complete multi-route coverage')
    require(len({tuple(p) for p in pairs}) == len(pairs), 'duplicate pair')
    policy = e['numerical_safety']
    tol = policy['safe_positive_margin']
    tier = e['verdict']['decision_tier']
    require(e['verdict']['status'] == 'SAFE', 'gain not historical SAFE')
    out = {'decision_tier': tier, 'legal_pairs_recorded': pairs,
           'numerical_policy_sha256': sha(canonical(policy)),
           'historical_evidence_level': 'HZ_POLICY_ACCEPTED',
           'source_complete_positive_proof': False}
    if tier == 'TIER2_F0':
        records = e['tier2']['pairs']
        require(sorted(p['pair'] for p in records) == sorted(pairs), 'missing pair obligation')
        obligations = []
        for pair in records:
            rows = pair['property_rows']
            require(sorted(v['property_index'] for v in rows) == list(range(9)),
                    'missing/duplicate property obligation')
            for v in rows:
                require(v['status'] == 'SAFE' and math.isfinite(v['accepted_minimum'])
                        and v['accepted_minimum'] > tol, 'nonpositive accepted row')
                reused = v.get('solver_bound_kind') == 'scoped_tier1_interval'
                if reused:
                    sources = v['proof_sources']
                    require(sorted(s['expert'] for s in sources) == sorted(pair['pair']),
                            'reused expert mismatch')
                    for s in sources:
                        scope = s['scope']
                        require(scope['request_id'] == e['request_id'], 'reuse request mismatch')
                        for key in ('lower', 'upper', 'model_state', 'property'):
                            require(scope[key] == e['identity'][key], 'reuse scope mismatch')
                        require(scope['numerical_policy'] == policy, 'reuse policy mismatch')
                        require(s['property_index'] == v['property_index'], 'reuse property mismatch')
                else:
                    require(v['solver_status'] == 0, 'nonoptimal historical acceptance')
                obligations.append({'pair': pair['pair'], **{k: v.get(k) for k in
                    ('property_index', 'reason', 'accepted_minimum', 'solver_bound_kind',
                     'solver_status', 'solver_dual_objective', 'solver_primal_objective',
                     'solver_gap', 'lambda_bounds', 'difference_bounds')},
                    'reused': reused,
                    'record_sha256': sha(canonical(v))})
        out.update(obligations=obligations,
                   reused=sum(v['reused'] for v in obligations),
                   residual=sum(not v['reused'] for v in obligations),
                   minimum_recorded_acceptance=min(v['accepted_minimum'] for v in obligations))
    else:
        require(tier == 'TIER1_GATE_ELIMINATION', 'unexpected gain tier')
        branches = e['tier1']['branches']
        require(sorted(b['candidate'] for b in branches) == sorted(coverage['candidate_experts']),
                'missing candidate obligation')
        require(all(b['unknown_reason'] == 'SAFE_PROVED' for b in branches), 'unproved tier1 branch')
        out['branches'] = [{k: b.get(k) for k in ('candidate', 'solver_status', 'solver_metadata',
                'source_policy', 'unknown_reason')} | {'record_sha256': sha(canonical(b))}
                for b in branches]
        out.update(reused=None, residual=None, minimum_recorded_acceptance=None,
                   bound_note='Tier1 interval/expanded-violation infeasibility records; no scalar full-output dual saved.')
    return out


def analyze():
    started = time.monotonic()
    archive, selection, runtime = read(ARCHIVE), read(SELECTION), read(RAW / 'runtime.json')
    require(runtime['git_head'] == archive['experiment_head'] == HEAD, 'execution head mismatch')
    sources = frozen_sources(runtime['source_sha256'])
    require(archive['source_sha256'] == sources['all_act_python_sha256'], 'archive source mismatch')
    refs = {str(p): ref(p) for p in (ARCHIVE, SELECTION, RAW / 'runtime.json', RAW / 'rows.jsonl')}
    for key in ('runtime', 'audit_file'):
        p = archive['full'][key]
        bound_ref(p['path'], p['sha256'])
    bound_ref(SELECTION, runtime['config']['selection_sha256'])
    cfgs = {k: read(v['path']) for k, v in runtime['config']['methods'].items()}
    for v in runtime['config']['methods'].values():
        bound_ref(v['path'], v['sha256'])
        require(sha(git_blob(str(Path(v['path']).relative_to(ROOT)))) == v['sha256'], 'config not frozen')
    samples = selection['samples']
    require([s['sample_rank'] for s in samples] == list(range(100)), 'cohort roster mismatch')
    epsilon = selection['request']['epsilon']
    require(epsilon == 2 / 255, 'epsilon changed')
    rows = [json.loads(s) for s in (RAW / 'rows.jsonl').read_text().splitlines()]
    index = {(r['model'], r['rank'], r['method']): r for r in rows}
    expected_keys = {(m, n, a) for m in selection['models'] for n in range(100) for a in cfgs}
    require(len(rows) == 900 and set(index) == expected_keys, 'missing/duplicate terminal')
    # Only saved request tensors are decoded. No checkpoint, dataset or ACT import.
    import torch
    torch.set_num_threads(1)
    boxes, evidence, inventory, decoded = {}, {}, [], 0
    for row in rows:
        job = RAW / row['job_id']
        terminal = job / 'terminal.json'
        require(read(terminal) == row, 'terminal/ledger disagreement')
        item = {'job_id': row['job_id'], 'terminal': ref(terminal), 'package': None}
        if row['package'] is None:
            require(row['outer_timeout'] and row['status'] == 'TIMEOUT', 'unaccounted missing package')
            inventory.append(item)
            continue
        p = Path(row['package'])
        require(p == job / 'package', 'package path mismatch')
        item['manifest'] = bound_ref(p / 'manifest.json', row['manifest_sha256'])
        manifest = read(p / 'manifest.json')
        e = read(p / 'evidence.json')
        item['package'] = bound_ref(p / 'evidence.json', manifest['evidence_sha256'])
        item['request'] = bound_ref(p / 'request.pt', manifest['request']['sha256'])
        require(Path(manifest['request']['path']) == p / 'request.pt', 'saved request path mismatch')
        require(manifest['status'] == row['status'] == e['verdict']['status'], 'status mismatch')
        require(e['execution']['git_head'] == HEAD, 'package execution mismatch')
        require(e['execution']['config_sha256'] == runtime['config']['methods'][row['method']]['sha256'],
                'execution configuration mismatch')
        sample = samples[row['rank']]
        require(sample['dataset_index'] == row['dataset_index'] == e['execution']['dataset_index'], 'input index mismatch')
        bind_identity(e, sample, selection['models'][row['model']], cfgs[row['method']], epsilon)
        require(manifest['request_id'] == e['request_id'], 'manifest request id mismatch')
        req = torch.load(p / 'request.pt', weights_only=True, map_location='cpu')
        require(set(req) == {'center', 'lower', 'upper', 'request_id'}, 'request tensor roster')
        require(req['request_id'] == e['request_id'], 'tensor request id mismatch')
        for key in ('center', 'lower', 'upper'):
            require(tensor_identity(req[key]) == sample[key], 'tensor identity mismatch')
        decoded += 1
        rank = row['rank']
        if rank not in boxes:
            values = {k: req[k].reshape(-1).tolist() for k in ('center', 'lower', 'upper')}
            b = summarize_box(sample, values, epsilon)
            b['representative_saved_request'] = item['request']
            b['saved_package_copies_checked'] = 0
            boxes[rank] = b
        boxes[rank]['saved_package_copies_checked'] += 1
        evidence[row['job_id']] = e
        inventory.append(item)
    require(set(boxes) == set(range(100)), 'missing saved cohort box')
    totals = {a: dict(Counter(r['status'] for r in rows if r['method'] == a)) for a in cfgs}
    for arm, total in totals.items():
        require(total == archive['descriptive_supplement']['totals'][arm]['states'], 'historical counts changed')
    gains = [r for r in rows if r['method'] == 'adaptive' and r['status'] == 'SAFE'
             and index[r['model'], r['rank'], 'matched']['status'] != 'SAFE']
    old_gains = archive['descriptive_supplement']['safe_discordances']['matched']['gained_safe']
    require({r['job_id'] for r in gains} == {r['adaptive']['job_id'] for r in old_gains}
            and len(gains) == 23, 'gain roster mismatch')
    inv = {r['job_id']: r for r in inventory}
    gain_records = []
    for row in gains:
        e = evidence[row['job_id']]
        old = next(r['adaptive'] for r in old_gains if r['adaptive']['job_id'] == row['job_id'])
        bound_ref(old['evidence']['path'], old['evidence']['sha256'])
        require(old['dataset_index'] == row['dataset_index'], 'archived gain index mismatch')
        gain_records.append({k: row[k] for k in ('job_id', 'model', 'rank', 'dataset_index')} | {
            'request_id': e['request_id'], 'identity': e['identity'],
            'evidence_refs': inv[row['job_id']], 'acceptance_basis': gain_basis(e),
            'input_scope': boxes[row['rank']]['comparisons'],
            'upstream_source_status': 'NOT_INDEPENDENTLY_CHECKED_FOR_HISTORICAL_OUTPUTS',
            'historical_intermediate_HZ_trace': 'NOT_PRESENT_IN_STANDARD_PACKAGE'})
    summary = {}
    for edge in EDGES:
        vals = [b['comparisons'][edge] for b in boxes.values()]
        summary[edge] = {'requests_with_inward_coordinates': sum(not v['contains'] for v in vals),
            'inward_coordinates': sum(v['inward_coordinates'] for v in vals),
            'max_gap_exact': str(max(Q(v['max_gap_exact']) for v in vals)),
            'coordinate_count_min': min(v['inward_coordinates'] for v in vals),
            'coordinate_count_max': max(v['inward_coordinates'] for v in vals),
            'gain_requests_with_inward_coordinates': sum(not r['input_scope'][edge]['contains'] for r in gain_records)}
    result = {'schema': 'main_table_source_applicability_v1',
        'audit_status': 'COMPLETED_WITH_SOURCE_CONTAINMENT_GAPS',
        'scope': 'Saved evidence identity/acceptance inventory plus exact INPUT containment; NOT independent SAFE reproof or third-party review.',
        'historical_statuses_preserved': True, 'source_complete_positive_proof': False,
        'new_solver_calls': 0, 'new_model_forwards': 0, 'new_source_propagations': 0,
        'new_output_bounds': 0, 'input98_followup': 'STOP_INPUT98_FOLLOWUP',
        'requested_epsilon_rational': '2/255', 'requested_epsilon_binary64': str(Q(epsilon)),
        'center_semantics': 'exact binary value of saved center; no raw-image preprocessing equivalence checked',
        'reconstruction_semantics': 'frozen binary64 sparse input midpoint/radius formula; exact denotation, NOT saved historical HZ trace; no network/guard/output containment established',
        'frozen_sources': sources, 'historical_runtime_versions': runtime['versions'],
        'saved_artifacts': list(refs.values()), 'historical_totals': totals,
        'summary': summary, 'packages_decoded': decoded, 'terminals': len(rows),
        'missing_packages_retained': len(rows) - decoded,
        'unique_boxes_checked': len(boxes), 'coordinates_checked': sum(b['coordinates'] for b in boxes.values()),
        'positive_radii_dropped': sum(b['positive_radii_dropped'] for b in boxes.values()),
        'numerical_policy': cfgs['adaptive']['numerical_safety'],
        'boxes': [boxes[n] for n in range(100)], 'gain_records': gain_records,
        'artifact_inventory': inventory}
    return result, time.monotonic() - started


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='fresh saved-only reread; compare with new applicability ledger')
    args = p.parse_args()
    result, elapsed = analyze()
    if args.check:
        require(read(OUTPUT) == result, 'applicability ledger differs on fresh reread')
    else:
        with OUTPUT.open('x') as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
    print(json.dumps({'audit_status': result['audit_status'], 'seconds': elapsed,
          'fresh_reread_matches': args.check, 'packages': result['packages_decoded'],
          'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
