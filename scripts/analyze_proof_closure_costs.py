"""Read-only five-case closure/cost analysis. No solver or model imports."""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

from portable_proof.runtime import digest, original_bytes

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/moe/results'
ARCHIVE = ROOT / 'act/pipeline/moe/results'


def classify(required, positive, unavailable=0):
    if unavailable:
        return 'NO_COMPLETE_CHECKABLE_EVIDENCE'
    if positive < required:
        return 'CHECKED_NONPOSITIVE_BOUND'
    return 'COMPLETE_CONDITIONAL_PROOF'


def analyze():
    inputs = {}
    def read(path):
        data = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = digest(data)
        return json.loads(data)
    old = read(ARCHIVE / 'request_lp_act_only_review_20260915_r1.json')
    # Audit all recorded source hashes; do not rerun proof generation/checks.
    for name, expected in old['raw_hashes'].items():
        if digest((RAW / 'request_lp_act_only_20260915_r1' / name).read_bytes()) != expected:
            raise ValueError('ACT-only frozen evidence drift: ' + name)
    rows = []
    for c in old['cases']:
        model, index = c['model'], c['dataset_index']
        directory = RAW / 'request_lp_act_only_20260915_r1' / f'{model}_{index}'
        m = read(directory / 'manifest.json')
        timed = defaultdict(float)
        for proof in m['proofs'].values():
            if proof['status'] != 'CHECKED':
                raise ValueError('unavailable proof contradicts archive')
            timed[proof['kind']] += proof['proposal_and_check_seconds']
        total = c['check']['required']
        unresolved = [o for o in c['obligations'] if not o['positive']]
        rows.append({'case': f'{model}/{index}', 'evidence_path': 'pre_F0_rational',
                     'required': total, 'positive': total - len(unresolved),
                     'classification': classify(total, total-len(unresolved)),
                     'unresolved': unresolved, 'trusted_base': c['check']['trusted_base'],
                     'generation_seconds': c['generation_seconds'],
                     'check_seconds': c['independent_check_seconds'],
                     'grouped_inline_seconds': dict(timed),
                     'unseparated_generation_overhead_seconds': c['generation_seconds'] - sum(timed.values()),
                     'propagation_exclusive_seconds': None, 'serialization_exclusive_seconds': None,
                     'raw_bytes': c['raw_directory_bytes']})
    supplied_archive = read(ARCHIVE / 'conv_request_sign_lp_review_20260915_r1.json')
    pre_archive = read(ARCHIVE / 'conv_pre_f0_review_20260915_r2.json')
    for archive, directory in [(supplied_archive, 'conv_request_sign_lp_20260915_r1'),
                               (pre_archive, 'conv_pre_f0_rational_20260915_r2')]:
        for item in archive['artifact_inventory']:
            if digest((RAW / directory / item['path']).read_bytes()) != item['sha256']:
                raise ValueError('convolutional archive drift')
    supplied = read(RAW / 'conv_request_sign_lp_20260915_r1/runtime.json')
    diagnostics = read(ARCHIVE / 'conv_full_v2_obligations_20260915.json')
    for index in (16, 98):
        c = next(c for c in supplied['cases'] if c['case']['dataset_index'] == index)
        context = next(r for r in diagnostics['rows'] if r['dataset_index'] == index and r['method'] == 'monolithic')
        result = c['result']
        if index == 16:
            stages = c['stages']
            manifest = read(RAW / 'conv_request_sign_lp_20260915_r1' / c['case']['job_id'] / 'manifest.json')
            groups = {'supplied_weighted_LP_read_propose_write': sum(r.get('proposal_seconds', 0) for r in manifest['obligations'] if r['kind']=='lp')}
            path = 'supplied_floating_F0_HZ'
        else:
            runtime = read(RAW / 'conv_pre_f0_rational_20260915_r2/runtime.json')
            stages = runtime['stages']; result = runtime['result']; path = 'pre_F0_rational'
            calls = read(RAW / 'conv_pre_f0_rational_20260915_r2/rank24_monolithic/query_log.json')
            groups = defaultdict(float)
            for call in calls:
                kind = 'router_order' if call['key'].startswith('gate_') else ('rational_weighted' if call['key'].endswith('_rational') else 'difference')
                groups[kind] += call['seconds']
            groups = dict(groups)
        nonpositive = [r for r in result['obligations'] if r['lower_bound'] is None or Fraction(r['lower_bound']) <= 0]
        rows.append({'case': f'conv/{index}', 'evidence_path': path,
                     'required': result['required_obligations'], 'positive': result['positive_obligations'],
                     'classification': classify(result['required_obligations'], result['positive_obligations']),
                     'unresolved': nonpositive, 'trusted_base': result['trusted_base'],
                     'capture_seconds': stages['capture']['wall_seconds'],
                     'proposal_build_local_checks_seconds': stages['proposal']['wall_seconds'],
                     'grouped_inline_seconds': groups,
                     'unseparated_proposal_overhead_seconds': stages['proposal']['wall_seconds']-sum(groups.values()),
                     'check_seconds': stages['check']['wall_seconds'],
                     'second_check_seconds': stages['independent']['wall_seconds'],
                     'all_stages_seconds': sum(s['wall_seconds'] for s in stages.values()),
                     'propagation_exclusive_seconds': None, 'serialization_exclusive_seconds': None,
                     'production_context': {'classification': 'SOLVER_LIMIT_WITHOUT_CHECKABLE_CERTIFICATE',
                         'old_status': 'TIMEOUT', 'obligation_counts': context['obligation_counts'],
                         'native_categories': context['native_categories'],
                         'source_job': context['job_id']}})
    return {'schema': 'PROOF_CLOSURE_COSTS_V1', 'status': 'PASS', 'new_solves': 0,
            'source_sha256': inputs, 'cases': rows,
            'cost_contract': {
                'ACT_only_inline': 'export serialization, manifest writes, proposal and local arithmetic/construction checks; not pure solver time',
                'conv_preF0_inline': 'proposal including its local rational bound calculation and certificate write; excludes outer construction rechecks',
                'capture': 'startup/load/route/propagation/support/export; NOT exclusive propagation',
                'unseparated_overhead': 'residual elapsed time, not assigned to serialization or propagation',
                'null': 'not separately measured; not zero',
                'checks': 'separate processes, additional to generation; no overlap claimed',
                'claim': 'different proof-study executions/trust boundaries, not paired speedup'},
            'production_acceptance_changed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        raise ValueError('new output under repository required')
    result = analyze()
    a.output.write_bytes(original_bytes(result))
    for row in result['cases']:
        print(row['case'], row['classification'], row['positive'], '/', row['required'], row['grouped_inline_seconds'])
