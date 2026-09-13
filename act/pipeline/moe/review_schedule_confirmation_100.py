"""Archive-only reconstruction of the frozen hundred-input experiment.

No verification/optimization queries. The existing auditor replays UNSAFE
concretely and checks structures; this is not an independent SAFE proof checker.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import subprocess

from act.pipeline.moe.experiment1 import PROJECT_ROOT, _sha256
from act.pipeline.moe.paired_followup import save
from act.pipeline.moe.schedule_confirmation import audit
from act.pipeline.moe.route_complexity_paired import counters

BASE = PROJECT_ROOT/'data/moe/results'
FULL = BASE/'schedule_confirmation_100_full_20260912_r1'
SMOKE = BASE/'schedule_confirmation_100_smoke_20260912_r1'
OUTPUT = PROJECT_ROOT/'act/pipeline/moe/results/schedule_confirmation_100_review_20260914_r1.json'
EXECUTION_HEAD = 'bc0791976b00879c28c268692aecc5854b3bc091'


def frozen_source_identity():
    """Ignore newly added archival tools, never changes to execution sources."""
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', EXECUTION_HEAD, 'act'],
                                    cwd=PROJECT_ROOT, text=True).splitlines()
    digest = hashlib.sha256()
    for name in sorted(n for n in names if n.endswith('.py')):
        digest.update(f'{name}:{_sha256(PROJECT_ROOT/name)}\n'.encode())
    return digest.hexdigest()


def reference(path):
    return {'path': str(path), 'sha256': _sha256(path)}


def evidence_summary(e):
    if e is None:
        return {'decision_tier': None, 'reason': 'OUTER_HARD_DEADLINE',
                'selected_path': None, 'exact_pair_count': None, 'property_counters': None}
    schedule = e.get('route_complexity_schedule', {})
    coverage = e['route_coverage']
    count = len(coverage['feasible_route_sets']) if coverage['route_sets_exact'] else None
    # A fact inventory is not the same as facts actually used in the certificate.
    # The existing counter validates each recorded scoped reuse row.
    return {'decision_tier': e['verdict']['decision_tier'], 'reason': e['verdict']['reason'],
            'selected_path': schedule.get('selected_path'), 'exact_pair_count': count,
            'feasible_route_sets': coverage['feasible_route_sets'],
            'available_fact_count': e.get('proof_reuse', {}).get('available_fact_count'),
            'property_counters': counters(e), 'tier2_invoked': e['tier2']['invoked'],
            'tier2_elapsed_seconds': e['tier2'].get('elapsed_seconds'),
            'common_fact_seconds': schedule.get('common_fact_seconds'),
            'unfinished_properties': [{k: p.get(k) for k in
                ('property_index', 'reason', 'solver_status', 'solver_gap', 'elapsed')}
                for p in e['tier2'].get('property_rows', []) if p.get('status') != 'SAFE']}


def derive(full=FULL):
    rows = [json.loads(line) for line in (full/'rows.jsonl').read_text().splitlines()]
    details = {}; missing = []; joint = {}; cost_by_state = {}; totals = {}
    for row in rows:
        directory = full/row['job_id']; terminal = directory/'terminal.json'
        if json.loads(terminal.read_text()) != row:
            raise ValueError('terminal file disagrees with append-only ledger')
        e = None; evidence_ref = None
        if row['package'] is not None:
            p = Path(row['package'])/'evidence.json'; e = json.loads(p.read_text()); evidence_ref = reference(p)
        detail = {k: row[k] for k in ('job_id', 'model', 'rank', 'dataset_index', 'status', 'wall_seconds', 'outer_timeout')}
        detail.update(terminal=reference(terminal), evidence=evidence_ref,
                      snapshot=reference(directory/'common_facts.json') if row['snapshot_sha256'] else None,
                      **evidence_summary(e))
        if detail['exact_pair_count'] is None and detail['snapshot']:
            payload = json.loads((directory/'common_facts.json').read_text())['payload']
            # Snapshot schema is independently checked by the frozen auditor.
            detail['exact_pair_count'] = len(payload['feasible_route_sets'])
        details[row['model'], row['rank'], row['method']] = detail
        if e is None:
            if not row['outer_timeout'] or row['status'] != 'TIMEOUT':
                raise ValueError('missing package is not a retained outer TIMEOUT')
            missing.append(detail)
    for arm in ('adaptive', 'matched', 'legacy'):
        rr = [r for r in rows if r['method'] == arm]
        totals[arm] = {'states': dict(Counter(r['status'] for r in rr)),
                       'rows': len(rr), 'mean_seconds': mean(r['wall_seconds'] for r in rr)}
        cost_by_state[arm] = {}
        for status in ('SAFE', 'UNSAFE', 'UNKNOWN', 'TIMEOUT'):
            times = [r['wall_seconds'] for r in rr if r['status'] == status]
            cost_by_state[arm][status] = {'count': len(times),
                'mean_seconds': mean(times) if times else None,
                'median_seconds': median(times) if times else None}
        joint[arm] = dict(Counter(
            ('package' if r['package'] else 'no_package') + ('+snapshot' if r['snapshot_sha256'] else '+no_snapshot')
            for r in rr))
    contrasts = {}
    for arm in ('matched', 'legacy'):
        gain = []; loss = []
        for model in ('seed0', 'seed1', 'seed2'):
            for rank in range(100):
                a, b = details[model, rank, 'adaptive'], details[model, rank, arm]
                if (a['status']=='SAFE') != (b['status']=='SAFE'):
                    (gain if a['status']=='SAFE' else loss).append({'adaptive': a, 'baseline': b})
        contrasts[arm] = {'gained_safe': gain, 'lost_safe': loss,
            'gained_safe_pair_counts': dict(Counter(str(r['adaptive']['exact_pair_count']) for r in gain)),
            'gained_safe_decision_tiers': dict(Counter(r['adaptive']['decision_tier'] for r in gain)),
            'gained_safe_with_recorded_reuse': sum(
                (r['adaptive']['property_counters'] or {}).get('reused_pair_properties', 0)>0 for r in gain)}
    return {'totals': totals, 'safe_discordances': contrasts, 'package_snapshot_joint_counts': joint,
            'missing_package_terminals': missing, 'cost_by_terminal_state': cost_by_state,
            'scope': 'Descriptive post-run extraction, not causal ablation. State-conditioned costs use different solution sets. Counters describe recorded rows, not optimal MILP counts; null is not zero.'}


def build():
    source = frozen_source_identity(); result = {}
    for name, root in [('smoke', SMOKE), ('full', FULL)]:
        runtime = json.loads((root/'runtime.json').read_text())
        if runtime['git_head'] != EXECUTION_HEAD or runtime['source_sha256'] != source:
            raise ValueError('frozen execution source identity changed')
        actual = audit(root)
        if actual != json.loads((root/'audit.final.json').read_text()):
            raise ValueError('independent audit differs from saved final summary')
        result[name] = {'raw_root': str(root), 'audit_file': reference(root/'audit.final.json'),
                        'runtime': reference(root/'runtime.json'), 'audit': actual}
    result.update(schema='schedule_confirmation_100_review_v1', experiment_head=EXECUTION_HEAD,
                  source_sha256=source, independent_review={'status': 'PASS', 'issues': [],
                  'separate_process_reaudit_equals_saved': True,
                  'scope': 'Frozen structural auditor plus full-model UNSAFE replay; not independent reproof of SAFE'},
                  descriptive_supplement=derive())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true'); mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.write and OUTPUT.exists(): raise ValueError('do not overwrite an archived review')
    result = build()
    if args.write: save(OUTPUT, result)
    elif result != json.loads(OUTPUT.read_text()): raise ValueError('archived review does not reconstruct')
    print(json.dumps({'status': 'PASS', 'review': str(OUTPUT), 'rows': result['full']['audit']['rows']}))


if __name__ == '__main__': main()
