"""Saved-only exact recheck/differential and full cost archive; no new queries."""
import argparse
from fractions import Fraction as F
from pathlib import Path
import statistics
import time

from scoped_proof.io import ROOT, load, save, sha
from scoped_proof.evidence import roster
from source_enclosure.format import identity
from checked_route_frontier.study import CONFIG, OUTPUT

REPORT = ROOT/'docs/checked_route_frontier_results_20260924_r1.json'


def review():
    cfg = load(CONFIG)
    launch = load(OUTPUT/'launch.json')
    if launch['config_sha256'] != sha(CONFIG) or launch['config'] != cfg:
        raise ValueError('execution freeze identity')
    for name, digest in cfg['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('frozen source changed: '+name)
    executed = load(OUTPUT/'execution.json')
    if (executed['new_real_requests'] != 0 or executed['new_solver_calls'] != 0 or
            len(executed['rows']) != len(cfg['calls'])):
        raise ValueError('execution roster')
    summaries = []
    for call, execution in zip(cfg['calls'], executed['rows']):
        if any(execution[k] != v for k, v in call.items()):
            raise ValueError('execution call changed')
        root = OUTPUT/call['id']
        if execution['root'] != str(root):
            raise ValueError('execution directory')
        plan, cost, terminal = [load(root/name) for name in ('plan.json', 'cost.json', 'terminal.json')]
        if (terminal['plan_sha256'] != sha(root/'plan.json') or cost['terminal_sha256'] != sha(root/'terminal.json') or
                plan['mode'] != call['mode'] or plan['fixture'] != call['fixture'] or
                plan['budget_seconds'] != cfg['budget_seconds'] or
                plan['expected_source_sha256'] != cfg['fixture_sha256'][call['fixture']] or
                abs(cost['stage_seconds']+cost['overhead_seconds']-cost['end_to_end_seconds']) > 1e-9 or
                cost['overhead_seconds'] < 0 or execution['seconds'] < cost['end_to_end_seconds']):
            raise ValueError('request identity/cost closure')
        phases = terminal['phases']
        if (sum(row['seconds'] for row in phases) != cost['stage_seconds'] or
                [row['phase'] for row in phases] != ['build', 'check'][:len(phases)]):
            raise ValueError('phase inventory/cost')
        row = {**call, 'status': execution['status'], 'returned_seconds': execution['seconds'],
               'ledger_seconds': cost['end_to_end_seconds'], 'sampled_peak_rss': cost['sampled_peak_rss'],
               'phase_seconds': {p['phase']: p['seconds'] for p in phases},
               'synthetic_positive': False, 'independent_recheck': None}
        if execution['status'] == 'COMPLETED_SYNTHETIC_CHECK':
            if (cost['status'] != execution['status'] or terminal['status'] != execution['status'] or
                    len(phases) != 2 or any(p['status'] != 'COMPLETED' or p['returncode'] != 0 for p in phases) or
                    (root/'publication_timeout.json').exists() or cost['end_to_end_seconds'] >= cfg['budget_seconds'] or
                    terminal['check_sha256'] != sha(root/'check.json')):
                raise ValueError('late/incomplete proof cannot be accepted')
            doc, bundle, candidates, stored = [load(root/name) for name in
                ('source.json', 'construction.json', 'candidates.json', 'check.json')]
            if identity(doc) != plan['expected_source_sha256']:
                raise ValueError('wrong source')
            if call['mode'] == 'exhaustive':
                from scoped_proof.evidence import aggregate
            else:
                from checked_route_frontier.evidence import aggregate
            scope = {k: v for k, v in doc['request'].items() if k not in ('top_k','gate','tie_policy','training')}
            checked = aggregate(scope, doc, bundle, {int(k): v for k,v in candidates.items()},
                invocation=plan['invocation'], proposal_complete=True, deadline=time.monotonic()+300)
            if checked != stored or terminal['synthetic_positive'] != checked['complete_output_positive_proof']:
                raise ValueError('fresh exact aggregation differs')
            expected = roster(scope)
            if len(checked['rows']) != len(expected) or any(
                    any(actual[k] != value for k,value in obligation.items())
                    for actual, obligation in zip(checked['rows'], expected)):
                raise ValueError('original obligation inventory')
            row.update(synthetic_positive=checked['complete_output_positive_proof'],
                independent_recheck='MATCH', required=checked['required'],
                checked_bounds=checked['checked_bounds'],
                excluded=checked.get('discharged_by_exclusion', 0),
                source_sha256=identity(doc),
                construction_bytes=(root/'construction.json').stat().st_size,
                evidence_bytes=sum((root/name).stat().st_size for name in
                    ('source.json','construction.json','candidates.json','check.json')))
        elif terminal['synthetic_positive'] and execution['status'] not in ('TIMEOUT', 'ERROR'):
            raise ValueError('partial success classification')
        summaries.append(row)
    differences = []
    for fixture in ('prunable', 'tied'):
        for repeat in range(3):
            roots = {mode: OUTPUT/f'{fixture}_{repeat}_{mode}' for mode in ('exhaustive', 'frontier')}
            states = {row['mode']: row for row in summaries if row['fixture']==fixture and row['repeat']==repeat}
            if not all(v['status']=='COMPLETED_SYNTHETIC_CHECK' for v in states.values()):
                differences.append({'fixture': fixture, 'repeat': repeat, 'status': 'INCOMPLETE_NOT_EQUAL'})
                continue
            old, new = [load(roots[mode]/'construction.json') for mode in ('exhaustive','frontier')]
            old_pairs = {tuple(p['pair']): p for p in old['pairs']}
            if (new['prefix']['input'] != old['input'] or new['prefix']['router'] != old['networks'][0] or
                    any(p != old_pairs[tuple(p['pair'])] for p in new['pairs'])):
                raise ValueError('retained matrix differential')
            old_traces = {t['name']:t for t in old['networks']}
            if any(t != old_traces[t['name']] for t in new['experts']):
                raise ValueError('retained expert source differential')
            differences.append({'fixture': fixture, 'repeat': repeat, 'status': 'EXACT_RETAINED_MATCH',
                'retained_pairs': len(new['pairs']), 'needed_experts': len(new['experts']),
                'frontier_minus_exhaustive_seconds': states['frontier']['returned_seconds']-states['exhaustive']['returned_seconds']})
    statistics_rows = []
    for fixture in ('prunable', 'tied'):
        for mode in ('exhaustive', 'frontier'):
            rows = [r for r in summaries if r['fixture']==fixture and r['mode']==mode]
            statistics_rows.append({'fixture': fixture, 'mode': mode, 'attempted': len(rows),
                'completed': sum(r['status']=='COMPLETED_SYNTHETIC_CHECK' for r in rows),
                'median_returned_seconds': statistics.median(r['returned_seconds'] for r in rows),
                'median_build_seconds': statistics.median(r['phase_seconds']['build'] for r in rows),
                'median_check_seconds_if_present': statistics.median(r['phase_seconds']['check'] for r in rows
                    if 'check' in r['phase_seconds']) if any('check' in r['phase_seconds'] for r in rows) else None,
                'median_evidence_bytes_if_complete': statistics.median(r['evidence_bytes'] for r in rows
                    if 'evidence_bytes' in r) if any('evidence_bytes' in r for r in rows) else None})
    files = {str(p.relative_to(OUTPUT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    return {'schema': 'CHECKED_ROUTE_FRONTIER_SYNTHETIC_REVIEW_V1', 'status': 'PASS', 'issues': [],
            'execution_head': launch['head'], 'config_sha256': sha(CONFIG), 'calls': summaries,
            'differentials': differences, 'statistics': statistics_rows, 'raw_files': files,
            'new_real_requests': 0, 'new_native_solver_calls': 0,
            'claim': 'Analytic mechanism/cost controls only; no real speedup, external win or native-float proof.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    result = review()
    if a.check:
        if load(REPORT) != result:
            raise ValueError('archive changed')
    else:
        save(REPORT, result)
    print({'status': result['status'], 'calls': len(result['calls']), 'statistics': result['statistics']})
