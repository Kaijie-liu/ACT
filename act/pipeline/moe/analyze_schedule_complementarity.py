"""Read-only, versioned phase correction for the frozen four-arm results.

No solving, inferred missing facts, or changes to historical verdicts. This
classifies observed stopping locations, not causal solver difficulty.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from act.pipeline.moe.experiment1 import _inside, _sha256, WRITE_ROOT
from act.pipeline.moe.analyze_paired_followup import describe_rows


def stopping_location(evidence):
    if evidence is None:
        return "UNAVAILABLE_PACKAGE"
    route = evidence['route_coverage']
    branches = evidence['tier1'].get('branches', [])
    if evidence['verdict']['reason'] == 'UNKNOWN_SOLVER_LIMIT':
        if (route['candidate_set_minimal'] and route['route_sets_exact'] and route['coverage_complete']
                and any(b.get('solver_reason') == 'violation_region_undecided' for b in branches)):
            return 'TIER1_EXPERT_VIOLATION_REGION_UNDECIDED'
        if not route['candidate_set_minimal'] or not route['route_sets_exact']:
            return 'INCOMPLETE_ROUTE_ANALYSIS'
        return 'UNKNOWN_SOLVER_LIMIT_PHASE_UNRESOLVED'
    return evidence['verdict']['reason']


def analyze(root):
    root = _inside(Path(root), WRITE_ROOT)
    audit_path = root/'audit.review_20260911.json'
    checked = json.loads(audit_path.read_text())
    if checked['status'] != 'PASS' or checked['issues']:
        raise ValueError('passing historical audit required')
    for name, suffix in [('rows', 'jsonl'), ('runtime', 'json')]:
        if _sha256(root/f'{name}.{suffix}') != checked[f'{name}_sha256']:
            raise ValueError('historical audit/raw drift')
    rows = [json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()]
    describe_rows(rows, list(range(100)))  # complete schedule and conflict checks
    by = {(r['model'], r['rank'], r['method']): r for r in rows}
    entries = []
    for model in sorted({r['model'] for r in rows}):
        for rank in range(100):
            a, b = [by[model, rank, method] for method in ('staged', 'monolithic_f0')]
            if (a['status'] == 'SAFE') == (b['status'] == 'SAFE'):
                continue
            loaded = {}; refs = {}
            for row in (a, b):
                if not row['package']:
                    loaded[row['method']] = None
                    refs[row['method']] = None
                    continue
                package = _inside(Path(row['package']), root)
                manifest = json.loads((package/'manifest.json').read_text())
                if _sha256(package/'manifest.json') != row['manifest_sha256']:
                    raise ValueError('package manifest drift')
                if _sha256(package/'evidence.json') != manifest['evidence_sha256']:
                    raise ValueError('package evidence drift')
                e = json.loads((package/'evidence.json').read_text())
                if e['verdict']['status'] != row['status'] or e['execution']['dataset_index'] != row['dataset_index']:
                    raise ValueError('terminal/package mismatch')
                loaded[row['method']] = e
                refs[row['method']] = {'path': str(package), 'manifest_sha256': row['manifest_sha256'],
                                       'evidence_sha256': manifest['evidence_sha256']}
            winner = 'staged' if a['status'] == 'SAFE' else 'monolithic_f0'
            e = loaded[winner]; coverage = e['route_coverage']
            if not coverage['route_sets_exact'] or not coverage['coverage_complete']:
                raise ValueError('SAFE winner lacks exact route coverage')
            loser = loaded['monolithic_f0' if winner == 'staged' else 'staged']
            if loser and loser['route_coverage']['route_sets_exact']:
                if loser['route_coverage']['feasible_route_sets'] != coverage['feasible_route_sets']:
                    raise ValueError('exact paired route sets disagree')
            tier1 = loaded['staged']
            entries.append({
                'model': model, 'rank': rank, 'dataset_index': a['dataset_index'], 'safe_only': winner,
                'pair_count': len(coverage['feasible_route_sets']),
                'route_complexity': 'single' if len(coverage['feasible_route_sets']) == 1 else 'multiple',
                'staged_terminal_reason': a['reason'], 'staged_stop_location': stopping_location(tier1),
                'staged_route_analysis_complete': (tier1['route_coverage']['candidate_set_minimal']
                    and tier1['route_coverage']['route_sets_exact']) if tier1 else None,
                'staged_branch_metadata': [{k: v.get(k) for k in
                    ('candidate', 'unknown_reason', 'solver_status', 'solver_reason', 'solve_stages')}
                    for v in tier1['tier1'].get('branches', [])
                    if v.get('solver_reason') == 'violation_region_undecided']
                    if tier1 and a['reason'] == 'UNKNOWN_SOLVER_LIMIT' else None,
                'packages': refs,
            })
    summary = {}
    for winner in ('staged', 'monolithic_f0'):
        subset = [e for e in entries if e['safe_only'] == winner]
        summary[winner] = {'safe_only_count': len(subset),
                          'route_complexity': dict(Counter(e['route_complexity'] for e in subset)),
                          'staged_stop_locations': dict(Counter(e['staged_stop_location'] for e in subset))}
    return {'schema': 'schedule_complementarity_phase_review_v1', 'raw_root': str(root),
            'audit_sha256': _sha256(audit_path), 'rows_sha256': checked['rows_sha256'],
            'runtime_sha256': checked['runtime_sha256'], 'summary': summary, 'entries': entries,
            'scope': 'Associated stopping locations and exact legal-pair counts; no causal attribution, no new solver queries, no historical fact reconstruction.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2, sort_keys=True))
