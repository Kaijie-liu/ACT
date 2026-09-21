"""Independent full-roster saved attack replay and compact empirical archive."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from audit_rome_multinorm import audit
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def archive(path, destination):
    start = time.monotonic()
    cfg = json.loads(path.read_text())
    for p, h in cfg['files'].items():
        if sha256(p) != h:
            raise ValueError('batch identity')
    summary_path = Path(cfg['output_root'])/'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['config_sha256'] != sha256(path):
        raise ValueError('summary identity')
    expected = [json.loads(Path(r['config']).read_text()) for r in cfg['requests']]
    if [(r['index'], r['norm']) for r in summary['rows']] != [(r['index'],r['norm']) for r in expected]:
        raise ValueError('incomplete/reordered roster')
    destination.mkdir(exist_ok=False)
    rows, blocked = [], False
    for i, (source, request, row) in enumerate(zip(cfg['requests'], expected, summary['rows'])):
        if row['status'] == 'NOT_STARTED_AFTER_ERROR':
            if not blocked:
                raise ValueError('unexplained missing outcome')
            rows.append({k: row[k] for k in ['index','norm','status']})
            continue
        root = Path(request['output_root'])
        terminal = json.loads((root/'terminal.json').read_text())
        if row['terminal'] != terminal or row['status'] != terminal['status']:
            raise ValueError('batch/terminal mismatch')
        review = audit(Path(source['config']))
        write(destination/f'audit{i:02d}.json', review)
        done = row['status'] == 'ATTACK_EVALUATION_COMPLETED'
        if done and (review['audit'] != 'INDEPENDENT_FULL_MODEL_REPLAY_PASS' or
                sha256(root/'result.json') != row['result_sha256']):
            raise ValueError('missing completed attack replay')
        if done and terminal['execution_seconds'] > request['total_seconds']:
            raise ValueError('late completed result')
        if not done and review['audit'] != 'RECORDED_NON_COMPLETION':
            raise ValueError('failure relabeled')
        result = review.get('result', {})
        rows.append({'index': row['index'], 'norm': row['norm'], 'status': row['status'],
            'execution_seconds': terminal['execution_seconds'],
            'with_postflight_seconds': terminal['total_with_postflight_seconds'],
            'clean_prediction': result.get('clean_prediction'),
            'adversarial_prediction': result.get('adversarial_prediction'),
            'label': result.get('label'),
            'empirically_correct_after_attack': result.get('empirically_correct_after_attack'),
            'distance_float64': review.get('registered_norm_distance_float64'),
            'inside_epsilon_without_tolerance': review.get('inside_requested_epsilon_in_float64_check'),
            'audit_sha256': sha256(destination/f'audit{i:02d}.json'),
            'terminal_sha256': sha256(root/'terminal.json'),
            'receipt_sha256': sha256(root/'receipt.json'), 'result_sha256': row.get('result_sha256')})
        blocked = row['status'] not in ['ATTACK_EVALUATION_COMPLETED', 'TIMEOUT']
    union = []
    for index in cfg['indices']:
        group = [r for r in rows if r['index'] == index]
        complete = all(r['status'] == 'ATTACK_EVALUATION_COMPLETED' for r in group)
        broken = any(r.get('empirically_correct_after_attack') is False for r in group)
        union.append({'index':index, 'status': 'ATTACK_FOUND' if broken else
            ('NO_ATTACK_FOUND_ALL_THREE' if complete else 'INCOMPLETE_NOT_ROBUST'),
            'all_three_completed': complete})
    if union != summary['union']:
        raise ValueError('union aggregation mismatch')
    return {'audit': 'INDEPENDENT_EMPIRICAL_BATCH_REVIEW_PASS', 'rows': rows, 'union':union,
        'config_sha256': sha256(path), 'raw_summary_sha256': sha256(summary_path),
        'denominator_inputs':len(cfg['indices']), 'denominator_norm_requests':len(expected),
        'counts':dict(Counter(r['status'] for r in rows)),
        'execution_seconds':sum(r.get('execution_seconds',0) for r in rows),
        'total_with_postflight_seconds':sum(r.get('with_postflight_seconds',0) for r in rows),
        'independent_audit_seconds_separate':time.monotonic()-start,
        'evidence_grade':'EMPIRICAL_DEPLOYMENT_ONLY', 'formal_SAFE':False,
        'paper_scale_reproduction':False, 'ACT_fair_comparison_completed':False,
        'scope':'fixed source-default MAX model; no attack parameters changed or timeout counted robust'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    write(a.output, archive(a.config,a.audit_root))
