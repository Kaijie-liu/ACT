"""Archive a fresh structural/witness audit; never launch verification solves."""
import argparse
import json

from scripts.audit_conv_full_v2 import audit
from scripts.conv_full_v2_contract import ROOT, DEFAULT
from scripts.conv_three_arm_contract import read
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256

OUTPUT = ROOT / 'act/pipeline/moe/results/conv_full_v2_review_20260915.json'


def witness_context(rows):
    """Union is retrospective context, not a new single-budget method."""
    unsafe = {r['dataset_index'] for r in rows if r['status'] == 'UNSAFE'}
    return {
        'unsafe_method_runs': sum(r['status'] == 'UNSAFE' for r in rows),
        'distinct_inputs_with_replayed_witness': sorted(unsafe),
        'timeout_with_other_arm_witness': {
            arm: sorted(r['dataset_index'] for r in rows
                        if r['method'] == arm and r['status'] == 'TIMEOUT'
                        and r['dataset_index'] in unsafe)
            for arm in ('adaptive', 'monolithic')},
        'scope': 'Cross-arm observations only; no historical verdict is relabelled.'}


def review():
    fresh = audit(DEFAULT)
    summary = read(DEFAULT / 'FULL_SUMMARY.json')
    supervisor = read(DEFAULT / 'supervisor.json')
    if (fresh != read(DEFAULT / 'audit.final.json')
            or fresh != read(DEFAULT / 'audit.independent.json')
            or fresh != summary['audit'] or fresh['status'] != 'PASS'
            or fresh['rows'] != 90 or fresh['unattempted']
            or summary['run_terminal'] != read(DEFAULT / 'run_terminal.json')
            or not summary['audits_match'] or summary['additional_experiment_queued']
            or supervisor['state'] != 'COMPLETED_AUDITED'
            or supervisor['completed'] != 90 or supervisor['additional_experiment_queued']):
        raise ValueError('full run/audit closure differs')
    rows = [json.loads(l) for l in (DEFAULT / 'rows.jsonl').read_text().splitlines()]
    runtime = read(DEFAULT / 'runtime.json')
    return {
        'schema': 'CONV_FULL_V2_ARCHIVAL_REVIEW_V1', 'status': 'PASS', 'issues': [],
        'execution_head': runtime['git_head'], 'raw_root': str(DEFAULT),
        'automatic_separate_and_fresh_audits_equal': True,
        'audit_summary': {k: v for k, v in fresh.items() if k != 'details'},
        'terminals': [{k: r[k] for k in ('job_id', 'rank', 'dataset_index', 'method',
                                        'status', 'outer_timeout', 'wall_seconds', 'evidence_level')}
                      for r in rows],
        'witness_context': witness_context(rows),
        'launch_to_completed_audited_hours': (supervisor['updated_unix'] - runtime['started_unix']) / 3600,
        'artifact_inventory': [
            {'path': str(p.relative_to(DEFAULT)), 'bytes': p.stat().st_size, 'sha256': _sha256(p)}
            for p in sorted(DEFAULT.rglob('*')) if p.is_file()],
        'scope': 'Fresh structural, identity, journal and full-model witness review. No new solver query, '
                 'no independent SAFE reproof. CROWN numerical positives are not formal SAFE. '
                 'Frozen R1 and all V2 bytes retained; descriptive unit is 30 inputs.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = review()
    if args.check:
        if read(OUTPUT) != result:
            raise ValueError('archive differs from fresh reconstruction')
    else:
        if OUTPUT.exists():
            raise FileExistsError('archive exists; use --check')
        atomic_json(OUTPUT, result)
    print(json.dumps({k: result[k] for k in ('status', 'witness_context',
                                           'launch_to_completed_audited_hours')}, indent=2))
