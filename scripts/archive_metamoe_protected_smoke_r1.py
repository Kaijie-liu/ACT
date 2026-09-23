"""Saved-only archive: rerun the frozen auditor, no inference/solver/retry."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_metamoe_protected_smoke_r1 import audit
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def archive(config, review):
    started = time.monotonic()
    saved = json.loads(review.read_text())
    fresh = audit(config)
    strip = lambda value: {k: v for k, v in value.items() if k != 'separate_audit_seconds'}
    if strip(saved) != strip(fresh):
        raise ValueError('saved review differs from fresh saved-record audit')
    cfg = json.loads(config.read_text())
    root = Path(cfg['output_root'])
    details = []
    for folder in sorted((root/'mnist_0_act/protected').glob('evaluation_*')):
        if not (folder/'result.json').exists():
            continue
        queries = []
        for qdir in sorted(folder.glob('query_*')):
            ret = json.loads((qdir/'return.json').read_text())
            native = ret['native_result']
            queries.append({'scope': ret['scope'], 'terminal': ret['terminal'],
                'returned_status': ret['returned_status'],
                'local_allocation_seconds': ret['deadline_monotonic']-ret['started_monotonic'],
                'return_elapsed_seconds': ret['return_elapsed_seconds'],
                'observed_overrun_seconds': max(0., ret['started_monotonic']+ret['return_elapsed_seconds']-ret['deadline_monotonic']),
                'completed_native_status': native['status'] if native else None,
                'completed_native_seconds': native['native_seconds'] if native else None,
                'query_sha256': sha256(qdir/'extra.npz') if (qdir/'extra.npz').exists() else None})
        details.append({'evaluation': folder.name, 'queries': queries,
            'native_child_launches': len(list(folder.glob('child_*.stdout'))),
            'total_query_return_seconds': sum(q['return_elapsed_seconds'] for q in queries),
            'total_recorded_cleanup_seconds': sum(q['cleanup_seconds'] for q in
                json.loads((folder/'result.json').read_text())['metadata']['queries']),
            'property_identity': json.loads((folder/'property_identity.json').read_text())})
    value = {**fresh, 'archive_schema': 1, 'config_path': str(config),
        'review_path': str(review), 'review_sha256': sha256(review),
        'archive_source_sha256': sha256(Path(__file__)),
        'source_binding_count': len(cfg['files']), 'saved_query_details': details,
        'original_separate_audit_seconds': saved['separate_audit_seconds'],
        'reconstruction_audit_seconds': fresh['separate_audit_seconds'],
        'raw_file_count': len(fresh['files']),
        'raw_total_bytes': sum(v['bytes'] for v in fresh['files'].values()),
        'work_notes': [
            'First direct auditor launch lacked project PYTHONPATH and raised ModuleNotFoundError: act before reading evidence; corrected launcher set PYTHONPATH=/data1/Kane/MOE/ACT. No frozen source edit, solver retry, dependency change or evidence rewrite.',
            'Native-entry markers for interrupted calls are not completed native results or measured internal native time.',
            'The three infeasible expanded regions are policy-accepted property exclusions; base remains UNKNOWN, so they are not a whole-request SAFE.',
            'Historical base feasible is NOT imported into this request. Original R4 and observation R1 remain sealed.',
            'No default solver replacement, paired speed claim, new cohort, or proof of relaxation inadequacy.'],
        'archive_through_construction_seconds': time.monotonic()-started,
        'archive_cost_excludes': 'interpreter startup/interactive gaps/final JSON serialization+write'}
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = archive(a.config, a.review)
    write(a.output, value)
    print(value['audit'], value['row']['status'], value['raw_file_count'], 'raw files')
