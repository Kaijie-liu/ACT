"""Compact saved-only archive; independently reconstruct review before copying."""
import argparse
import json
from pathlib import Path
import time

from audit_metamoe_expert_diagnostic import audit
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def archive(config, review_path):
    started = time.monotonic()
    saved = json.loads(review_path.read_text())
    checked = audit(config)
    strip = lambda d: {k: v for k, v in d.items() if k != 'separate_audit_seconds'}
    if strip(saved) != strip(checked):
        raise ValueError('saved review differs from independent reconstruction')
    trace = checked['trace']
    by_id = {s['id']: s for s in trace['spans']}
    native = [{**s, 'phase_parent': by_id[s['parent']]['name']} for s in trace['spans']
              if s['name'].endswith('.milp')]
    properties = [q for ev in checked['diagnosis']['expert_evaluations'] for q in ev['queries']
                  if q['arguments'].get('extra_rows', 0) > 0]
    result = {k: v for k, v in checked['result'].items() if k != 'sparse_resource_events'}
    sparse = checked['result'].get('sparse_resource_events', [])
    return {'archive_schema': 1, 'audit': checked['audit'],
        'config_path': str(config), 'config_sha256': checked['config_sha256'],
        'review_path': str(review_path), 'review_sha256': sha256(review_path),
        'archive_source_sha256': sha256(Path(__file__)),
        'row': checked['row'], 'receipt': checked['receipt'], 'result': result,
        'batch_cost': checked['batch_cost'], 'diagnosis': checked['diagnosis'],
        'native_calls_all_phases': native,
        'property_feasibility_wrapper_attempts': len(properties),
        'property_native_calls': sum(len(q['native']) for q in properties),
        'trace': {k: v for k, v in trace.items() if k != 'spans'},
        'sparse_admission_events': len(sparse),
        'source_binding_count': len(json.loads(config.read_text())['files']),
        'files': checked['files'],
        'original_separate_audit_seconds': saved['separate_audit_seconds'],
        'reconstruction_audit_seconds': checked['separate_audit_seconds'],
        'archive_through_construction_seconds': time.monotonic()-started,
        'cost_excludes': 'interpreter/interactive gaps/final archive serialization and write',
        'opens_formal_cohort': False, 'independent_safe_reproof': False,
        'annotations': [
            'The generic trace property_index on verify_once is the CALLER EXPERT index, not an output property row. Only evaluate_spec lane/row/M locate properties.',
            'solves includes a feasibility wrapper returning before native invocation; it is not a native MILP count.',
            'HZ exact/exact_witness are implementation flags, not independent proof of source-to-HZ conversion.',
            'New observation only; do not replace the sealed R4 result or infer its exact native trajectory.',
            'Outer receipt source_unchanged is unset; post-run file/repository/environment validation is performed by the separate auditor.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    value = archive(args.config, args.review)
    write(args.output, value)
    print(value['audit'], 'property_native_calls', value['property_native_calls'])
