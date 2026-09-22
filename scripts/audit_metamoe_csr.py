"""Independent R3 terminal/resource audit; not a reproof of positive bounds."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def check_terminal(row, receipt, result):
    if receipt['status'] != 'COMPLETED':
        if row['status'] != receipt['status']:
            raise ValueError('outer failure upgraded')
    elif result is None:
        if row['status'] != 'ERROR':
            raise ValueError('missing candidate not ERROR')
    elif row['status'] != result['status']:
        raise ValueError('candidate terminal changed')
    if row['status'] in ('POSITIVE', 'SAFE', 'BACKEND_POSITIVE', 'UNSAFE_REPLAYED'):
        if row['seconds'] >= receipt['deadline_seconds']:
            raise ValueError('late conclusion')


def diagnostic_review(config, root):
    cfg = json.loads(config.read_text())
    receipt = json.loads((root/'receipt.json').read_text())
    if receipt['deadline_seconds'] != 90 or receipt['group_rss_limit_bytes'] != cfg['group_rss_limit_bytes']:
        raise ValueError('diagnostic resource binding')
    for stream in ('stdout', 'stderr'):
        if sha256(root/f'{stream}.txt') != receipt[f'{stream}_sha256']:
            raise ValueError('diagnostic log binding')
    file = root/'diagnostic.json'
    result, parse_error = None, None
    if file.exists():
        try:
            result = json.loads(file.read_text())
        except ValueError as exc:
            parse_error = repr(exc)
    passed = False
    if result:
        if result['config_sha256'] != sha256(config):
            raise ValueError('diagnostic request binding')
        for i, event in enumerate(result['events'], 1):
            if json.loads((root/f'layer_{i:03d}.json').read_text()) != event:
                raise ValueError('diagnostic progress binding')
        phases = {e['phase'] for e in result['events'] if e['sparse'] is not None}
        passed = (receipt['status'] == 'COMPLETED' and result['status'] == 'COMPLETE' and
            receipt['execution_including_preflight_seconds'] < 90 and
            receipt['peak_sampled_group_rss_bytes'] <= cfg['group_rss_limit_bytes'] and
            phases == {'router', 'guarded_expert_0', 'guarded_expert_1'} and
            result['completed_phases'] == ['router', 'guarded_expert_0', 'guarded_expert_1'] and
            result['solver_calls'] == 0 and all(not e['dense_retained'] for e in result['events']))
    return {'passed': passed, 'config_sha256': sha256(config), 'receipt': receipt, 'result': result,
            'parse_error': parse_error,
            'files': {str(f): sha256(f) for f in sorted(root.glob('*.json'))}}


def audit(config):
    started = time.monotonic()
    cfg = json.loads(config.read_text())
    for name, digest in cfg['files'].items():
        if sha256(name) != digest:
            raise ValueError('execution/input identity: '+name)
    root = Path(cfg['output_root'])
    summary = json.loads((root/'summary.json').read_text())
    if summary['config_sha256'] != sha256(config):
        raise ValueError('summary binding')
    expected = [(r['id'], arm) for i, r in enumerate(cfg['requests'])
                for arm in (['act', 'author'] if i % 2 == 0 else ['author', 'act'])]
    if [(r['id'], r['arm']) for r in summary['rows']] != expected:
        raise ValueError('roster/order')
    blocked, reviewed = False, []
    for row in summary['rows']:
        if row['status'] == 'NOT_STARTED_AFTER_ERROR':
            if not blocked:
                raise ValueError('unexplained missing request')
            reviewed.append(row)
            continue
        if blocked:
            raise ValueError('ran after stop-on-error')
        folder = root/f"{row['id']}_{row['arm']}"
        if json.loads((folder/'terminal.json').read_text()) != row:
            raise ValueError('terminal ledger binding')
        receipt = json.loads((folder/'receipt.json').read_text())
        if (sha256(folder/'receipt.json') != row['receipt_sha256'] or
                receipt['deadline_seconds'] != cfg['seconds'] or
                receipt['group_rss_limit_bytes'] != cfg['group_rss_limit_bytes'] or
                receipt['execution_including_preflight_seconds'] != row['seconds']):
            raise ValueError('resource/cost receipt mismatch')
        for stream in ('stdout', 'stderr'):
            if sha256(folder/f'{stream}.txt') != receipt[f'{stream}_sha256']:
                raise ValueError('log mismatch')
        result = None
        if row['result_sha256']:
            if sha256(folder/'result.json') != row['result_sha256']:
                raise ValueError('candidate bytes changed')
            try:
                result = json.loads((folder/'result.json').read_text())
                if not isinstance(result, dict) or not isinstance(result.get('status'), str):
                    raise ValueError('invalid candidate')
            except ValueError:
                result = None
                if not row['result_parse_error']:
                    raise ValueError('unrecorded parse failure')
        check_terminal(row, receipt, result)
        if result:
            req = next(r for r in cfg['requests'] if r['id'] == row['id'])
            if (result['config_sha256'] != sha256(config) or result['request_id'] != row['id'] or
                    result['arm'] != row['arm'] or result['label'] != req['label'] or
                    result['tensor_file_sha256'] != sha256(req['tensor_file'])):
                raise ValueError('same-object candidate binding')
        blocked = row['status'] in ('ERROR', 'SOURCE_CHANGED')
        reviewed.append({**row, 'grade': result.get('evidence_grade', 'NONE') if result else 'NONE',
            'reason': result.get('reason') if result else None,
            'peak_sampled_group_rss_bytes': receipt['peak_sampled_group_rss_bytes']})
    cost = json.loads((root/'batch_cost.json').read_text())
    if (cost['config_sha256'] != sha256(config) or cost['charged_request_seconds'] !=
            sum(r.get('seconds', 0.) for r in reviewed) or
            cost['batch_wall_through_summary_seconds'] < cost['charged_request_seconds']):
        raise ValueError('batch cost binding')
    # Ordinary complete numerical UNKNOWN does not invalidate intake. Resource
    # refusal, missing representations, timeout, and execution faults do.
    controls = all(r['status'] in ('POSITIVE', 'BACKEND_POSITIVE', 'UNSAFE_REPLAYED', 'UNKNOWN') and
        r.get('reason') not in ('sparse_representation_resource_limit', 'missing_correlated_router_hz')
        for r in reviewed)
    return {'audit': 'PASS', 'config_sha256': sha256(config),
        'summary_sha256': sha256(root/'summary.json'), 'rows': reviewed, 'batch_cost': cost,
        'counts': {arm: dict(Counter(r['status'] for r in reviewed if r['arm'] == arm)) for arm in ('act', 'author')},
        'execution_control_pass': controls, 'separate_audit_seconds': time.monotonic()-started,
        'trust': 'structural/identity/cost review, NOT independent positive-bound proof; original witness replay separate',
        'numerical_guarantees_equated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path)
    args = parser.parse_args()
    value = diagnostic_review(args.config, args.diagnostic) if args.diagnostic else audit(args.config)
    write(args.output, value)
    print('diagnostic_pass', value['passed']) if args.diagnostic else print(value['counts'])
