"""Independent full-roster and terminal accounting; no SAFE bound reproof."""
import argparse
from collections import Counter
import json
from pathlib import Path
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def audit(path):
    cfg = json.loads(path.read_text())
    root = Path(cfg['output_root'])
    for name, h in cfg['files'].items():
        if sha256(name) != h:
            raise ValueError('execution/input identity')
    summary = json.loads((root / 'summary.json').read_text())
    if summary['config_sha256'] != sha256(path):
        raise ValueError('summary binding')
    expected = [(r['id'], arm) for i, r in enumerate(cfg['requests'])
                for arm in (['act', 'author'] if i % 2 == 0 else ['author', 'act'])]
    if [(r['id'], r['arm']) for r in summary['rows']] != expected:
        raise ValueError('request/order coverage')
    reviewed = []
    blocked = False
    for row in summary['rows']:
        folder = root / f"{row['id']}_{row['arm']}"
        if row['status'] == 'NOT_STARTED_AFTER_ERROR':
            if not blocked:
                raise ValueError('unexplained skipped request')
            reviewed.append(row)
            continue
        receipt = json.loads((folder / 'receipt.json').read_text())
        if (sha256(folder / 'receipt.json') != row['receipt_sha256'] or
                receipt['deadline_seconds'] != 300 or
                row['seconds'] != receipt['execution_including_preflight_seconds']):
            raise ValueError('receipt/cost mismatch')
        for stream in ['stdout', 'stderr']:
            if sha256(folder / f'{stream}.txt') != receipt[f'{stream}_sha256']:
                raise ValueError('changed log')
        result = None
        if row['result_sha256'] is not None:
            if sha256(folder / 'result.json') != row['result_sha256']:
                raise ValueError('result identity')
            result = json.loads((folder / 'result.json').read_text())
            req = next(r for r in cfg['requests'] if r['id'] == row['id'])
            if (result['config_sha256'] != sha256(path) or result['request_id'] != row['id'] or
                    result['arm'] != row['arm'] or result['label'] != req['label'] or
                    result['tensor_file_sha256'] != sha256(req['tensor_file'])):
                raise ValueError('same-object binding')
        if receipt['status'] != 'COMPLETED':
            if row['status'] != receipt['status']:
                raise ValueError('outer failure upgraded')
        elif not result or result['status'] != row['status']:
            raise ValueError('missing/changed result')
        if row['status'] in ['SAFE', 'BACKEND_POSITIVE'] and row['seconds'] > cfg['seconds']:
            raise ValueError('late positive')
        blocked = blocked or row['status'] in ['ERROR', 'SOURCE_CHANGED']
        reviewed.append({**row, 'grade': result.get('evidence_grade', 'NONE') if result else 'NONE',
                         'reason': result.get('reason') if result else None})
    return {'audit': 'PASS', 'config_sha256': sha256(path), 'rows': reviewed,
            'counts': {arm: dict(Counter(r['status'] for r in reviewed if r['arm'] == arm)) for arm in ['act', 'author']},
            'execution_control_pass': not blocked and all(r['status'] != 'TIMEOUT' for r in reviewed),
            'trust': 'structural/input/terminal consistency, not independent SAFE proof; UNSAFE requires separate original-model replay',
            'numerical_guarantees_equated': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = audit(a.config)
    write(a.output, value)
    print(value['audit'], value['counts'])
