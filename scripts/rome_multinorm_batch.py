"""Frozen per-input/per-norm budgets; independent outcomes, no robust timeout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from recent_moe_deployment import sha256
from rome_autoattack_control import write


def summarize(rows, indices):
    result = []
    for index in indices:
        rr = [r for r in rows if r['index'] == index]
        if len(rr) != 3 or {r['norm'] for r in rr} != {'Linf', 'L1', 'L2'}:
            raise ValueError('incomplete registered roster')
        complete = all(r['status'] == 'ATTACK_EVALUATION_COMPLETED' for r in rr)
        broken = any(r.get('empirically_correct_after_attack') is False for r in rr)
        result.append({'index': index, 'status': 'ATTACK_FOUND' if broken else
                       ('NO_ATTACK_FOUND_ALL_THREE' if complete else 'INCOMPLETE_NOT_ROBUST'),
                       'all_three_completed': complete})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    for name, h in cfg['files'].items():
        if sha256(name) != h:
            raise ValueError('batch identity')
    root = Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    rows, blocked = [], False
    for item in cfg['requests']:
        one = json.loads(Path(item['config']).read_text())
        row = {'index': one['index'], 'norm': one['norm'], 'config': item['config']}
        if blocked:
            row['status'] = 'NOT_STARTED_AFTER_ERROR'
        else:
            code = subprocess.run([sys.executable, 'scripts/rome_multinorm_control.py', '--config', item['config']]).returncode
            folder = Path(one['output_root'])
            path = folder / 'terminal.json'
            terminal = json.loads(path.read_text()) if path.exists() else {'status': 'ERROR'}
            row.update(status=terminal['status'], terminal=terminal)
            if code != 0:
                row['status'] = 'ERROR'
            if row['status'] == 'ATTACK_EVALUATION_COMPLETED':
                out = json.loads((folder / 'result.json').read_text())
                row.update(empirically_correct_after_attack=out['empirically_correct_after_attack'],
                           result_sha256=sha256(folder / 'result.json'))
            blocked = row['status'] not in ['ATTACK_EVALUATION_COMPLETED', 'TIMEOUT']
        rows.append(row)
        write(root / f'progress{len(rows):02d}.json', row)
    write(root / 'summary.json', {'config_sha256': sha256(a.config), 'rows': rows,
        'union': summarize(rows, cfg['indices']), 'scope': 'EMPIRICAL_DEPLOYMENT_ONLY',
        'formal_SAFE': False, 'frozen_input_denominator': len(cfg['indices'])})
