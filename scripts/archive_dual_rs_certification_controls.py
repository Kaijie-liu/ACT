import argparse
import json
from pathlib import Path
from recent_moe_deployment import sha256


def audit():
    base = Path('/data1/Kane/MOE/baseline_runs')
    records = []
    for name, expected in [('dual_rs_certification_controls_20260921_r1', 'COMPLETED'),
                           ('dual_rs_certification_early_bind_rejection_20260921_r1', 'ERROR')]:
        root = base / name
        receipt = json.loads((root / 'receipt.json').read_text())
        if receipt['status'] != expected or not receipt['source_unchanged']:
            raise ValueError('control terminal mismatch')
        for kind in ['stdout', 'stderr']:
            if sha256(root / (kind + '.txt')) != receipt[kind + '_sha256']:
                raise ValueError('control log changed')
        log = (root / 'stderr.txt').read_text()
        if expected == 'COMPLETED':
            if 'Ran 6 tests' not in log or not log.rstrip().endswith('OK'):
                raise ValueError('native differential not fully passed')
        elif 'training not landed; keep recipe-only status' not in log:
            raise ValueError('unexpected rejection reason')
        records.append({'name': name, 'observed_status': receipt['status'],
            'expected_status': expected, 'receipt_sha256': sha256(root / 'receipt.json'),
            'execution_seconds': receipt['execution_including_preflight_seconds'],
            'with_postflight_seconds': receipt['total_with_postflight_seconds']})
    recipe = Path('configs/recent_moe/dual_rs_certification_recipe_r1.json')
    value = json.loads(recipe.read_text())
    for path, h in value['files'].items():
        if sha256(path) != h:
            raise ValueError('prepared recipe identity')
    return {'status': 'PREPARATION_CONTROLS_PASS', 'records': records,
        'recipe_sha256': sha256(recipe),
        'native_control': 'same certify output AND RNG after instrumentation; Clopper-Pearson differential, abstention/count/partial controls',
        'execution_status_at_preparation': 'BLOCKED_UNTIL_FINAL_EPOCH_LANDS',
        'real_certification_executed': False, 'formal_SAFE': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    value = audit()
    if a.check:
        if json.loads(a.output.read_text()) != value:
            raise ValueError('archive mismatch')
    else:
        with a.output.open('x') as f:
            json.dump(value, f, indent=2)
            f.write('\n')
    print('PASS')
