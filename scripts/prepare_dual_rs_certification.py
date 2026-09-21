"""Freeze execution recipe now; bind ONLY the audited final epoch after landing.

No automatic launch and no substitute early/best checkpoint. --bind-landed
rejects incomplete training without creating an execution config.
"""
import argparse
import json
from pathlib import Path
from recent_moe_deployment import sha256


def recipe():
    training = Path('configs/recent_moe/dual_rs_training_r1.json')
    cfg = json.loads(training.read_text())
    parent_path = Path(cfg['parent_config'])
    parent = json.loads(parent_path.read_text())
    if cfg['certification_after_training']['test_indices'] != [0, 1]:
        raise ValueError('earlier scientific recipe changed')
    weights = Path('/data1/Kane/MOE/baseline_weights/dual_rs_20260921')
    manifest = json.loads((weights / 'manifest.json').read_text())
    control = Path('/data1/Kane/MOE/baseline_runs/dual_rs_certification_controls_20260921_r1')
    receipt = json.loads((control / 'receipt.json').read_text())
    if (receipt['status'] != 'COMPLETED' or not receipt['source_unchanged'] or
            sha256(control / 'stderr.txt') != receipt['stderr_sha256'] or
            'Ran 6 tests' not in (control / 'stderr.txt').read_text() or
            not (control / 'stderr.txt').read_text().rstrip().endswith('OK')):
        raise ValueError('native certification controls did not pass without skips')
    files = [training, parent_path, weights / 'manifest.json',
        Path('scripts/dual_rs_certification_pipeline.py'), Path('scripts/dual_rs_certification_evidence.py'),
        Path('scripts/audit_dual_rs_certification.py'), Path('scripts/prepare_dual_rs_certification.py'),
        Path('scripts/recent_moe_deployment.py'), Path('scripts/recent_moe_env_inventory.py'),
        Path('scripts/dual_rs_training_control.py'), Path('scripts/dual_rs_training_state.py'),
        Path('scripts/dual_rs_epoch_pipeline.py'), Path('tests/test_dual_rs_certification.py'),
        control / 'receipt.json', control / 'stderr.txt']
    files += [weights / item['filename'] for item in manifest['files']]
    return {'schema': 'dual_rs_certification_recipe_v1', 'training_config': str(training),
        'training_parent': str(parent_path), 'author_repo': parent['author_repo'], 'python': parent['python'],
        'vit': str(weights / 'vit'), 'files': {str(p): sha256(p) for p in files},
        'test_indices': [0, 1], 'n0': 100, 'n': 10000, 'alpha': .0005,
        'selector_sigma': 1., 'sigma_candidates': [.25, .5, 1.], 'batch_size': 16,
        'total_seconds': 7200, 'epoch': 90, 'minimum_free_gpu_gib': 24, 'memory_fraction': .25,
        'seeds': {'0': [202609210, 202609211], '1': [202609212, 202609213]},
        'precision': 'native CUDA autocast, no fallback or parameter search',
        'selection': 'raw CIFAR test0,1 regardless of clean/selector/cert outcome',
        'accounting': 'one7200s outer includes setup, sampling, saves, child imports and audit; source postflight reported separately',
        'failure': 'retain partial counts, no retry/substitution/early epoch',
        'automatic_launch': False}


def bind(cfg, recipe_path):
    for path, h in cfg['files'].items():
        if sha256(path) != h:
            raise ValueError('recipe/source identity changed')
    train = json.loads(Path(cfg['training_config']).read_text())
    root = Path(train['output_root'])
    terminal = root / 'outer_terminal.json'
    if not terminal.exists() or json.loads(terminal.read_text())['status'] != 'TRAINING_LANDED':
        raise ValueError('training not landed; keep recipe-only status')
    audit = root / 'audit.json'
    record = json.loads(audit.read_text())
    final = root / 'training/epoch090.pt'
    info = root / 'training/epoch090.json'
    if (record['audit'] != 'PASS' or record['epochs'] != 90 or
            record['config_sha256'] != sha256(cfg['training_config']) or
            json.loads(info.read_text())['file_sha256'] != sha256(final)):
        raise ValueError('final epoch identity/audit failed')
    return {**cfg, 'schema': 'dual_rs_final_epoch_certification_execution_v1',
        'recipe_path': str(recipe_path), 'recipe_sha256': sha256(recipe_path),
        'selector_checkpoint': str(final),
        'files': {**cfg['files'], **{str(p): sha256(p) for p in [terminal, audit, final, info, recipe_path]}},
        'output_root': '/data1/Kane/MOE/baseline_runs/dual_rs_certification_20260921_r1'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bind-landed', action='store_true')
    a = p.parse_args()
    target = Path('configs/recent_moe/dual_rs_certification_recipe_r1.json')
    if a.bind_landed:
        value = bind(json.loads(target.read_text()), target)
        target = Path('configs/recent_moe/dual_rs_certification_execution_r1.json')
    else:
        value = recipe()
    with target.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    print(sha256(target))
