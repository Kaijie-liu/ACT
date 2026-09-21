"""Freeze final-epoch author-recipe training AFTER both real resume controls."""
import json
from pathlib import Path
import numpy as np
from recent_moe_deployment import sha256
from dual_rs_training_state import digest

if __name__ == '__main__':
    control = Path('configs/recent_moe/dual_rs_epoch_control_r1.json')
    cfg = json.loads(control.read_text())
    archive = Path('docs/dual_rs_epoch_control_archive_20260921_r1.json')
    audited = json.loads(archive.read_text())
    if audited['audit'] != 'PASS' or not audited['exact_epoch_continuation']:
        raise ValueError('epoch prerequisite not passed')
    parent = json.loads(Path(cfg['parent_config']).read_text())
    cfg.update(protocol='dual_rs_log_domain_author_recipe_training_r1', mode='training', epochs=90,
        total_seconds=43200, prefix_count=None,
        output_root='/data1/Kane/MOE/baseline_runs/dual_rs_selector_training_20260921_r1',
        launch_root='/data1/Kane/MOE/baseline_runs/dual_rs_selector_training_launch_20260921_r1',
        epoch_control_archive=str(archive), epoch_control_archive_sha256=sha256(archive),
        selection='ALL author eligible max-radius-nonzero train/test rows; native train shuffle and test order',
        checkpoint_selection='epoch90 only, regardless of accuracy or future certification',
        resume='no automatic retry; complete-epoch state available for separately authorized continuation')
    cfg['execution_files']['scripts/launch_dual_rs_frozen_training.py'] = sha256('scripts/launch_dual_rs_frozen_training.py')
    for split, path in [('train', parent['label_table']), ('test', cfg['test_label_table'])]:
        indices = np.flatnonzero(np.load(path, allow_pickle=False)[:, -2] != 0).tolist()
        cfg[f'{split}_count'] = len(indices)
        cfg[f'{split}_indices_sha256'] = digest(indices)
    cfg['certification_after_training'] = {
        'status': 'SCIENTIFIC_RECIPE_FIXED_NOT_AUTOMATICALLY_LAUNCHED',
        'test_indices': [0, 1], 'selection': 'raw test prefix, no clean-correct/certificate filter',
        'selector_checkpoint': 'training/epoch090.pt after saved-state audit',
        'noise_selector': 1.0, 'sigma_candidates': [.25, .5, 1.0],
        'N0_each_stage': 100, 'N_each_stage': 10000, 'alpha_each_stage': .0005,
        'confidence_scope': 'per-input two-stage failure probability <= .001 by union bound',
        'radius': 'minimum of selector and chosen classifier L2 radii, zero on abstention',
        'certification_precision': 'native CUDA autocast; must bind implementation before launch',
        'not_directly_comparable_to': 'deterministic HZ-policy L_inf results'}
    out = Path('configs/recent_moe/dual_rs_training_r1.json')
    with out.open('x') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    print(json.dumps({'config_sha256': sha256(out), 'train': cfg['train_count'], 'test': cfg['test_count']}))
