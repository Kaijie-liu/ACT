"""Register a two-epoch prefix control before a long author-recipe run."""
import json
from pathlib import Path
import numpy as np
from recent_moe_deployment import sha256
from dual_rs_training_state import digest


if __name__ == '__main__':
    parent = Path('configs/recent_moe/dual_rs_training_control_r2.json')
    p = json.loads(parent.read_text())
    archive = Path('docs/dual_rs_training_control_archive_20260921_r2.json')
    test_table = str(Path(p['label_table']).with_name('0.250_0.500_1.000_test.npy'))
    cfg = {'schema': 1, 'protocol': 'dual_rs_completed_epoch_control_r1', 'mode': 'epoch_control',
        'parent_config': str(parent), 'parent_config_sha256': sha256(parent),
        'step_control_archive': str(archive), 'step_control_archive_sha256': sha256(archive),
        'epochs': 2, 'prefix_count': 256, 'train_count': 256, 'test_count': 256,
        'test_label_table': test_table, 'total_seconds': 300,
        'output_root': '/data1/Kane/MOE/baseline_runs/dual_rs_epoch_control_20260921_r1',
        'execution_files': {f: sha256(f) for f in ['scripts/dual_rs_epoch_pipeline.py',
            'scripts/audit_dual_rs_epochs.py', 'scripts/dual_rs_training_control_r2.py',
            'scripts/dual_rs_training_state.py', 'scripts/audit_dual_rs_training_control.py']},
        'selection': 'first256 author max-radius-nonzero rows in each split; train shuffled per epoch, test ordered',
        'checkpoint_selection': 'completed epoch2; no best-accuracy/certification selection',
        'resume': 'fresh process from completed epoch1, then native train/test/scheduler epoch2',
        'failure_rule': 'stop, retain all partials, no retry or altered gate'}
    for split, path in [('train', p['label_table']), ('test', test_table)]:
        indices = np.flatnonzero(np.load(path, allow_pickle=False)[:, -2] != 0)[:256].tolist()
        cfg[f'{split}_indices_sha256'] = digest(indices)
    out = Path('configs/recent_moe/dual_rs_epoch_control_r1.json')
    with out.open('x') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    print(sha256(out))
