"""Freeze native two-epoch continuation; no full training automatically starts."""
import json
from pathlib import Path
from recent_moe_deployment import sha256


if __name__ == '__main__':
    parent = Path('configs/recent_moe/robust_experts_workflow_r3.json')
    cfg = json.loads(parent.read_text())
    for p, h in cfg['files'].items():
        if sha256(p) != h:
            raise ValueError('parent binding changed')
    cfg.update(protocol='robust_experts_native_two_epoch_resume_r1',
        parent_config=str(parent), parent_config_sha256=sha256(parent),
        output_root='/data1/Kane/MOE/baseline_runs/robust_experts_resume_20260922_r1',
        epochs=2, workers=0, total_seconds=600,
        phases=['reference', 'resume', 'audit'],
        scope='native E4/k1, two real epochs one batch2 each; full SGD/PolyLR/RNG and next augmentation equality',
        continuation='fresh process from reference epoch00; epoch01 must match; no retry on mismatch')
    cfg.pop('composed_config', None)
    for name in ['scripts/robust_experts_resume_control.py', 'scripts/freeze_robust_experts_resume.py',
                 'scripts/audit_dual_rs_training_control.py', 'tests/test_robust_experts_resume.py', str(parent)]:
        cfg['files'][name] = sha256(name)
    if Path(cfg['output_root']).exists():
        raise ValueError('run directory exists')
    with Path('configs/recent_moe/robust_experts_resume_r1.json').open('x') as f:
        json.dump(cfg, f, indent=2, allow_nan=False)
        f.write('\n')
    print('FROZEN_NOT_EXECUTED')
