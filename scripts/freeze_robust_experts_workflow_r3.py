"""Freeze an explicitly changed APGD mode-restoration variant, not native R2."""
import json
from pathlib import Path
import subprocess
from robust_experts_workflow_control import compose
from recent_moe_deployment import sha256


if __name__ == '__main__':
    parent = Path('configs/recent_moe/robust_experts_workflow_r2.json')
    cfg = json.loads(parent.read_text())
    for p, h in cfg['files'].items():
        if sha256(p) != h:
            raise ValueError('R2 identity changed')
    diagnosis = Path('docs/robust_experts_saved_state_audit_20260921_r2.json')
    if json.loads(diagnosis.read_text())['audit'] != 'SAVED_CHECKPOINT_PREDICTION_MISMATCH':
        raise ValueError('recorded R2 mismatch missing')
    old_repo = Path(cfg['repo'])
    repo = Path('/data1/Kane/MOE/baselines/robust_experts_compat_20260921_r3')
    patch = Path('configs/recent_moe/robust_experts_eval_mode_r3.patch')
    if subprocess.check_output(['git', '-C', str(repo), 'diff', '--'], text=True) != patch.read_text():
        raise ValueError('R3 patch mismatch')
    controls = Path('/data1/Kane/MOE/baseline_runs/robust_experts_apgd_mode_controls_20260921_r3/receipt.json')
    if json.loads(controls.read_text())['status'] != 'COMPLETED':
        raise ValueError('APGD mode controls not passed')
    cfg.update(protocol='robust_experts_eval_mode_workflow_control_r3', repo=str(repo),
        output_root='/data1/Kane/MOE/baseline_runs/robust_experts_workflow_20260921_r3',
        parent_config=str(parent), parent_config_sha256=sha256(parent))
    _, _, cfg['composed_config'] = compose(repo, Path(cfg['output_root']), Path(cfg['data_root']))
    new_source = {str(repo/Path(p).relative_to(old_repo)): sha256(repo/Path(p).relative_to(old_repo))
        for p in cfg['files'] if Path(p).is_relative_to(old_repo)}
    cfg['files'].update(new_source)
    for p in [parent, diagnosis, patch, controls, Path(__file__),
              Path('tests/test_robust_experts_apgd_mode.py'), Path('scripts/audit_robust_experts_saved_state.py')]:
        cfg['files'][str(p)] = sha256(p)
    cfg['execution_compatibility'].append('R3 semantic variant: APGD wrapper inherits caller model.training flag so torchattacks does not restore eval child to train; native R2 state drift preserved; gate/loss/PGD/optimizer unchanged')
    out = Path('configs/recent_moe/robust_experts_workflow_r3.json')
    with out.open('x') as f:
        json.dump(cfg, f, indent=2, allow_nan=False)
        f.write('\n')
    print(sha256(out))
