"""Same control, separately frozen NumPy1.x dependency compatibility."""
import json
from pathlib import Path
from robust_experts_workflow_control import compose
from recent_moe_deployment import sha256


if __name__ == '__main__':
    parent = Path('configs/recent_moe/robust_experts_workflow_r1.json')
    cfg = json.loads(parent.read_text())
    for p, h in cfg['files'].items():
        if sha256(p) != h:
            raise ValueError('R1 identity changed')
    old_failure = Path('docs/robust_experts_workflow_archive_20260921_r1.json')
    archived = json.loads(old_failure.read_text())
    if archived['terminal']['status'] != 'ERROR' or 'np.Inf' not in archived['stderr_tail']:
        raise ValueError('R1 failure attribution mismatch')
    cfg.update(protocol='robust_experts_native_pgd_workflow_control_r2',
        python='/data1/Kane/MOE/envs/robust-experts-workflow-cpu-20260921-r2/bin/python',
        environment='/data1/Kane/MOE/baseline_runs/robust_experts_workflow_env_20260921_r2.json',
        output_root='/data1/Kane/MOE/baseline_runs/robust_experts_workflow_20260921_r2',
        parent_config=str(parent), parent_config_sha256=sha256(parent))
    _, _, cfg['composed_config'] = compose(Path(cfg['repo']), Path(cfg['output_root']), Path(cfg['data_root']))
    cfg['execution_compatibility'].append('R2 only: NumPy1.26.4/OpenCV4.11/tifffile2024.8.30 for original Lightning np.Inf; frozen R1 retained; no model or attack changes')
    extra = [parent, old_failure, Path(cfg['environment']), Path(__file__)]
    cfg['files'].update({str(p): sha256(p) for p in extra})
    out = Path('configs/recent_moe/robust_experts_workflow_r2.json')
    with out.open('x') as f:
        json.dump(cfg, f, indent=2, allow_nan=False)
        f.write('\n')
    print(sha256(out))
