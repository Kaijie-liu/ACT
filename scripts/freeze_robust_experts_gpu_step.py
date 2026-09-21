"""Commit this two-architecture GPU control before executing it."""
import json
from pathlib import Path
from recent_moe_deployment import sha256
from recent_moe_env_inventory import inventory
from robust_experts_workflow_control import write


if __name__=='__main__':
    parent=json.loads(Path('configs/recent_moe/robust_experts_resume_r1.json').read_text())
    recipe=Path('configs/recent_moe/robust_experts_paper_training_recipe_r1.json')
    envroot=Path('/data1/Kane/MOE/baseline_runs/robust_experts_gpu_environment_20260922_r1')
    env=json.loads((envroot/'terminal.json').read_text())
    if env['status'] != 'DEPENDENCIES_READY' or not env['base_unchanged']:
        raise ValueError('GPU environment gate')
    py='/data1/Kane/MOE/envs/robust-experts-workflow-blackwell-20260922-r1/bin/python'
    files=dict(parent['files'])
    for p in [recipe,envroot/'terminal.json',Path(__file__),
        Path('scripts/robust_experts_gpu_step_control.py'),Path('tests/test_robust_experts_gpu_step.py')]:
        files[str(p)]=sha256(p)
    write('configs/recent_moe/robust_experts_gpu_step_r1.json',{
        'protocol':'robust_experts_native_full_batch_gpu_step_r1','repo':parent['repo'],
        'data_root':parent['data_root'],'recipes':json.loads(recipe.read_text())['recipes'],
        'files':files,'environment':inventory([py]),'python':py,'seconds_each':600,
        'output_root':'/data1/Kane/MOE/baseline_runs/robust_experts_gpu_step_20260922_r1',
        'gpu_gate':'free>=48GiB, own allocation cap40%; no eviction/fallback/retry',
        'training_launched':False})
