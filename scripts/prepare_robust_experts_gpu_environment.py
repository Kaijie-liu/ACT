"""Create a dedicated dependency overlay; NEVER modify its CUDA base or ACT env.

Explicit read-only inheritance avoids reinstalling the already working Blackwell
Torch stack. This is a named environment adaptation, not the author's original
environment. No GPU training or model query is launched here.
"""
import json
import os
from pathlib import Path
import subprocess
import time
from recent_moe_env_inventory import inventory
from robust_experts_workflow_control import write


if __name__ == '__main__':
    base = '/data1/Kane/MOE/envs/rt-er-blackwell/bin/python'
    root = Path('/data1/Kane/MOE/envs/robust-experts-workflow-blackwell-20260922-r1')
    record = Path('/data1/Kane/MOE/baseline_runs/robust_experts_gpu_environment_20260922_r1')
    if root.exists():
        raise ValueError('new environment only')
    record.mkdir(exist_ok=False)
    before = inventory([base])
    write(record/'base_before.json', before)
    start = time.monotonic()
    commands = [[base, '-m', 'venv', '--system-site-packages', str(root)],
        [str(root/'bin/python'), '-m', 'pip', 'install',
         'numpy==1.26.4', 'setuptools==68.2.2', 'pytorch-lightning==1.9.5',
         'lightning-bolts==0.7.0', 'torchmetrics==0.11.4', 'torchattacks==3.5.1',
         'albumentations==1.3.1', 'opencv-python-headless==4.11.0.86', 'tifffile==2024.8.30',
         'hydra-core==1.3.2', 'hydra-colorlog==1.2.0', 'python-dotenv==1.1.1',
         'clearml==1.18.0', 'wandb==0.23.1', 'einops==0.8.1', 'pathlib2==2.3.7.post1',
         'rich==14.3.3', 'timm==1.0.15'],
        [str(root/'bin/python'), '-m', 'pip', 'check']]
    env = dict(os.environ, PIP_CACHE_DIR=str(record/'pip_cache'), TMPDIR=str(record/'tmp'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    Path(env['TMPDIR']).mkdir()
    rows = []
    for i, command in enumerate(commands):
        with (record/f'step{i}.stdout').open('x') as out, (record/f'step{i}.stderr').open('x') as err:
            try:
                done = subprocess.run(command, env=env, stdout=out, stderr=err,
                    timeout=max(.01, 900-(time.monotonic()-start)))
                rows.append({'command': command, 'exit_code': done.returncode})
            except subprocess.TimeoutExpired:
                rows.append({'command': command, 'status': 'TIMEOUT'})
                break
        if done.returncode:
            break
    after = inventory([base])
    if before != after:
        raise ValueError('read-only base unexpectedly changed')
    accepted = len(rows) == len(commands) and all(r.get('exit_code') == 0 for r in rows)
    own = inventory([str(root/'bin/python')]) if (root/'bin/python').exists() else None
    write(record/'terminal.json', {'status': 'DEPENDENCIES_READY' if accepted else 'FAILED',
        'steps': rows, 'seconds': time.monotonic()-start, 'environment': own,
        'base_unchanged': True, 'base': before,
        'isolation': 'dedicated writable overlay with read-only CUDA base inheritance',
        'native_GPU_workflow_control': False, 'training_launched': False})
    print('DEPENDENCIES_READY' if accepted else 'FAILED')
