"""Launch the frozen supervised job; no git writes or automatic retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dual_rs_epoch_pipeline import validate_protocol
from recent_moe_deployment import sha256

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'feat/moe-route-verification':
        raise ValueError('wrong branch')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('freeze requires clean checkout')
    cfg = json.loads(a.config.read_text())
    if cfg['mode'] != 'training':
        raise ValueError('not the frozen training protocol')
    validate_protocol(cfg)
    if Path(cfg['output_root']).exists():
        raise ValueError('run exists; no implicit resume/retry')
    root = Path(cfg['launch_root'])
    root.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).with_name('dual_rs_epoch_pipeline.py').resolve()),
               '--config', str(a.config.resolve())]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (root / 'stdout.txt').open('x') as out, (root / 'stderr.txt').open('x') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
    launch = {'pid': process.pid, 'command': command, 'config_sha256': sha256(a.config),
        'launch_unix': time.time(), 'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'deadline_seconds': cfg['total_seconds'], 'automatic_git_push': False,
        'output_root': cfg['output_root'], 'terminal': cfg['output_root'] + '/outer_terminal.json'}
    with (root / 'launch.json').open('x') as f:
        json.dump(launch, f, indent=2)
        f.write('\n')
    print(json.dumps(launch, indent=2))
