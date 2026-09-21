"""Freeze one raw-order full MetaMoE compatibility request, not a selected success."""
import json
from pathlib import Path
import subprocess
from recent_moe_deployment import sha256, git_identity


if __name__ == '__main__':
    root = Path('/data1/Kane/MOE')
    repo = root / 'baselines/recent_moe_20260921/metamoe'
    checkpoint = repo / 'paper/artifacts/MoE_CNN_AT/meta_moe_ultra_verifiable_cnn_best_RT_eps0.03137.pth'
    data = root / 'baseline_data/metamoe_20260921'
    files = [checkpoint, data / 'cifar-10-batches-py/test_batch', data / 'cifar-10-batches-py/batches.meta',
             Path('scripts/metamoe_full_intake.py'), Path('scripts/recent_moe_deployment.py'),
             Path('act/back_end/moe/class_separated_top1.py')]
    files += [Path(p) for p in subprocess.check_output(['git', 'ls-files', 'act'], text=True).splitlines()
              if p.endswith('.py')]
    cfg = {'schema': 1, 'protocol': 'metamoe_full_class_separated_intake_r1',
        'repo': str(repo), 'commit': git_identity(repo)['head'], 'checkpoint': str(checkpoint),
        'data_root': str(data), 'python': '/data1/Kane/miniconda3/envs/act-py312/bin/python',
        'output_root': str(root / 'baseline_runs/metamoe_full_intake_20260921_r1'),
        'files': {str(p): sha256(p) for p in sorted(set(files))},
        'requests': [{'id': 'cifar10_test_0', 'index': 0, 'selection': 'first raw-order CIFAR10 test, no correct/routing filter'}],
        'request_seconds': 300, 'normalized_epsilon': 2 / 255, 'classification_margin': 1e-7,
        'solver_gate': 'unchanged HZ numerical policy; all tie-legal branches plus selected-score nonzero',
        'stop': 'no retry, threshold or input change; outer terminal wins',
        'scope': 'CPU float64 author state snapshot, original complete forward controls; not full-source float32 certificate'}
    out = Path('configs/recent_moe/metamoe_full_intake_r1.json')
    with out.open('x') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    print(sha256(out))
