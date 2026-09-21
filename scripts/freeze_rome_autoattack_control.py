import json
from pathlib import Path
from recent_moe_deployment import sha256, git_identity

if __name__ == '__main__':
    root = Path('/data1/Kane/MOE')
    repo = root / 'baselines/recent_moe_20260921/rome'
    ckpt = root / 'baseline_weights/rome_20260921/MAX_apgd_cifar10.pth'
    env = root / 'baseline_runs/rome_eval_env_20260921.json'
    identity = git_identity(repo)
    files = [ckpt, env, root / 'baseline_data/metamoe_20260921/cifar-10-batches-py/test_batch',
             Path('scripts/rome_autoattack_control.py'), Path('scripts/recent_moe_deployment.py')]
    aa_root = root / 'envs/rome-eval-cpu-20260921/lib/python3.12/site-packages/autoattack'
    files += list(aa_root.glob('*.py')) + [repo / p for p in identity['tracked_sha256'] if p.endswith('.py')]
    cfg = {'schema': 1, 'protocol': 'rome_native_standard_linf_deployment_control_r1',
        'repo': str(repo), 'commit': identity['head'], 'checkpoint': str(ckpt),
        'python': str(root / 'envs/rome-eval-cpu-20260921/bin/python'), 'environment': str(env),
        'data_root': str(root / 'baseline_data/metamoe_20260921'),
        'output_root': str(root / 'baseline_runs/rome_autoattack_20260921_r1'),
        'files': {str(p): sha256(p) for p in files}, 'index': 0, 'seed': 0,
        'epsilon': 8 / 255, 'norm': 'Linf', 'version': 'standard', 'batch_size': 1,
        'total_seconds': 600, 'witness_tolerance': 1e-7,
        'autoattack_commit': 'a39220048b3c9f2cca9a4d3a54604793c68eca7e',
        'source_s_b': [4, 6], 'state_compatibility': 'gate_proj_dim0 inference-only, strict prediction state',
        'selection': 'same raw-order index0 used for deployment, NOT new holdout',
        'stop': 'no retries, sample/attack/budget change or trained recipe selection',
        'grade': 'EMPIRICAL_ATTACK_ONLY'}
    out = Path('configs/recent_moe/rome_autoattack_control_r1.json')
    with out.open('x') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    print(sha256(out))
