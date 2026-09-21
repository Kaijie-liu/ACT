"""Freeze four seed0 author-order inputs x3 native standard-AA norms."""
import json
from pathlib import Path
import torch
from recent_moe_deployment import sha256


if __name__ == '__main__':
    parent = Path('configs/recent_moe/rome_autoattack_control_r1.json')
    cfg = json.loads(parent.read_text())
    for name, h in cfg['files'].items():
        if sha256(name) != h:
            raise ValueError('parent file changed')
    indices = torch.randperm(10000, generator=torch.Generator().manual_seed(0))[:4].tolist()
    names = ['scripts/rome_multinorm_control.py', 'scripts/rome_multinorm_batch.py',
             'scripts/freeze_rome_multinorm.py', 'tests/test_rome_multinorm.py', str(parent)]
    files = {**cfg['files'], **{n: sha256(n) for n in names}}
    requests = []
    for index in indices:
        for norm, eps in [('Linf', 8/255), ('L1', 12.), ('L2', .5)]:
            dest = Path(f'configs/recent_moe/rome_multinorm_r1_{index}_{norm}.json')
            one = {**cfg, 'protocol': 'rome_four_input_three_norm_control_r1', 'index': index,
                   'norm': norm, 'epsilon': eps, 'files': files,
                   'output_root': f'/data1/Kane/MOE/baseline_runs/rome_multinorm_20260922_r1_{index}_{norm}',
                   'selection': 'author seed0 torch.randperm first4, before evaluation; no clean filtering'}
            if Path(one['output_root']).exists():
                raise ValueError('run exists')
            with dest.open('x') as f:
                json.dump(one, f, indent=2, allow_nan=False)
                f.write('\n')
            requests.append({'config': str(dest), 'sha256': sha256(dest)})
    batch = {'protocol': 'rome_native_three_norm_batch_r1', 'indices': indices, 'requests': requests,
        'output_root': '/data1/Kane/MOE/baseline_runs/rome_multinorm_20260922_r1_batch',
        'files': {**files, **{r['config']: r['sha256'] for r in requests}},
        'seconds_per_input_norm': 600, 'total_registered_requests': 12,
        'failure': 'ERROR stops new requests, TIMEOUT remains incomplete; no retry or parameter change',
        'grade': 'EMPIRICAL_ATTACK_ONLY', 'automatic_100_input_expansion': False}
    with Path('configs/recent_moe/rome_multinorm_batch_r1.json').open('x') as f:
        json.dump(batch, f, indent=2, allow_nan=False)
        f.write('\n')
    print('FROZEN_NOT_EXECUTED', indices)
