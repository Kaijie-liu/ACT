"""Independent saved-witness replay on strict public RoME prediction state."""
import argparse
import json
from pathlib import Path
import sys
import time
from recent_moe_deployment import sha256


def audit(config):
    start = time.monotonic()
    cfg = json.loads(config.read_text())
    root = Path(cfg['output_root'])
    for p, h in cfg['files'].items():
        if sha256(p) != h:
            raise ValueError('frozen identity changed')
    receipt = json.loads((root / 'receipt.json').read_text())
    terminal = json.loads((root / 'terminal.json').read_text())
    if not receipt['source_unchanged']:
        raise ValueError('author source changed')
    for kind in ['stdout', 'stderr']:
        if sha256(root / (kind + '.txt')) != receipt[kind + '_sha256']:
            raise ValueError('execution log changed')
    if receipt['status'] != 'COMPLETED' or not terminal['result_accepted']:
        return {'audit': 'RECORDED_NON_COMPLETION', 'status': receipt['status'],
                'receipt_sha256': sha256(root / 'receipt.json'), 'formal_SAFE': False}
    result = json.loads((root / 'result.json').read_text())
    if result['config_sha256'] != sha256(config) or sha256(root / 'witness.pt') != result['witness_sha256']:
        raise ValueError('result/witness identity')
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    torch.set_num_threads(2)
    sys.path.insert(0, cfg['repo'])
    from models import build_model, inspect_checkpoint
    payload = torch.load(cfg['checkpoint'], weights_only=True, map_location='cpu')
    raw = payload['state_dict']
    state = {k.removeprefix('module.'): v for k, v in raw.items()}
    if len(raw) != len(state):
        raise ValueError('ambiguous prefix')
    spec = inspect_checkpoint(state)
    model = build_model('cifar10', pretrained=False, r=spec['r'], num_lora=spec['num_lora'],
        lora_alpha=spec['r'], mlp_gate=spec['mlp_gate'], gate_proj_dim=0, s=4., b=6.).eval()
    model.load_state_dict(state, strict=True)
    witness = torch.load(root / 'witness.pt', weights_only=True, map_location='cpu')
    x, label = CIFAR10(cfg['data_root'], train=False, download=False, transform=ToTensor())[cfg['index']]
    clean, adv = witness['clean'], witness['adversarial']
    if (not torch.equal(clean, x.unsqueeze(0)) or witness['label'].tolist() != [label]
            or adv.shape != clean.shape or not torch.isfinite(adv).all() or adv.min() < 0 or adv.max() > 1):
        raise ValueError('input/label/domain mismatch')
    delta = (adv.double() - clean.double()).reshape(-1)
    distance = float(delta.abs().max() if cfg['norm'] == 'Linf' else
                     torch.linalg.vector_norm(delta, ord=1 if cfg['norm'] == 'L1' else 2))
    if result['norm'] != cfg['norm'] or result['norm_distance'] != distance:
        raise ValueError('norm identity/distance mismatch')
    if distance > cfg['epsilon'] + cfg['witness_tolerance']:
        raise ValueError('outside registered NUMERICAL attack tolerance')
    with torch.no_grad():
        a, b = model(clean)[0], model(adv)[0]
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('nonfinite full-model replay')
    if int(a.argmax(1)) != result['clean_prediction'] or int(b.argmax(1)) != result['adversarial_prediction']:
        raise ValueError('independent prediction mismatch')
    return {'audit': 'INDEPENDENT_FULL_MODEL_REPLAY_PASS', 'result': result,
        'config_sha256': sha256(config), 'receipt_sha256': sha256(root / 'receipt.json'),
        'registered_norm_distance_float64': distance,
        'inside_requested_epsilon_in_float64_check': distance <= cfg['epsilon'],
        'excess_over_requested_epsilon': max(0., distance-cfg['epsilon']),
        'registered_numerical_tolerance': cfg['witness_tolerance'],
        'execution_seconds': terminal['execution_seconds'],
        'execution_with_postflight_seconds': terminal['total_with_postflight_seconds'],
        'independent_replay_seconds_separate_from_attack_budget': time.monotonic()-start,
        'formal_SAFE': False,
        'scope': 'single-input empirical deployment control; tolerance-accepted attack is NOT exact-box UNSAFE'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = audit(a.config)
    with a.output.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    print('Saved attack audit complete')

