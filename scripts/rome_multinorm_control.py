"""Frozen multi-norm native AutoAttack worker; incomplete attacks stay unknown."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from recent_moe_deployment import sha256, git_identity, supervise
from recent_moe_env_inventory import inventory


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def validate(cfg):
    if (cfg['norm'] not in ['Linf', 'L1', 'L2'] or
            (cfg['version'], cfg['batch_size'], cfg['source_s_b']) != ('standard', 1, [4, 6])):
        raise ValueError('unsupported frozen attack/source settings')
    if not math.isfinite(cfg['epsilon']) or cfg['epsilon'] != {'Linf': 8/255, 'L1': 12., 'L2': .5}[cfg['norm']]:
        raise ValueError('invalid epsilon')
    for file, h in cfg['files'].items():
        if sha256(file) != h:
            raise ValueError('frozen file identity: ' + file)
    identity = git_identity(cfg['repo'])
    if identity['status'] or identity['head'] != cfg['commit']:
        raise ValueError('source identity')
    if inventory([cfg['python']]) != json.loads(Path(cfg['environment']).read_text()):
        raise ValueError('environment changed')


def terminal(receipt, result):
    accepted = bool(result and result.get('status') == 'ATTACK_EVALUATION_COMPLETED'
                    and receipt['status'] == 'COMPLETED')
    return {'status': result['status'] if accepted else
            ('ERROR' if receipt['status'] == 'COMPLETED' else receipt['status']),
            'result_accepted': accepted,
            'execution_seconds': receipt['execution_including_preflight_seconds'],
            'total_with_postflight_seconds': receipt['total_with_postflight_seconds'],
            'postflight_in_execution_budget': False,
            'evidence_grade': 'EMPIRICAL_ATTACK_ONLY' if accepted else 'NONE'}


def worker(cfg, config):
    start = time.monotonic()
    validate(cfg)
    sys.path.insert(0, cfg['repo'])
    import torch
    from torch import nn
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from models import build_model, inspect_checkpoint
    from autoattack import AutoAttack
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    payload = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in payload['state_dict'].items()}
    spec = inspect_checkpoint(state)
    model = build_model('cifar10', pretrained=False, r=spec['r'], lora_alpha=spec['r'],
                        num_lora=spec['num_lora'], mlp_gate=spec['mlp_gate'],
                        gate_proj_dim=0, s=4., b=6.)
    model.load_state_dict(state, strict=True)
    model.eval()
    class LogitsOnly(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
        def forward(self, x):
            return self.wrapped(x)[0]
    wrapped = LogitsOnly(model).eval()
    dataset = CIFAR10(cfg['data_root'], train=False, download=False, transform=ToTensor())
    x, label = dataset[cfg['index']]
    x = x.unsqueeze(0)
    y = torch.tensor([label])
    root = Path(cfg['output_root'])
    with torch.no_grad():
        clean = wrapped(x)
    if not torch.isfinite(clean).all():
        raise ValueError('nonfinite clean output')
    setup = time.monotonic() - start
    write(root / 'prepared.json', {'config_sha256': sha256(config), 'index': cfg['index'], 'label': label,
        'clean_prediction': int(clean.argmax(1)), 'setup_seconds': setup,
        'input_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest()})
    attacker = AutoAttack(wrapped, norm=cfg['norm'], eps=cfg['epsilon'], version='standard',
                          seed=cfg['seed'], device='cpu', log_path=str(root / 'autoattack.log'))
    adv = attacker.run_standard_evaluation(x, y, bs=1)
    if not torch.isfinite(adv).all() or adv.shape != x.shape or adv.min() < 0 or adv.max() > 1:
        raise ValueError('invalid attack input')
    delta = (adv.double() - x.double()).reshape(-1)
    distance = float(delta.abs().max() if cfg['norm'] == 'Linf' else
                     torch.linalg.vector_norm(delta, ord=1 if cfg['norm'] == 'L1' else 2))
    if distance > cfg['epsilon'] + cfg['witness_tolerance']:
        raise ValueError('attack outside registered numerical budget')
    with torch.no_grad():
        output = wrapped(adv)
    if not torch.isfinite(output).all():
        raise ValueError('nonfinite full-model replay')
    torch.save({'clean': x, 'adversarial': adv, 'label': y}, root / 'witness.pt')
    write(root / 'result.json', {'status': 'ATTACK_EVALUATION_COMPLETED', 'config_sha256': sha256(config),
        'index': cfg['index'], 'label': label, 'clean_prediction': int(clean.argmax(1)),
        'adversarial_prediction': int(output.argmax(1)), 'norm': cfg['norm'], 'norm_distance': distance,
        'empirically_correct_after_attack': bool(output.argmax(1).item() == label),
        'evidence_grade': 'EMPIRICAL_ATTACK_ONLY', 'formal_SAFE': False,
        'attacks': attacker.attacks_to_run, 'witness_sha256': sha256(root / 'witness.pt'),
        'setup_seconds': setup, 'attack_replay_save_seconds': time.monotonic() - start - setup,
        'worker_total_seconds': time.monotonic() - start,
        'semantics': 'original dense multilayer variable-weight logits, inference-only unused auxiliary head absent',
        'scope': 'frozen four-input three-norm deployment; not paper accuracy or ACT certification'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.worker:
        worker(cfg, a.config)
    else:
        root = Path(cfg['output_root'])
        receipt = supervise([cfg['python'], str(Path(__file__).resolve()), '--config', str(a.config.resolve()), '--worker'],
            str(Path(__file__).resolve().parents[1]), root, cfg['total_seconds'], 'AUTHOR_EMPIRICAL_ATTACK_CONTROL', cfg['repo'])
        result = json.loads((root / 'result.json').read_text()) if (root / 'result.json').exists() else None
        write(root / 'terminal.json', terminal(receipt, result))

