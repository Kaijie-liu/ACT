"""Restricted full RoME public-state intake; no attack or accuracy table."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    checkpoint = Path(manifest['path'])
    if hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest() != manifest['local_sha256']:
        raise ValueError('public file identity')
    sys.path.insert(0, str(a.repo.resolve()))
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from models import build_model, inspect_checkpoint
    from act.back_end.moe.layered_author_intake import LayeredAuthorIntake
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):
        raise ValueError('unreviewed checkpoint globals')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    raw = payload['state_dict'] if 'state_dict' in payload else payload
    state = {k.removeprefix('module.'): v for k, v in raw.items()}
    if len(state) != len(raw):
        raise ValueError('ambiguous module prefix remapping')
    spec = inspect_checkpoint(state)
    if spec['num_classes'] != 10 or not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError('invalid public prediction state')
    model = build_model('cifar10', pretrained=False, r=spec['r'], num_lora=spec['num_lora'],
                        lora_alpha=spec['r'], mlp_gate=spec['mlp_gate'], s=4.0, b=6.0)
    # No silent strict=False despite the author's permissive convenience loader.
    missing = sorted(set(model.state_dict()) - set(state))
    unexpected = sorted(set(state) - set(model.state_dict()))
    if missing or unexpected:
        print(json.dumps({'status': 'STATE_MAPPING_BLOCKED', 'spec': spec,
                          'missing': missing, 'unexpected': unexpected}, indent=2), flush=True)
        raise ValueError('full state mapping requires a separate reviewed adaptation')
    model.load_state_dict(state, strict=True)
    model.eval()
    source = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in a.repo.rglob('*.py')
              if '__pycache__' not in p.parts and '.git' not in p.parts}
    adapter = LayeredAuthorIntake(model, family='rome', source_hashes=source)
    dataset = CIFAR10(str(a.data), train=False, download=False, transform=ToTensor())
    x, label = dataset[0]
    x = x.unsqueeze(0).requires_grad_(True)
    native = model(x)[0]
    grad1, = torch.autograd.grad(native.square().mean(), x)
    out, trace = adapter.forward_with_trace(x)
    grad2, = torch.autograd.grad(out.square().mean(), x)
    if not torch.isfinite(out).all() or not torch.isfinite(grad2).all() or not torch.equal(out, native) or not torch.equal(grad1, grad2):
        raise ValueError('trained complete-model intake/gradient mismatch')
    print(json.dumps({'status': 'CONTROL_PASS', 'checkpoint_sha256': manifest['local_sha256'],
        'strict_prediction_state_load': True, 'trained_checkpoint': True,
        'source_hyperparameters': {'s': 4, 'b': 6, 'alpha': spec['r'], 'origin': 'author code-default, not paper b2'},
        'spec': spec, 'index': 0, 'label': label, 'prediction': int(out.argmax(1)),
        'complete_forward_and_gradient_bitwise_equal': True, 'gate_calls': len(trace['events']),
        'scope': 'single raw-order input intake, not paper accuracy/attack/certification'}, indent=2))
