"""Inference-only absence of auxiliary global_proj, explicitly versioned.

The original checkpoint lacks48 auxiliary projection tensors. They feed only
returned loss telemetry, never gate_scores/LoRA output/logits (pinned source).
Construct author's supported gate_proj_dim=0 for STRICT prediction-state load;
compare with a full original module plus synthetic auxiliary tensors. Neither
object is a valid resumed-training checkpoint.
"""
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
    file = Path(manifest['path'])
    if hashlib.file_digest(file.open('rb'), 'sha256').hexdigest() != manifest['local_sha256']:
        raise ValueError('weight identity')
    sys.path.insert(0, str(a.repo.resolve()))
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from models import build_model, inspect_checkpoint
    from act.back_end.moe.layered_author_intake import LayeredAuthorIntake
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    if torch.serialization.get_unsafe_globals_in_checkpoint(file):
        raise ValueError('unreviewed checkpoint type')
    payload = torch.load(file, map_location='cpu', weights_only=True)
    raw = payload['state_dict']
    state = {k.removeprefix('module.'): v for k, v in raw.items()}
    if len(state) != len(raw) or not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('invalid prediction state')
    spec = inspect_checkpoint(state)
    options = dict(dataset='cifar10', pretrained=False, r=spec['r'], num_lora=spec['num_lora'],
                   lora_alpha=spec['r'], mlp_gate=spec['mlp_gate'], s=4., b=6.)
    model = build_model(**options, gate_proj_dim=0)
    model.load_state_dict(state, strict=True)
    model.eval()
    full = build_model(**options)
    missing = set(full.state_dict()) - set(state)
    expected = {f'blocks.{i}.attn.{kind}_adapter.global_proj.{part}'
                for i in range(12) for kind in ['qkv', 'proj'] for part in ['weight', 'bias']}
    if missing != expected or set(state) - set(full.state_dict()):
        raise ValueError('not exactly the registered48 inference-unused auxiliary tensors')
    completed = dict(state)
    # Distinct synthetic auxiliaries; NONE silently called public trained state.
    for key in sorted(missing):
        completed[key] = torch.randn_like(full.state_dict()[key]) * 10
    full.load_state_dict(completed, strict=True)
    full.eval()
    dataset = CIFAR10(str(a.data), train=False, download=False, transform=ToTensor())
    x, label = dataset[0]
    x = x.unsqueeze(0).requires_grad_(True)
    y = model(x)[0]
    z = full(x)[0]
    g, = torch.autograd.grad(y.square().mean(), x)
    h, = torch.autograd.grad(z.square().mean(), x)
    if not torch.isfinite(y).all() or not torch.isfinite(g).all() or not torch.equal(y, z) or not torch.equal(g, h):
        raise ValueError('unused-auxiliary concrete/gradient control failed')
    source = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in a.repo.rglob('*.py') if '.git' not in p.parts and '__pycache__' not in p.parts}
    adapter = LayeredAuthorIntake(model, family='rome', source_hashes=source)
    result, trace = adapter.forward_with_trace(x)
    if not torch.equal(result, y):
        raise ValueError('full trained intake differs')
    print(json.dumps({'status': 'CONTROL_PASS', 'checkpoint_sha256': manifest['local_sha256'],
        'spec': spec, 'prediction_state_tensors': len(state), 'missing_auxiliary_tensors': sorted(missing),
        'strict_prediction_load': True, 'unused_auxiliary_variant': 'gate_proj_dim=0_inference_only',
        'synthetic_auxiliary_full_model_and_gradient_equal': True,
        'native_gate_scores_preserved': True, 'gate_calls': len(trace['events']),
        'source_options': {'s': 4, 'b': 6, 'alpha': spec['r'], 'scope': 'code-default, not reconstructed training recipe'},
        'index': 0, 'label': label, 'prediction': int(y.argmax(1)),
        'source_hashes': source, 'scope': 'trained-state inference control only; not resumed training or paper accuracy'}, indent=2))
