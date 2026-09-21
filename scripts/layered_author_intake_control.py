"""Full-size original forward/gradient trace controls; no route freezing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('family', choices=['rome', 'robust_experts'])
    p.add_argument('--repo', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from act.back_end.moe.layered_author_intake import LayeredAuthorIntake
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    if args.family == 'rome':
        from models import build_model
        model = build_model('cifar10', pretrained=False, s=4.0, b=6.0)
    else:
        from src.models.nn.moe.resnet_conv_moe import ResNetConvMoE
        from src.models.nn.moe.routing import GlobalAvgLinearRoutingNetwork
        model = ResNetConvMoE(layers=18, num_channels=3, small_inputs=True,
            num_classes=100, num_experts=4, k=2, routing_layer_type=GlobalAvgLinearRoutingNetwork,
            expert_capacity=None, balancing_loss_type='entropy', balancing_loss=.5, moe_layer_prefix='layer4')
        if any('BatchNorm' in type(m).__name__ and type(m) is not torch.nn.BatchNorm2d for m in model.modules()):
            raise ValueError('optional SyncBN compatibility supports ordinary BN2d only')
    model.eval()
    files = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in args.repo.rglob('*.py') if '.git' not in f.parts and '__pycache__' not in f.parts}
    adapter = LayeredAuthorIntake(model, family=args.family, source_hashes=files)
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    native = model(x)
    native = native[0] if isinstance(native, tuple) else native
    g1, = torch.autograd.grad(native.square().mean(), x)
    output, trace = adapter.forward_with_trace(x)
    g2, = torch.autograd.grad(output.square().mean(), x)
    if not torch.equal(native, output) or not torch.equal(g1, g2) or not torch.isfinite(g2).all():
        raise ValueError('full-forward/gradient trace changes execution')
    if trace['executed_gate_calls'] < 2:
        raise ValueError('expected multiple dependent routing layers')
    if adapter.verify_box()['status'] != 'UNSUPPORTED':
        raise ValueError('trace must never imply box verification')
    # Identity mutation must reject before forwarding.
    name = next(iter(adapter.source_hashes))
    previous = adapter.source_hashes[name]
    adapter.source_hashes[name] = '0' * 64
    try:
        adapter.forward_with_trace(x)
    except ValueError:
        pass
    else:
        raise ValueError('source mutation accepted')
    adapter.source_hashes[name] = previous
    model.train()
    try:
        adapter.forward_with_trace(x)
    except ValueError:
        pass
    else:
        raise ValueError('train semantics accepted')
    model.eval()
    print(json.dumps({'family': args.family, 'status': 'CONTROL_PASS', 'full_forward_bitwise_equal': True,
        'input_gradient_bitwise_equal': True, 'source_mutation_rejected': True, 'train_mode_rejected': True,
        'parameters': sum(p.numel() for p in model.parameters()), 'trained_checkpoint': False,
        'source_hashes': files, 'box_verification': adapter.verify_box(),
        'route_events': [{k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in e.items()
                         if k != 'weights'} for e in trace['events']],
        'scope': 'original full-size architecture intake/trace; not training, accuracy or certification'}, indent=2))
