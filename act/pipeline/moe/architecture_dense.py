"""S1 dense capacity controls. No routing or verification-based selection."""
import torch
from torch import nn


def parameter_count(widths):
    return sum((a + 1) * b for a, b in zip(widths, widths[1:]))


def matched_widths(target, input_dim=3072, classes=10):
    # Preserve the expert's two hidden layers and 2:1 width ratio. Select only
    # by parameter arithmetic, not accuracy or verification. Smaller wins ties.
    h = 1
    while parameter_count([input_dim, 2*h, h, classes]) < target:
        h += 1
    choices = range(max(1, h-1), h+1)
    h = min(choices, key=lambda j: (abs(parameter_count(
        [input_dim, 2*j, j, classes])-target), j))
    return [input_dim, 2*h, h, classes]


def build(config, seed):
    widths = config['widths']
    if len(widths) != 4 or any(type(n) is not int or n <= 0 for n in widths):
        raise ValueError('dense S1 requires two positive-width hidden layers')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers = [nn.Flatten()]
        for i, (a, b) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['format'] != 'act-dense-capacity-s1-v1':
        raise ValueError('dense S1 checkpoint format')
    model = build(payload['architecture'], payload['seed'])
    model.load_state_dict(payload['state_dict'], strict=True)
    if sum(p.numel() for p in model.parameters()) != payload['parameter_count']:
        raise ValueError('dense S1 parameter count')
    return model.eval(), payload
