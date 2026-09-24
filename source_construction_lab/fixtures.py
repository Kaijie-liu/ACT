"""Deterministic synthetic declared graphs, no model, checkpoint or data load."""
import base64
import hashlib
import math
import random
import struct
from router_source.checker import tensor


def document(*, experts=4, classes=4, width=16, depth=2, seed=724, tied=False,
             radius='1/8', constant=False):
    rng = random.Random(seed)
    def encode(values, shape):
        return {'dtype': 'torch.float64', 'shape': shape, 'byte_order': 'little',
            'bytes': base64.b64encode(struct.pack('<' + 'd'*len(values), *values)).decode()}
    networks = []
    inventory = []
    for name, prefix, out in [('router', 'router', experts)] + [
            (f'expert{i}', f'experts.{i}', classes) for i in range(experts)]:
        layers = [{'kind': 'Flatten', 'index': 0, 'training': False, 'dimensions': [1, -1]}]
        for index in range(depth + 1):
            n = out if index == depth else width
            zero = constant or (tied and name == 'router')
            w = [0. if zero else rng.uniform(-1., 1.) / width for _ in range(n*width)]
            b = [0. if tied or constant else rng.uniform(-.01, .01) for _ in range(n)]
            at = len(layers)
            layer = {'kind': 'Linear', 'index': at, 'training': False}
            for role, values, shape in [('weight', w, [n, width]), ('bias', b, [n])]:
                obj = encode(values, shape)
                layer[role] = obj
                layer[role+'_name'] = f'{prefix}.{at}.{role}'
                inventory.append({'name': layer[role+'_name'], **tensor(obj)[0]})
            layers.append(layer)
            if index != depth:
                layers.append({'kind': 'ReLU', 'index': len(layers), 'training': False, 'inplace': False})
        networks.append({'name': name, 'layers': layers})
    inventory.sort(key=lambda x: x['name'])
    h = hashlib.sha256()
    for v in inventory:
        h.update(v['name'].encode()); h.update(v['sha256'].encode())
    center = encode([0.] * width, [1, 1, width])
    return {'schema': 'SCOPED_DECLARED_TOP2_V1', 'request': {
        'experts': experts, 'classes': classes, 'label': classes-1, 'top_k': 2,
        'gate': 'SELECTED_SOFTMAX', 'tie_policy': 'ANY_LEGAL_TOPK', 'training': False,
        'center': tensor(center)[0], 'radius': radius, 'margin': '1/100', 'clip': ['-1', '1'],
        'model_state': {'sha256': h.hexdigest(), 'tensor_count': len(inventory),
            'parameter_count': sum(math.prod(v['shape']) for v in inventory)}},
        'center': center, 'state_inventory': inventory, 'networks': networks,
        'trust': 'synthetic declared real graph; no dataset/checkpoint, no positive output claim'}
