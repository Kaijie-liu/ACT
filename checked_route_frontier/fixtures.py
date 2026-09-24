"""Explicit analytic fixtures, never checkpoint/input selection by outcome."""
import base64
import hashlib
import math
import struct

from router_source.checker import tensor
from source_construction_lab.fixtures import document


def encode(values, shape):
    return {'dtype': 'torch.float64', 'shape': shape, 'byte_order': 'little',
            'bytes': base64.b64encode(struct.pack('<' + 'd' * len(values), *values)).decode()}


def rebind(doc):
    inventory = []
    for graph in doc['networks']:
        for layer in graph['layers']:
            if layer['kind'] == 'Linear':
                for role in ('weight', 'bias'):
                    inventory.append({'name': layer[role + '_name'], **tensor(layer[role])[0]})
    inventory.sort(key=lambda row: row['name'])
    h = hashlib.sha256()
    for value in inventory:
        h.update(value['name'].encode())
        h.update(value['sha256'].encode())
    doc['state_inventory'] = inventory
    doc['request']['model_state'] = {'sha256': h.hexdigest(), 'tensor_count': len(inventory),
        'parameter_count': sum(math.prod(v['shape']) for v in inventory)}
    return doc


def analytic(kind='prunable', *, experts=4, classes=3, width=1, depth=0):
    doc = document(experts=experts, classes=classes, width=width, depth=depth,
                   constant=True, radius='1')
    router = doc['networks'][0]['layers'][-1]
    if kind == 'crossing':
        if experts != 4 or width != 1 or depth != 0:
            raise ValueError('crossing analytic dimensions')
        weights, bias = [1., -1., 0., 0.], [0., 0., 0., -3.]
    elif kind in ('prunable', 'tied'):
        # For depth zero the output scores share the same nonzero input term;
        # last-affine duals eliminate it exactly in each difference.
        weights = [1. if j == 0 else 0. for _ in range(experts) for j in range(width)]
        bias = [float(experts - i) * 3 for i in range(experts)] if kind == 'prunable' else [0.] * experts
    else:
        raise ValueError('unknown analytic fixture')
    router['weight'] = encode(weights, [experts, width])
    router['bias'] = encode(bias, [experts])
    # Constant, safe experts allow full exact output aggregation without solver.
    for graph in doc['networks'][1:]:
        graph['layers'][-1]['bias'] = encode([0.] * (classes - 1) + [3.], [classes])
    return rebind(doc)


def timing(kind):
    """Fixed two-fixture timing design, declared before measurement.

Nontrivial hidden layers; identical final router weights isolate cancellation.
Constant safe final expert outputs allow solver-free complete proof controls.
This is NOT a trained model or a claim about its performance distribution.
"""
    if kind not in ('prunable', 'tied'):
        raise ValueError('fixed timing fixture')
    doc = document(experts=8, classes=10, width=8, depth=2, seed=724, radius='1/8')
    last = doc['networks'][0]['layers'][-1]
    last['weight'] = encode([1/8] * 64, [8, 8])
    last['bias'] = encode([float(8-i)*3 if kind == 'prunable' else 0. for i in range(8)], [8])
    for graph in doc['networks'][1:]:
        graph['layers'][-1]['weight'] = encode([0.] * 80, [10, 8])
        graph['layers'][-1]['bias'] = encode([0.] * 9 + [3.], [10])
    return rebind(doc)
