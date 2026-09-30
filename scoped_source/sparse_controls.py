"""Deterministic synthetic H1 controls; never loads real models or data."""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from fractions import Fraction as F
from source_enclosure.format import identity
from router_source.checker import tensor


def source(*, experts=3, classes=3, width=3, tied=False, unsafe=False,
           dense=False, relu_bias=0., constant=False, relational=False, router_coordinate=0):
    def packed(values, shape):
        raw = b''.join(struct.pack('<d', float(v)) for v in values)
        return {'dtype': 'torch.float64', 'byte_order': 'little', 'shape': shape,
                'bytes': base64.b64encode(raw).decode()}
    center = packed([0.]*width, [1, width]); center_id, _ = tensor(center)
    inventory = []; networks = []

    def linear(name, i, weights, bias):
        prefix = 'router' if name == 'router' else 'experts.'+name[6:]
        layer = {'index': i, 'kind': 'Linear', 'training': False}
        for role, values, shape in [('weight', sum(weights, []), [len(weights), len(weights[0])]),
                                     ('bias', bias, [len(bias)])]:
            item = packed(values, shape); ident, _ = tensor(item); key = f'{prefix}.{i}.{role}'
            inventory.append({'name': key, **ident}); layer[role] = item; layer[role+'_name'] = key
        return layer

    router_w = [[0.]*width for _ in range(experts)]
    if not tied:
        router_w[0][router_coordinate] = 1.; router_w[1][router_coordinate] = -1.
    networks.append({'name': 'router', 'layers': [linear('router', 0, router_w, [0.]*experts)]})
    for e in range(experts):
        name = f'expert{e}'
        hidden = [[float(i == j) for j in range(width)] for i in range(width)]
        last = [[0.]*width for _ in range(classes)]
        if not constant:
            last[0] = [1. if dense or j == 0 else 0. for j in range(width)]
        bias = [(-10. if unsafe else 10.)+e/4] + [0.]*(classes-1)
        hidden_bias = [relu_bias]*width
        if relational:
            if width < 2: raise ValueError('relational control needs two hidden values')
            last = [[0.]*width for _ in range(classes)]
            bias = [.75]+[0.]*(classes-1)
            if e == 0:
                hidden[1] = hidden[0][:]; hidden_bias[1] = 2.
                last[0][0] = 1.; last[0][1] = -1.; bias[0] = 2.75
        networks.append({'name': name, 'layers': [linear(name, 0, hidden, hidden_bias),
            {'index': 1, 'kind': 'ReLU', 'training': False, 'inplace': False},
            linear(name, 2, last, bias)]})
    inventory.sort(key=lambda v: v['name']); h = hashlib.sha256()
    for v in inventory: h.update(v['name'].encode()); h.update(v['sha256'].encode())
    request = {'experts': experts, 'classes': classes, 'label': 0, 'top_k': 2,
               'gate': 'SELECTED_SOFTMAX', 'tie_policy': 'ANY_LEGAL_TOPK', 'training': False,
               'center': center_id, 'radius': '1', 'clip': ['-1', '1'], 'margin': '1/100',
               'model_state': {'sha256': h.hexdigest(), 'tensor_count': len(inventory),
                               'parameter_count': sum(math.prod(v['shape']) for v in inventory)}}
    return {'schema': 'SCOPED_DECLARED_TOP2_V1', 'request': request, 'center': center,
            'state_inventory': inventory, 'networks': networks}


def exact_routes(doc, point):
    """Route witnesses only: evaluate the declared graph with exact rationals."""
    from scoped_source.sparse_ir import index
    r, nodes, out = index(doc, identity(doc), lambda: None); values = {}
    if len(point) != math.prod(doc['center']['shape'][1:]):
        raise ValueError('point shape')
    for name, node in nodes.items():
        if node['kind'] == 'input':
            value = F(point[int(name.split('/')[1])]); lo, hi = node['bounds']
            if not lo <= value <= hi: raise ValueError('point outside declared box')
        elif node['kind'] == 'affine':
            value = node['bias'] + sum((w*values[k] for k, w in node['terms'].items()), F(0))
        else: value = max(F(0), values[node['parent']])
        values[name] = value
    scores = [values[k] for k in out['router']]
    from itertools import combinations
    routes = [list(p) for p in combinations(range(r['experts']), 2)
              if all(scores[i] >= scores[j] for i in p for j in range(r['experts']) if j not in p)]
    return {'point': list(map(str, point)), 'scores': list(map(str, scores)), 'legal_pairs': routes}


def run():
    from act.back_end.solver.lp_certificate import propose
    from scoped_source.sparse_build import build
    from scoped_source.sparse_check import check
    cases = []
    for name, options in [('sparse_relu', {}), ('dense_relu', {'dense': True}),
                          ('relation_needed', {'relational': True}),
                          ('tied_constant', {'tied': True, 'constant': True}),
                          ('unsafe_control', {'unsafe': True, 'tied': True})]:
        doc = source(**options); item = {'case': name, 'source_sha256': identity(doc), 'arms': {}}
        for mode in ('dependency', 'full'):
            start = time.monotonic(); deadline = start+300
            package = build(doc, expected_source_sha256=identity(doc), deadline=deadline, mode=mode, proposer=propose)
            encoded = json.dumps(package, sort_keys=True, separators=(',', ':')).encode()
            checked = check(doc, json.loads(encoded), expected_source_sha256=identity(doc), deadline=deadline)
            item['arms'][mode] = {'status': checked['status'], 'required': checked['required'],
                'positive': checked['positive'], 'lp_bounds_checked': checked['lp_bounds_checked'],
                'source_nodes': checked['source_nodes'], 'source_blocks_checked': checked['source_blocks_checked'],
                'lp_rows_reconstructed': checked['lp_rows_reconstructed'],
                'duty_source_nodes': checked['duty_source_nodes'], 'duty_union_nodes': checked['duty_union_nodes'],
                'package_bytes': len(encoded), 'generation': package['stats'],
                'observed_generate_serialize_check_seconds': time.monotonic()-start,
                'lower_bounds': [r['lower_bound'] for r in checked['obligations']]}
        item['route_witnesses'] = [exact_routes(doc, [x, 0, 0]) for x in (-1, 1)]
        cases.append(item)
    return {'schema': 'H1_SYNTHETIC_CONTROL_REPORT_V1', 'cases': cases,
            'real_requests_started': 0, 'hard_supervision_integrated': False,
            'performance_claim': False, 'same_relaxation_full_vs_dependency': True,
            'timing_scope': 'In-process generation, proposal, proof JSON serialization and check only; excludes imports and fixture/source creation; fixed arm order; not end-to-end competitive timing.',
            'scope': 'Declared synthetic real Linear/ReLU graphs; not production/HZ differential or native FP proof.'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
