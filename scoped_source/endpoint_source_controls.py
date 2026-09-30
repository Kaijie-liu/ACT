"""Fixed synthetic source controls for H2; no training or real checkpoint."""
import base64
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from router_source.checker import tensor
from scoped_source.sparse_controls import source, exact_routes
from scoped_source.sparse_ir import index
from scoped_source.endpoint_controls import exact_point
from scoped_source.endpoint_source_build import build, mc_lp, THRESHOLD as BUILD_THRESHOLD
from scoped_source.endpoint_source_check import check
from scoped_source.endpoint_check import THRESHOLD as CHECK_THRESHOLD
from source_enclosure.format import identity

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/h2_source_controls_20260930.json'
PROTOCOL_SHA256 = '154e6171c8c2ffad14b6394ae19419827f22184214522a64a27cd3a727a7180a'


def protocol(value=None):
    value = json.loads(PROTOCOL.read_text()) if value is None else value
    if (identity(value) != PROTOCOL_SHA256 or
            F(value['acceptance_threshold']) != BUILD_THRESHOLD or BUILD_THRESHOLD != CHECK_THRESHOLD):
        raise ValueError('frozen H2 source protocol changed')
    return value


def weighted_source():
    config = protocol()['weighted_sign']
    inventory = []; networks = []
    def encode(values, shape):
        return {'dtype': 'torch.float64', 'byte_order': 'little', 'shape': shape,
                'bytes': base64.b64encode(b''.join(struct.pack('<d', float(v)) for v in values)).decode()}
    def linear(name, layer_index, weights, bias):
        prefix = 'router' if name == 'router' else 'experts.'+name[6:]
        result = {'index': layer_index, 'kind': 'Linear', 'training': False}
        for role, values, shape in [('weight', sum(weights, []), [len(weights), len(weights[0])]),
                                    ('bias', bias, [len(bias)])]:
            data = encode(values, shape); key = f'{prefix}.{layer_index}.{role}'
            result[role] = data; result[role+'_name'] = key
            inventory.append({'name': key, **tensor(data)[0]})
        return result
    networks.append({'name': 'router', 'layers': [linear('router', 0, config['router_weights'], config['router_bias'])]})
    for i, carry in enumerate(config['output_carry_coefficients']):
        name = f'expert{i}'
        networks.append({'name': name, 'layers': [
            linear(name, 0, config['hidden_weights'], config['hidden_bias']),
            {'index': 1, 'kind': 'ReLU', 'training': False, 'inplace': False},
            linear(name, 2, [config['output_relu_coefficients']+[carry], [0., 0., 0.]],
                   [config['output_bias_binary64'][i], 0.])]})
    inventory.sort(key=lambda t: t['name']); digest = hashlib.sha256()
    for item in inventory: digest.update(item['name'].encode()); digest.update(item['sha256'].encode())
    center = encode([0.], [1, 1])
    r = {'experts': 3, 'classes': config['classes'], 'label': config['label'], 'top_k': 2,
         'gate': 'SELECTED_SOFTMAX', 'tie_policy': 'ANY_LEGAL_TOPK', 'training': False,
         'center': tensor(center)[0], 'radius': config['radius'], 'clip': config['clip'], 'margin': config['margin'],
         'model_state': {'sha256': digest.hexdigest(), 'tensor_count': len(inventory),
                         'parameter_count': sum(math.prod(v['shape']) for v in inventory)}}
    return {'schema': 'SCOPED_DECLARED_TOP2_V1', 'request': r, 'center': center,
            'state_inventory': inventory, 'networks': networks}


def cases():
    protocol()
    return [('weighted_sign', weighted_source(), []),
            ('tied_partial_reuse', source(experts=4, classes=4, width=2, tied=True, constant=True),
             [((0, 1), 1), ((0, 2), 2), ((2, 3), 3)]),
            ('unsafe_tied', source(width=2, unsafe=True, tied=True), []),
            ('unresolved_sign', source(dense=True), [])]


def values_at(doc, point):
    _, nodes, _ = index(doc, identity(doc), lambda: None); values = {}
    for name, node in nodes.items():
        if node['kind'] == 'input':
            value = F(point[int(name.split('/')[1])])
            if not node['bounds'][0] <= value <= node['bounds'][1]: raise ValueError('point outside source')
        elif node['kind'] == 'affine':
            value = node['bias']+sum((v*values[k] for k, v in node['terms'].items()), F(0))
        else: value = max(F(0), values[node['parent']])
        values[name] = value
    return values


def weighted_negative_point(doc, package):
    if package['mode'] != 'mccormick': raise ValueError('MC package required for MC witness')
    check(doc, package, expected_source_sha256=identity(doc), expected_mode='mccormick', deadline=time.monotonic()+300)
    duty = package['request']['duties'][0]; lp = mc_lp(duty); values = values_at(doc, [0])
    if identity(lp) != package['proof']['duties'][0]['lp_sha256']:
        raise ValueError('MC witness must bind the independently checked duty')
    point = [values[v] for v in duty['variables']]+[F(1, 4), F(-1, 8)]
    objective = exact_point(lp, point)
    if objective >= 0: raise ValueError('fixed source control no longer separates MC')
    return {'pair': duty['pair'], 'competitor': duty['competitor'], 'lp_sha256': identity(lp),
            'point': list(map(str, point)), 'checked_objective': str(objective),
            'network_counterexample': False, 'lp_optimum_claimed': False}


def run(proposer):
    protocol()
    rows = []
    for name, doc, reuse in cases():
        item = {'case': name, 'source_sha256': identity(doc), 'arms': {}}
        for mode in ('endpoints', 'mccormick'):
            deadline = time.monotonic()+300
            package = build(doc, expected_source_sha256=identity(doc), deadline=deadline,
                            mode=mode, reuse_keys=reuse, proposer=proposer)
            checked = check(doc, package, expected_source_sha256=identity(doc), expected_mode=mode, deadline=deadline)
            item['arms'][mode] = {'checked': checked, 'proposal_stats': package['proposal_stats'],
                                  'proposal_errors': package['proposal_errors'], 'package_sha256': identity(package)}
            if name == 'weighted_sign' and mode == 'mccormick':
                item['mc_negative_point'] = weighted_negative_point(doc, package)
        if name == 'weighted_sign':
            item['route_witnesses'] = [exact_routes(doc, [x]) for x in (-1, F(-1, 2), 1)]
        rows.append(item)
    return {'schema': 'H2_SOURCE_CONTROL_REPORT_V1', 'protocol_sha256': identity(json.loads(PROTOCOL.read_text())),
            'cases': rows, 'real_requests': 0, 'hard_supervision': False, 'runtime_comparison': False,
            'guarantee': 'declared real binary64-coefficient graph; not native floating execution'}
