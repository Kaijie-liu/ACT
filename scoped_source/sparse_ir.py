"""Shared syntax/serialization for H1 controls, not a proof constructor.

The declared graph is the semantic anchor. Tensor/program correspondence and
deployed floating point are not established by this scalar index.
"""
from fractions import Fraction as F
from scoped_source.graph import validate, operator


def index(doc, expected, tick):
    request, lower, upper = validate(doc, expected, tick)
    nodes = {}
    inputs = []
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        key = f'input/{i}'
        nodes[key] = {'kind': 'input', 'bounds': (lo, hi)}
        inputs.append(key)
    outputs = {}
    for graph in doc['networks']:
        current = inputs[:]
        shape = request['center']['shape']
        for layer in graph['layers']:
            tick()
            shape, weights, bias = operator(shape, layer)
            if layer['kind'] == 'Flatten':
                continue
            following = []
            for i in range(len(current) if weights is None else len(weights)):
                key = f"{graph['name']}/layer/{layer['index']}/value/{i}"
                if key in nodes:
                    raise ValueError('duplicate source node')
                nodes[key] = ({'kind': 'relu', 'parent': current[i]}
                              if weights is None else
                              {'kind': 'affine', 'bias': bias[i],
                               'terms': {current[j]: v for j, v in weights[i].items()}})
                following.append(key)
            current = following
        outputs[graph['name']] = current
    return request, nodes, outputs


def row(sense, terms, rhs):
    return {'sense': sense, 'terms': {k: str(v) for k, v in sorted(terms.items()) if v},
            'rhs': str(rhs)}


def lp_record(variables, bounds, rows, objective, offset):
    """Mechanical named-row to CSR conversion only; no lowering rule here."""
    if len(set(variables)) != len(variables):
        raise ValueError('variable alias')
    positions = {name: i for i, name in enumerate(variables)}
    result = {'matrix_format': 'csr_v1', 'offset': str(offset),
              'c': [str(objective.get(v, F(0))) for v in variables],
              'lower': [str(bounds[v][0]) for v in variables],
              'upper': [str(bounds[v][1]) for v in variables]}
    if any(v not in positions for v in objective):
        raise ValueError('unbound objective variable')
    for sense, matrix, rhs in [('le', 'A', 'b'), ('eq', 'E', 'h')]:
        selected = [r for r in rows if r['sense'] == sense]
        data, indices, indptr = [], [], [0]
        for item in selected:
            for name, value in sorted(item['terms'].items(), key=lambda p: positions[p[0]]):
                data.append(value); indices.append(positions[name])
            indptr.append(len(data))
        result[matrix] = {'shape': [len(selected), len(variables)], 'data': data,
                          'indices': indices, 'indptr': indptr}
        result[rhs] = [r['rhs'] for r in selected]
    if any(r['sense'] not in ('le', 'eq') for r in rows):
        raise ValueError('unknown row sense')
    return result
