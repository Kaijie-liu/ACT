"""Lossless append-only state transport. No mathematical acceptance logic."""
import copy
try:
    from proof_format import identity
except ImportError:
    from source_enclosure.format import identity


def suffix(matrix, start):
    at = matrix['indptr'][start]
    return {'shape':[matrix['shape'][0]-start, matrix['shape'][1]],
            'data':matrix['data'][at:], 'indices':matrix['indices'][at:],
            'indptr':[p-at for p in matrix['indptr'][start:]]}


def encode(source, target):
    s, t = source['hz'], target['hz']
    return {'schema':'APPEND_SOURCE_DELTA_V1', 'source':identity(source), 'target':identity(target),
            'continuous_suffix':target['continuous_ids'][len(source['continuous_ids']):],
            'binary_suffix':target['binary_ids'][len(source['binary_ids']):],
            'outputs':{k:t[k] for k in ('c','Gc','Gb')},
            'constraints':{**{k:suffix(t[k], len(s[r])) for k,r in
                [('Ac','b'),('Ab','b'),('Auc','ub'),('Aub','ub')]},
                **{k:t[k][len(s[k]):] for k in ('b','ub')}}}


def restore(source, delta):
    if set(delta) != {'schema','source','target','continuous_suffix','binary_suffix','outputs','constraints'}:
        raise ValueError('delta schema')
    if delta['schema'] != 'APPEND_SOURCE_DELTA_V1' or identity(source) != delta['source']:
        raise ValueError('delta parent binding')
    if set(delta['outputs']) != {'c','Gc','Gb'} or set(delta['constraints']) != {'Ac','Ab','b','Auc','Aub','ub'}:
        raise ValueError('delta inventory')
    t = copy.deepcopy(source)
    for kind in ('continuous','binary'): t[kind+'_ids'] += delta[kind+'_suffix']
    t['hz'].update(delta['outputs'])
    for k in ('b','ub'): t['hz'][k] += delta['constraints'][k]
    for key, kind in [('Ac','continuous'),('Auc','continuous'),('Ab','binary'),('Aub','binary')]:
        a, b = source['hz'][key], delta['constraints'][key]; width = len(t[kind+'_ids'])
        if b['shape'][1] != width: raise ValueError('delta column count')
        t['hz'][key] = {'shape':[a['shape'][0]+b['shape'][0],width],
            'data':a['data']+b['data'], 'indices':a['indices']+b['indices'],
            'indptr':a['indptr']+[len(a['data'])+p for p in b['indptr'][1:]]}
    if identity(t) != delta['target']: raise ValueError('delta target binding')
    return t
