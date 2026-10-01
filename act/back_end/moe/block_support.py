"""Opt-in CPU multi-objective support on shared/private SparseHZono blocks.

No joint HZ/LP construction or global sparse column remapping. Global dense
residuals remain; this is not constant-memory or a different mathematical domain.
"""
from copy import deepcopy
import math

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.moe.check_block_support import SCHEMA, ALGORITHM, structure, project, validate_batch, evaluate
from scoped_source.rowwise_bound import clock, identity, rational


def snapshot(hz, tick):
    if not isinstance(hz, SparseHZono): raise ValueError('actual SparseHZono required')
    if not 1 <= hz.n_cont+hz.n_bin <= 128 or not 1 <= hz.n_out <= 128 or hz.n_eq+hz.n_ineq > 256:
        raise ValueError('finite HZ control capacity')
    result = {'frame_id': hz.frame_id, 'exact': bool(hz.exact)}
    for key in ('c', 'b', 'ub'):
        v = getattr(hz, key)
        if not np.isfinite(v).all(): raise ValueError('nonfinite HZ vector')
        result[key] = v.reshape(-1).tolist()
    for key in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'):
        m = getattr(hz, key)
        if not sp.isspmatrix_csr(m): raise ValueError('original canonical CSR required')
        m.check_format(full_check=True)
        if not np.isfinite(m.data).all(): raise ValueError('nonfinite HZ coefficient')
        for i in range(m.shape[0]):
            tick(); cols = m.indices[m.indptr[i]:m.indptr[i+1]]
            if np.any(cols[1:] <= cols[:-1]): raise ValueError('noncanonical HZ row')
        result[key] = {'shape': list(m.shape), 'indptr': m.indptr.tolist(),
                       'indices': m.indices.tolist(), 'data': m.data.tolist()}
    tick(); return result


def prepare_batch(common, entry, a, b, queries, *, context, deadline):
    tick = clock(deadline); live = dict(common=common, entry=entry, a=a, b=b)
    sources = {k: snapshot(h, tick) for k, h in live.items()}; anchors = identity(sources)
    parsed, layout, _ = structure(sources, tick); roster = []
    for q in queries:
        item = {'id': q['id'], 'q': [str(rational(v)) for v in q['q']],
                'offset': str(rational(q['offset'])), 'side': q['side']}
        c, d = project(parsed, layout, item, tick)
        roster.append(item | {'c': list(map(str, c)), 'constant': str(d)})
    batch = {'schema': SCHEMA, 'context': deepcopy(context), 'sources': sources, 'layout': layout,
             'queries': roster, 'relaxation': 'BINARY_FACTORS_TO_CONTINUOUS_BOX'}
    validate_batch(batch, expected_batch_sha256=identity(batch), deadline=deadline)
    if identity({k: snapshot(h, tick) for k, h in live.items()}) != anchors:
        raise ValueError('original HZ changed during preparation')
    tick(); return batch


def _candidate_columns(parts, targets, *, deadline):
    tick = clock(deadline)
    if torch.get_num_threads() != 1: raise ValueError('one CPU thread required')
    def tensor(values):
        v = torch.tensor(values, dtype=torch.float64, device='cpu')
        if not bool(torch.isfinite(v).all()): raise ValueError('nonfinite tensor conversion')
        tick(); return v
    def matrix(rows, width):
        rr = []; cc = []; data = []
        for i, row in enumerate(rows):
            tick()
            for j, v in sorted(row.items()): rr.append(i); cc.append(j); data.append(float(v))
        ix = torch.tensor([rr, cc], dtype=torch.int64, device='cpu')
        return torch.sparse_coo_tensor(ix, tensor(data), (len(rows), width),
                                       dtype=torch.float64, device='cpu').coalesce()
    c = tensor([[float(v) for v in obj] for obj, _ in targets]).T.contiguous()
    d = tensor([float(v) for _, v in targets]); k = len(targets); blocks = []
    for p in parts:
        a, e = matrix(p['A'], p['width']), matrix(p['E'], p['width'])
        y = torch.zeros((len(p['b']), k), dtype=torch.float64, device='cpu')
        t = torch.zeros((len(p['h']), k), dtype=torch.float64, device='cpu')
        blocks.append({'a': a, 'e': e, 'at': a.transpose(0,1).coalesce(), 'et': e.transpose(0,1).coalesce(),
                       'b': tensor(list(map(float, p['b'])))[:,None], 'h': tensor(list(map(float, p['h'])))[:,None],
                       'cols': torch.tensor(p['columns'], dtype=torch.int64, device='cpu'),
                       'y': y, 't': t, 'best_y': y.clone(), 'best_t': t.clone()})
    best = torch.full((k,), -math.inf, dtype=torch.float64, device='cpu')
    with torch.no_grad():
        for iteration in range(129):
            tick(); r = c.clone(); value = d.clone()
            for p in blocks:
                r.index_add_(0, p['cols'], -torch.sparse.mm(p['at'], p['y']))
                r.index_add_(0, p['cols'], -torch.sparse.mm(p['et'], p['t']))
                value += (p['b']*p['y']).sum(0) + (p['h']*p['t']).sum(0)
            value -= r.abs().sum(0)
            if not bool(torch.isfinite(value).all() and torch.isfinite(r).all()):
                raise ValueError('nonfinite candidate iteration')
            improved = value > best; best = torch.where(improved, value, best)
            for p in blocks:
                if not bool(torch.isfinite(p['y']).all() and torch.isfinite(p['t']).all()):
                    raise ValueError('nonfinite dual iterate')
                p['best_y'][:,improved] = p['y'][:,improved]; p['best_t'][:,improved] = p['t'][:,improved]
            if iteration == 128: break
            x = -torch.sign(r); step = .125/math.sqrt(iteration+1)
            for p in blocks:
                local = x.index_select(0, p['cols'])
                p['y'] = torch.minimum(p['y'] + step*(p['b']-torch.sparse.mm(p['a'], local)), torch.zeros_like(p['y']))
                p['t'] = p['t'] + step*(p['h']-torch.sparse.mm(p['e'], local))
    tick()
    return [[{'source': src['source'], 'y': p['best_y'][:,i].tolist(), 't': p['best_t'][:,i].tolist()}
             for src, p in zip(parts, blocks)] for i in range(k)]


def propose_batch(batch, *, expected_batch_sha256, deadline, device='cpu'):
    tick = clock(deadline)
    if device != 'cpu': raise ValueError('only CPU frozen block control admitted')
    parts, targets = validate_batch(batch, expected_batch_sha256=expected_batch_sha256, deadline=deadline)
    proposed = _candidate_columns(parts, targets, deadline=deadline)
    if len(proposed) != len(targets): raise ValueError('partial candidate roster')
    entries = []
    for query, target, dual in zip(batch['queries'], targets, proposed):
        value, _ = evaluate(parts, target, dual, tick)
        zero = [{'source': p['source'], 'y': [0]*len(p['b']), 't': [0]*len(p['h'])} for p in parts]
        fallback, _ = evaluate(parts, target, zero, tick)
        if fallback > value: value, dual = fallback, zero
        entries.append({'id': query['id'], 'duals': dual, 'claimed_lower_bound': str(value),
                        'zero_candidate_lower_bound': str(fallback)})
    if identity(batch) != expected_batch_sha256: raise ValueError('batch changed during proposal')
    tick(); return {'batch_sha256': expected_batch_sha256, 'entries': entries, 'algorithm': ALGORITHM,
                    'iterations': 128, 'dtype': 'float64', 'device': 'cpu'}
