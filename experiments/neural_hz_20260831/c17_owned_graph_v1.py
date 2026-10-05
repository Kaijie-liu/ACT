"""Same affine DAG; packed reverse ownership replaces discarded support counts."""

import numpy as np
from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import live_value_rows, _source_row
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import OwnerEngine, retag, RADIX, UID_LIMIT


def graph(expr, keep, max_work, *, uid_start):
    nodes, intern = [], {}

    def node(kind, key, **fields):
        if key not in intern:
            intern[key] = len(nodes)
            nodes.append({'kind': kind, **fields})
        return intern[key]

    def factor(terms):
        groups = {}
        for source, ops in terms:
            key = ('op', id(ops[-1])) if ops else ('source', id(source))
            groups.setdefault(key, []).append((source, ops))
        children = []
        for key, members in groups.items():
            if key[0] == 'source':
                source = members[0][0]
                child = node('source', key, source=source, width=source.n_out, parents=())
                children.extend([child] * len(members))
            else:
                op = members[0][1][-1]
                parent = factor([(s, ops[:-1]) for s, ops in members])
                child = node('op', ('op_node', parent, id(op)), op=op, width=op.shape[0], parents=(parent,))
                children.append(child)
        if len(children) == 1:
            return children[0]
        if not children or len({nodes[c]['width'] for c in children}) != 1:
            raise ValueError('invalid exact sum node')
        return node('sum', ('sum', tuple(children)), parents=tuple(children), width=nodes[children[0]]['width'])

    root = factor([(t.source, t.operators) for t in expr.terms])
    if nodes[root]['width'] != expr.n_out:
        raise ValueError('factor graph output width mismatch')
    if type(uid_start) is not int or uid_start < 0:
        raise ValueError('invalid original predicate UID prefix')
    uid_bases, cursor = [], uid_start
    for n in nodes:
        uid_bases.append(cursor)
        cursor += n['width']
    if cursor + 16_384 + 3 * expr.n_out > UID_LIMIT:
        raise MemoryError('stable logical/radix/future-phase UID reservation exhausted')
    engine = OwnerEngine(max_work)
    ownership = []
    for n in nodes:
        before = engine.visits
        engine.charge(n['width'])
        ownership.append(np.zeros(n['width'], np.int64))
        if n['kind'] == 'source':
            support = live_value_rows(n['source'])
        elif n['kind'] == 'op':
            support = engine.compute(n['op'], nodes[n['parents'][0]]['support']) != 0
        else:
            support = np.logical_or.reduce([nodes[p]['support'] for p in n['parents']])
        n.update(support=support, needed=np.zeros(n['width'], dtype=bool), support_work=engine.visits - before)
    nodes[root]['needed'] = keep & nodes[root]['support']
    for ni in reversed(range(len(nodes))):
        n = nodes[ni]
        before = engine.visits
        if n['kind'] == 'op':
            p = nodes[n['parents'][0]]
            local = engine.owners(n['op'], n['needed'])
            enabled = (local != 0) & p['support']
            p['needed'] |= enabled
            if enabled.any():
                engine.charge(4 * p['width'])
                ownership[n['parents'][0]][enabled] += retag(local[enabled], uid_bases[ni])
        elif n['kind'] == 'sum':
            for pid in set(n['parents']):
                enabled = n['needed'] & nodes[pid]['support']
                nodes[pid]['needed'] |= enabled
                if enabled.any():
                    engine.charge(4 * n['width'])
                    coords = np.flatnonzero(enabled)
                    ownership[pid][coords] += RADIX + uid_bases[ni] + coords
        n['support_work'] += engine.visits - before
    counts = []
    path_costs = []
    for n in nodes:
        nc = nb = centers = 0
        rows = np.flatnonzero(n['needed'])
        if n['kind'] == 'source':
            s = n['source']
            for row in rows:
                c, b = _source_row(s, row)
                nc += c[1].size
                nb += b[1].size
            centers = int(np.count_nonzero(s.c[rows]))
        elif n['kind'] == 'op':
            p = nodes[n['parents'][0]]
            nc = int(engine.compute(n['op'], p['support'])[n['needed']].sum())
        else:
            nc = sum(int(np.count_nonzero(n['needed'] & nodes[p]['support'])) for p in set(n['parents']))
        cost = 16 * (nc + nb + centers + rows.size) + n['support_work']
        counts.append({'kind': n['kind'], 'width': n['width'], 'auxiliaries': int(rows.size),
            'continuous_edges': nc, 'binary_edges': nb, 'center_edges': centers,
            'encoding_work_upper': int(cost - n['support_work']), 'support_work': n['support_work']})
        path_costs.append(cost + max((path_costs[p] for p in n['parents']), default=0))
    total = engine.visits + sum(c['encoding_work_upper'] for c in counts)
    return nodes, root, {'node_counts': counts, 'support_work': engine.visits,
        'total_work_upper': total, 'largest_branch_work_upper': max(path_costs),
        'auxiliaries': sum(c['auxiliaries'] for c in counts),
        'continuous_edges': sum(c['continuous_edges'] for c in counts),
        'binary_edges': sum(c['binary_edges'] for c in counts),
        'temporary_support_cache_bytes': engine.cache_bytes,
        'stable_uid_start': uid_start, 'radix_uid_base': cursor,
        'ownership_reverse_replaces_boolean_count': True}, ownership, uid_bases

