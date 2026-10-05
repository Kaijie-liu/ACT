"""DAG birth blocks with no candidate parent need no coefficient incidence scan.

Source coefficient/box correctness is a prerequisite, not inferred from exact=True.
This router freshly binds the complete paths, slots and output geometry. Packed
radix definitions are always inspected and cannot disappear behind a block skip.
"""
from collections import Counter
import numpy as np
import scipy.sparse as sp


def route_rows(saved,raw_columns,*,pool,enabled=False):
    if not enabled:return None
    hz=saved['hz'];nodes=saved['definition_graph'];expr=saved['expression']
    old=int(saved['old_n_cont']);logical=int(saved['logical_n_cont']);oe=int(saved['old_n_eq'])
    root_id=saved['root'];main=logical-old
    if not 0<=old<=logical<=hz.n_cont<=64_000_000 or not 0<=root_id<len(nodes):raise ValueError('complete birth frame required')
    pool.charge('c63_complete_birth_block_paths_and_maps',8*(sum(n['width'] for n in nodes)+main+hz.n_eq)+128*len(nodes)+1024)
    if (hz.n_bin!=saved['old_n_bin'] or hz.frame_id!=expr.frame_id
            or len(saved['eq_roots'])!=oe+main):raise ValueError('original nonconvex frame differs')
    raw=np.zeros(hz.n_cont,bool);raw[raw_columns]=True
    routed=np.zeros(hz.n_eq,bool);routed[saved['def_rows']]=True
    # Old predicates are over the old frame. For canonical rows, their last
    # column is a sufficient fresh check; empty rows are allowed.
    prior=saved['eq_roots'][:oe];starts=hz.Ac.indptr[prior];ends=hz.Ac.indptr[prior+1]
    nonempty=ends>starts
    # A packed old row can mention a later radix slot: conservatively inspect
    # that whole row instead of inferring membership from its last column.
    routed[prior[nonempty]]=hz.Ac.indices[ends[nonempty]-1]>=old
    paths=[];has_raw=[];frontier=old;blocks=[];physical_seen=np.zeros(hz.n_eq,bool)
    physical_seen[prior]=True;physical_seen[saved['def_rows']]=True
    for index,node in enumerate(nodes):
        kind=node['kind'];ps=node['parents'];width=node['width']
        if kind not in ('source','op','sum') or any(not 0<=p<index for p in ps):raise ValueError('complete topological birth graph required')
        if any(node[n].shape!=(width,) for n in ('slots','exponents','needed','support')):raise ValueError('complete birth coordinate maps required')
        if node['needed'].dtype!=np.dtype(bool) or np.any(node['needed'] & ~node['support']):raise ValueError('birth liveness differs')
        coordinates=np.flatnonzero(node['needed']);slots=node['slots'][coordinates]
        if not np.array_equal(slots,np.arange(frontier,frontier+len(slots))):raise ValueError('contiguous original birth slots differ')
        frontier+=len(slots);physical=saved['eq_roots'][oe+slots-old]
        if np.any(physical_seen[physical]):raise ValueError('overlapping original physical birth rows')
        physical_seen[physical]=True
        if kind=='source':
            if ps or node['source'].n_cont>old:raise ValueError('source contains future continuous factors')
            expanded=[(id(node['source']),())];possible=False
        elif kind=='op':
            if len(ps)!=1:raise ValueError('operator birth arity differs')
            expanded=[(s,(*ops,id(node['op']))) for s,ops in paths[ps[0]]]
            possible=has_raw[ps[0]]
        else:
            expanded=[term for p in ps for term in paths[p]];possible=any(has_raw[p] for p in ps)
        if len(expanded)>len(expr.terms):raise ValueError('unregistered source path')
        paths.append(expanded);has_raw.append(bool(raw[slots].any()))
        if possible:routed[physical]=True
        coefficients=int((hz.Ac.indptr[physical+1]-hz.Ac.indptr[physical]).sum())
        blocks.append(dict(node=index,kind=kind,MAIN_rows=len(slots),raw_factors=int(raw[slots].sum()),
                           parent_can_contain_raw=bool(possible),physical_coefficients=coefficients))
    if frontier!=logical or not physical_seen.all():raise ValueError('complete birth physical partition differs')
    if Counter(paths[root_id])!=Counter((id(t.source),tuple(id(op) for op in t.operators)) for t in expr.terms):raise ValueError('original expression paths differ')
    root=nodes[root_id];coords=np.flatnonzero(saved['keep'] & root['support'])
    if not np.array_equal(root['needed'],saved['keep'] & root['support']):raise ValueError('complete original output liveness differs')
    expected=sp.csr_matrix((np.ldexp(np.ones(len(coords)),root['exponents'][coords]),(coords,root['slots'][coords])),shape=hz.Gc.shape)
    if (any(not np.array_equal(getattr(expected,n),getattr(hz.Gc,n)) for n in ('data','indices','indptr'))
            or hz.Gb.nnz or not np.array_equal(expr.bias,hz.c)):raise ValueError('complete original output mapping differs')
    if np.any(raw[root['slots'][coords]]):raise ValueError('live output cannot be a raw candidate')
    return routed,dict(schema='c63_complete_DAG_birth_incidence_routing_v1',nodes=len(nodes),blocks=blocks,
        complete_path_slot_output_binding=True,radix_rows_always_inspected=len(saved['def_rows']),
        original_coefficient_and_box_binding_inherited=True,original_numeric_source_rows_requalified=False,
        skipped_MAIN_blocks=sum(not b['parent_can_contain_raw'] for b in blocks),
        skipped_MAIN_rows=sum(b['MAIN_rows'] for b in blocks if not b['parent_can_contain_raw']),
        skipped_MAIN_coefficients=sum(b['physical_coefficients'] for b in blocks if not b['parent_can_contain_raw']),
        no_HZ_writer_or_runtime_payment_claim=True,formal_gain=0)
