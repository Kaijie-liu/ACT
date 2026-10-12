"""Opt-in exact coefficient-reference support for an actual SparseHZono.

Caller-provided links are compression suggestions only: all original native
rows are checked independently. This does not check network-to-HZ provenance.
No optimizer, GPU, default analyzer change or expanded-vector cache.
"""
from fractions import Fraction as F
import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from scoped_source.coefficient_dag import SCHEMA, Graph, evaluate
from scoped_source.hz_request_store import POLICY, Work, compact, decode, digest, identity, require

KINDS = (('output', 'c', 'Gc', 'Gb'), ('eq', 'b', 'Ac', 'Ab'), ('ine', 'ub', 'Auc', 'Aub'))


def _row(hz, spec, i, work):
    kind, vk, ck, bk = spec
    work.visit('row_visits'); entries = []
    for m, shift in ((getattr(hz,ck),0), (getattr(hz,bk),hz.n_cont)):
        previous = -1
        for pos in range(m.indptr[i], m.indptr[i+1]):
            work.visit('coefficient_visits'); v = float(m.data[pos])
            require(np.isfinite(v), 'native coefficient finite')
            column = int(m.indices[pos])
            require(previous < column < m.shape[1], 'native sorted column identity')
            previous = column
            if v: entries.append((column+shift, F.from_float(v)))
    return kind, f'{kind}/{i}', F.from_float(float(getattr(hz,vk).reshape(-1)[i])), tuple(entries)


def _row_at(hz, row_index, work):
    for spec in KINDS:
        size = len(getattr(hz,spec[1]))
        if row_index < size:return _row(hz,spec,row_index,work)
        row_index -= size
    raise ValueError('native row index')


def native_rows(hz, work):
    require(isinstance(hz, SparseHZono), 'actual SparseHZono required')
    require(0 < hz.n_cont+hz.n_bin <= POLICY['active'] and 0 < hz.n_out and
            hz.n_out+hz.n_eq+hz.n_ineq <= POLICY['rows'], 'native shape capacity')
    require(sum(getattr(hz,k).nnz for k in ('Gc','Gb','Ac','Ab','Auc','Aub')) <= POLICY['nnz'],
            'native intake nnz capacity')
    for kind, vk, ck, bk in KINDS:
        vector, cm, bm = (getattr(hz, k) for k in (vk, ck, bk))
        require(vector.ndim == 1 and vector.dtype == np.float64 and np.isfinite(vector).all(), 'native vector dtype/finite')
        for m, width in ((cm, hz.n_cont), (bm, hz.n_bin)):
            require(sp.isspmatrix_csr(m) and m.dtype == np.float64 and
                m.shape == (len(vector), width), 'native CSR type/shape')
            # Do not ask SciPy's C canonicality routine to inspect unvalidated
            # pointer buffers, and do not trust its potentially stale cache.
            require(m.data.ndim == m.indices.ndim == m.indptr.ndim == 1 and
                np.issubdtype(m.indices.dtype,np.signedinteger) and np.issubdtype(m.indptr.dtype,np.signedinteger)
                and len(m.indices)==len(m.data) and len(m.indptr)==len(vector)+1
                and m.indptr[0]==0 and m.indptr[-1]==len(m.data) and
                bool(np.all(m.indptr[1:]>=m.indptr[:-1])), 'native CSR pointer integrity')
        for i in range(len(vector)):
            yield _row(hz,(kind,vk,ck,bk),i,work)


def _native_hasher(hz, factors):
    import hashlib
    return hashlib.sha256(compact(dict(frame=hz.frame_id, factors=factors, exact=bool(hz.exact),
        shape=[hz.n_out,hz.n_cont,hz.n_bin,hz.n_eq,hz.n_ineq])))


def _hash_row(h, row):
    kind, name, rhs, entries = row
    h.update(compact([kind, name, str(rhs), [[j, str(v)] for j, v in entries]]) + b'\n')


def native_identity(hz, factors, work):
    """Streaming identity of ALL given rows; not a checkpoint identity."""
    h = _native_hasher(hz, factors)
    for row in native_rows(hz, work):_hash_row(h,row)
    work.tick(); return h.hexdigest()


def check_native_rows(graph, hz, factors, *, work):
    """Independent forward expansion per original row, discarded after use."""
    work.tick();raw=compact(factors)
    require(len(raw)<=POLICY['header_bytes'],'native factor metadata capacity');factors=decode(raw)
    require(graph.frame == hz.frame_id and tuple(map(tuple, factors)) == graph.factors,
            'native frame/factor binding')
    require(len(factors) == hz.n_cont+hz.n_bin and
            [f[1] for f in factors] == ['continuous']*hz.n_cont+['binary']*hz.n_bin,
            'native continuous/binary factor map')
    require(native_identity(hz, factors, work) == graph.source, 'native source changed')
    consumed = _native_hasher(hz, factors)
    original = native_rows(hz, work); count = 0
    for root, row in zip(graph.roots, original, strict=True):
        _hash_row(consumed,row)  # Bind exactly the immutable row compared below.
        kind, name, rhs, entries = row
        require((root.kind, root.name, root.rhs) == (kind, name, rhs), 'native root/rhs inventory')
        result = {}
        def expand_path(ni, scale):
            work.visit('row_visits')
            node = graph.nodes[ni]
            for j, value in node.local:
                work.visit('coefficient_visits')
                result[j] = work.exact(result.get(j,F(0))+work.exact(scale*value))
            for parent, weight in node.refs:
                work.visit('coefficient_visits')
                expand_path(parent,work.exact(scale*weight))
        # Independent forward path substitution; one accumulated row, not a
        # cache of expanded ancestors. Repeated diamond paths are all charged.
        expand_path(root.node,F(1))
        require({j:v for j,v in result.items() if v} == dict(entries), 'native coefficient equivalence')
        count += 1
    require(count == len(graph.roots), 'native row coverage')
    require(consumed.hexdigest()==graph.source,'native compared-row source identity')
    # Detect mutation during the bridge; data are then represented by immutable Graph.
    require(native_identity(hz, factors, work) == graph.source, 'native source changed after bridge')
    work.tick(); return dict(rows=count, source=graph.source, graph=graph.identity,
                            guarantee='GIVEN_NATIVE_HZ_ROWS_ONLY')


def prepare(hz, *, request, factors, links, work):
    """links[row] = (earlier_row, exact rational scale); RHS is never shared."""
    work.tick();require(isinstance(hz,SparseHZono) and type(hz.frame_id) is int, 'explicit native frame')
    require(len(factors) == hz.n_cont+hz.n_bin and
            [f[1] for f in factors] == ['continuous']*hz.n_cont+['binary']*hz.n_bin,
            'native continuous/binary factor map')
    factor_raw=compact(factors);require(len(factor_raw)<=POLICY['header_bytes'],'factor metadata capacity')
    factors=decode(factor_raw)
    require(type(links) is dict and all(type(i) is int and 0 <= i < hz.n_out+hz.n_eq+hz.n_ineq for i in links),
            'compression hint inventory')
    links=dict(links)
    nodes = []; roots = []
    for i, (kind, name, rhs, entries) in enumerate(native_rows(hz, work)):
        work.visit('row_visits'); local = dict(entries); refs = []
        if i in links:
            parent, scale = links[i]
            require(type(parent) is int and 0 <= parent < i and type(scale) is F and scale != 0,
                    'earlier exact compression hint')
            require(max(scale.numerator.bit_length(),scale.denominator.bit_length())<=POLICY['stored_bits'],
                    'compression hint rational capacity')
            for j, v in _row_at(hz,parent,work)[3]:
                work.visit('coefficient_visits')
                local[j] = work.exact(local.get(j, F(0))-work.exact(scale*v))
            refs = [[parent, str(scale)]]
        nodes.append([name, 0, [[j, str(v)] for j, v in sorted(local.items()) if v], refs])
        roots.append([name, 0, kind, str(rhs), i])
    experts = sorted({f[2] for f in factors if f[2] != -1})
    doc = dict(schema=SCHEMA, request=request, source=native_identity(hz, factors, work), frame=hz.frame_id,
        factors=factors, domains=[dict(name='given-domain', frame=hz.frame_id, experts=experts,
        guard=None, parents=[])], target=0, nodes=nodes, roots=roots)
    raw = compact(doc)
    graph = Graph(raw, expected_graph=digest(raw), expected_request=request,
                  expected_source=doc['source'], work=work)
    bridge = check_native_rows(graph, hz, factors, work=work)
    return graph, raw, bridge


def support(hz, *, request, factors, links, weights, offset, duals, deadline):
    """One cumulative native intake/bridge/exact-support budget; no solver."""
    from scoped_source.coefficient_dag import query
    work = Work(deadline)
    graph, raw, bridge = prepare(hz, request=request, factors=factors, links=links, work=work)
    q = query(graph, weights, offset)
    bound = evaluate(graph, q, duals, expected_query=identity(q), work=work)
    return dict(bound=bound, bridge=bridge, work=dict(work.counts), graph=raw)
