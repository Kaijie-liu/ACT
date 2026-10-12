"""Exact coefficient-reference support. Standard library; given rows only.

The external source identity is a binding, NOT proof of network/HZ lowering.
Roots retain their own rhs and inventory. Only coefficient vectors are shared.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from types import MappingProxyType

from scoped_source.hz_request_store import (POLICY, compact, decode, digest,
    rational, require, sha)

SCHEMA = 'HZ_COEFFICIENT_DAG_V1'
QUERY = 'HZ_COEFFICIENT_DAG_QUERY_V1'


@dataclass(frozen=True)
class Domain:
    name: str
    experts: tuple
    guard: object
    ancestors: frozenset
    depth: int


@dataclass(frozen=True)
class Node:
    name: str
    domain: int
    local: tuple
    refs: tuple
    depth: int


@dataclass(frozen=True)
class Root:
    name: str
    domain: int
    kind: str
    rhs: F
    node: int


def index(value, size, reason):
    require(type(value) is int and 0 <= value < size, reason)
    return value


def label(value):
    require(type(value) is str and 0 < len(value) <= 160, 'bounded nonempty name')
    return value


def rational_entries(entries, size, work):
    require(type(entries) is list and len(entries) <= size, 'entry inventory')
    result = []; previous = -1
    for item in entries:
        work.visit('coefficient_visits')
        require(type(item) is list and len(item) == 2, 'entry pair')
        j = index(item[0], size, 'entry column/reference')
        require(j > previous, 'sorted unique entries')
        q = rational(item[1]); require(q != 0, 'explicit zero entry')
        result.append((j, q)); previous = j
    return tuple(result)


class Graph:
    """Own immutable parsed local vectors, not recursively expanded vectors."""
    __slots__ = ('identity', 'request', 'source', 'frame', 'factors', 'domains',
                 'target', 'nodes', 'roots', 'active', 'stats', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('immutable coefficient graph')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('immutable coefficient graph')

    def __init__(self, raw, *, expected_graph, expected_request, expected_source, work):
        work.tick()
        require(type(raw) is bytes and len(raw) <= POLICY['total_bytes'], 'graph byte capacity')
        require(digest(raw) == sha(expected_graph), 'graph identity')
        doc = decode(raw)
        require(compact(doc) == raw, 'canonical graph bytes')
        require(type(doc) is dict and set(doc) == {'schema', 'request', 'source', 'frame',
            'factors', 'domains', 'target', 'nodes', 'roots'} and doc['schema'] == SCHEMA,
            'graph schema')
        require(doc['request'] == sha(expected_request) and doc['source'] == sha(expected_source),
                'request/source identity')
        require(type(doc['frame']) is int and 0 <= doc['frame'] < 2**63, 'frame identity')
        self.identity, self.request, self.source = expected_graph, expected_request, expected_source
        self.frame = doc['frame']; factors = []; names = set()
        require(type(doc['factors']) is list and 0 < len(doc['factors']) <= POLICY['factors'],
                'factor capacity')
        for item in doc['factors']:
            work.visit('row_visits')
            require(type(item) is list and len(item) == 3, 'factor record')
            name, kind, owner = item; label(name)
            require(name not in names and kind in ('continuous', 'binary') and
                type(owner) is int and -1 <= owner < 8, 'factor identity/owner')
            names.add(name); factors.append((name, kind, owner))
        self.factors = tuple(factors)
        require(type(doc['domains']) is list and 0 < len(doc['domains']) <= POLICY['views'],
                'domain capacity')
        domains = []; names = set()
        for di, item in enumerate(doc['domains']):
            work.visit('row_visits')
            require(type(item) is dict and set(item) == {'name', 'frame', 'experts', 'guard', 'parents'},
                    'domain record')
            name = label(item['name']); experts = item['experts']; parents = item['parents']
            require(name not in names and item['frame'] == self.frame and type(item['frame']) is int,
                    'domain name/frame')
            require(type(experts) is list and len(experts) <= 8 and
                all(type(e) is int and 0 <= e < 8 for e in experts) and experts == sorted(set(experts)),
                'domain expert roster')
            guard = item['guard']
            if guard is not None:
                require(type(guard) is list and len(guard) == 2 and
                    all(type(e) is int and e in experts for e in guard) and guard[0] < guard[1],
                    'domain guard')
                guard = tuple(guard)
            require(type(parents) is list and all(type(p) is int and 0 <= p < di for p in parents)
                    and parents == sorted(set(parents)), 'earlier domain parents')
            ancestors = {di}; depth = 1
            for p in parents:
                parent = domains[p]
                require(set(parent.experts) <= set(experts) and
                    (parent.guard is None or parent.guard == guard), 'domain weakening/guard promotion')
                ancestors.update(parent.ancestors); depth = max(depth, parent.depth+1)
            require(depth <= POLICY['depth'], 'domain depth')
            domains.append(Domain(name, tuple(experts), guard, frozenset(ancestors), depth)); names.add(name)
        self.domains = tuple(domains)
        self.target = index(doc['target'], len(domains), 'target domain')
        require(type(doc['nodes']) is list and 0 < len(doc['nodes']) <= POLICY['rows'], 'node capacity')
        nodes = []; names = set(); nnz = 0; edges = 0; active = set()
        for ni, item in enumerate(doc['nodes']):
            work.visit('row_visits')
            require(type(item) is list and len(item) == 4, 'node record')
            name, di, local, refs = item; label(name)
            require(name not in names, 'unique node name')
            index(di, len(domains), 'node domain'); domain = domains[di]
            local = rational_entries(local, len(factors), work)
            refs = rational_entries(refs, ni, work)
            nnz += len(local); edges += len(refs)
            require(nnz <= POLICY['nnz'] and edges <= 65536, 'local nnz/reference capacity')
            depth = 1
            for j, _ in local:
                require(factors[j][2] == -1 or factors[j][2] in domain.experts, 'private factor visibility')
                active.add(j)
            for parent, _ in refs:
                require(nodes[parent].domain in domain.ancestors, 'reference domain ancestry')
                depth = max(depth, nodes[parent].depth+1)
            require(depth <= POLICY['depth'], 'node depth')
            nodes.append(Node(name, di, local, refs, depth)); names.add(name)
        require(len(active) <= POLICY['active'], 'active factor capacity')
        self.nodes = tuple(nodes); self.active = tuple(sorted(active))
        require(type(doc['roots']) is list and 0 < len(doc['roots']) <= POLICY['rows'], 'root capacity')
        roots = []; names = set(); reached = set()
        for item in doc['roots']:
            work.visit('row_visits')
            require(type(item) is list and len(item) == 5, 'root record')
            name, di, kind, rhs, ni = item; label(name)
            require(name not in names and kind in ('output', 'eq', 'ine'), 'root identity/kind')
            index(di, len(domains), 'root domain'); index(ni, len(nodes), 'root node')
            require(di in domains[self.target].ancestors and nodes[ni].domain in domains[di].ancestors,
                    'root domain ancestry')
            roots.append(Root(name, di, kind, rational(rhs), ni)); names.add(name); reached.add(ni)
        require(any(r.kind == 'output' for r in roots), 'missing outputs')
        for ni in range(len(nodes)-1, -1, -1):
            work.visit('row_visits')
            if ni in reached: reached.update(j for j, _ in nodes[ni].refs)
        require(len(reached) == len(nodes), 'unreferenced coefficient node')
        self.roots = tuple(roots)
        self.stats = MappingProxyType(dict(nodes=len(nodes), roots=len(roots), local_nnz=nnz,
            references=edges, active=len(active), bytes=len(raw)))
        work.tick(); self._sealed = True


def query(graph, weights, offset='0'):
    return dict(schema=QUERY, graph=graph.identity, request=graph.request, source=graph.source,
                target=graph.target, weights=list(weights), offset=offset)


@dataclass(frozen=True)
class Bound:
    lower: F
    constant: F
    residual: tuple
    local_coefficient_visits: int
    reference_visits: int


def evaluate(graph, q, duals, *, expected_query, work):
    """Exact support for one complete given-row query; no SAFE classification."""
    require(type(graph) is Graph and graph._sealed, 'checked graph required')
    work.tick()
    raw = compact(q)
    require(len(raw) <= POLICY['header_bytes'] and digest(raw) == sha(expected_query), 'query identity/capacity')
    q = decode(raw)  # Never read the caller-owned object after its binding check.
    require(type(q) is dict and set(q) == {'schema', 'graph', 'request', 'source', 'target', 'weights', 'offset'}
        and q['schema'] == QUERY and q['graph'] == graph.identity and q['request'] == graph.request
        and q['source'] == graph.source and type(q['target']) is int and q['target'] == graph.target,
        'query binding')
    dual_raw = compact(duals)
    require(len(dual_raw) <= POLICY['header_bytes'], 'dual byte capacity')
    duals = decode(dual_raw)
    require(type(duals) is dict and set(duals) == {'eq', 'ine'}, 'complete dual record')
    vectors = {}
    for kind in ('output', 'eq', 'ine'):
        values = q['weights'] if kind == 'output' else duals[kind]
        require(type(values) is list and len(values) == sum(r.kind == kind for r in graph.roots),
                'complete property/dual inventory')
        vectors[kind] = []
        for v in values:
            work.visit('row_visits'); value = rational(v)
            require(kind != 'ine' or value <= 0, 'inequality dual sign')
            vectors[kind].append(value)
    counters = dict(output=0, eq=0, ine=0)
    alpha = [F(0)]*len(graph.nodes); residual = [F(0)]*len(graph.factors)
    constant = rational(q['offset'])
    for root in graph.roots:
        work.visit('row_visits')
        value = vectors[root.kind][counters[root.kind]]; counters[root.kind] += 1
        constant = work.exact(constant + work.exact(value*root.rhs))
        weight = value if root.kind == 'output' else -value
        alpha[root.node] = work.exact(alpha[root.node]+weight)
    local_visits = 0; ref_visits = 0
    for ni in range(len(graph.nodes)-1, -1, -1):
        work.visit('row_visits'); node = graph.nodes[ni]; weight = alpha[ni]
        # All local data were already checked, even if the adjoint now cancels.
        if not weight: continue
        for parent, scale in node.refs:
            work.visit('coefficient_visits'); ref_visits += 1
            alpha[parent] = work.exact(alpha[parent]+work.exact(weight*scale))
        for j, value in node.local:
            work.visit('coefficient_visits'); local_visits += 1
            residual[j] = work.exact(residual[j]+work.exact(weight*value))
    lower = constant
    for j in graph.active: lower = work.exact(lower-abs(residual[j]))
    work.tick()
    return Bound(lower, constant, tuple((j, residual[j]) for j in graph.active), local_visits, ref_visits)


def check_candidate(graph, q, candidate, *, expected_query, work):
    work.tick(); raw = compact(candidate)
    require(len(raw) <= POLICY['header_bytes'], 'candidate byte capacity')
    candidate = decode(raw)
    require(type(candidate) is dict and set(candidate) == {'query', 'eq', 'ine', 'lower'}
            and candidate['query'] == expected_query, 'candidate identity/fields')
    bound = evaluate(graph, q, {k: candidate[k] for k in ('eq', 'ine')},
                     expected_query=expected_query, work=work)
    require(rational(candidate['lower']) <= bound.lower, 'candidate overclaim')
    work.tick(); return bound
