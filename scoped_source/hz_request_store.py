"""Immutable given-HZ row views and direct exact block consumption.

Standard library only. This checks a supplied representation, not its network
provenance. No source producer, optimizer, numerical package or legacy HZ parser.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import time
from types import MappingProxyType

SCHEMA = 'REQUEST_OWNED_HZ_STORE_V1'
POLICY = MappingProxyType(dict(factors=32768, active=16384, views=128, depth=64,
    view_rows=16384, rows=65536, nnz=1048576, members=4096, page_rows=64,
    header_bytes=8*2**20, member_bytes=2**20, total_bytes=128*2**20,
    stored_bits=4096, work_bits=8192, visits=8000000))


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw): return hashlib.sha256(raw).hexdigest()
def identity(value): return digest(compact(value))


def require(ok, reason):
    if not ok: raise ValueError(reason)


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'mandatory identity')
    return value


class Work:
    def __init__(self, deadline):
        require(type(deadline) in (int, float) and math.isfinite(deadline)
                and deadline-time.monotonic() <= 300, 'one finite at-most-300s deadline')
        self.deadline = deadline
        self.counts = dict(row_visits=0, coefficient_visits=0, arithmetic=0)
        self.tick()

    def tick(self):
        if time.monotonic() >= self.deadline: raise TimeoutError('store deadline')

    def visit(self, kind, amount=1):
        self.tick(); self.counts[kind] += amount
        require(sum(self.counts.values()) <= POLICY['visits'], 'store work capacity')

    def exact(self, q):
        self.visit('arithmetic')
        require(max(q.numerator.bit_length(), q.denominator.bit_length()) <= POLICY['work_bits'],
                'store intermediate rational capacity')
        return q


def rational(value):
    require(type(value) is str and len(value) <= 2470
            and re.fullmatch(r'-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?', value), 'canonical stored rational')
    q = F(value)
    require(str(q) == value and max(q.numerator.bit_length(), q.denominator.bit_length()) <= POLICY['stored_bits'],
            'store rational capacity/canonical form')
    return q


def decode(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read(root, name, limit, expected):
    require(name == 'manifest.json' or re.fullmatch(r'rows/[0-9a-f]{64}\.json', name), 'store member path')
    root = Path(root); p = root
    require(not root.is_symlink(), 'store symlink')
    for part in Path(name).parts:
        p = p/part; require(not p.is_symlink(), 'store symlink')
    require(p.is_file() and p.stat().st_size <= limit, 'store member absent/capacity')
    with p.open('rb') as stream: raw = stream.read(limit+1)
    require(len(raw) <= limit and digest(raw) == sha(expected), 'store member bytes/identity')
    value = decode(raw)
    require(compact(value) == raw, 'canonical store member bytes')
    return value, len(raw)


@dataclass(frozen=True)
class Row:
    name: str
    rhs: F
    coefficients: tuple


@dataclass(frozen=True)
class Rows:
    parent: object
    pages: tuple
    count: int

    def __iter__(self):
        if self.parent is not None: yield from self.parent
        for page in self.pages: yield from page

    def suffix(self, ancestor):
        if self is ancestor: return
        require(self.parent is not None, 'constraint ancestor not retained')
        yield from self.parent.suffix(ancestor)
        for page in self.pages: yield from page


@dataclass(frozen=True)
class View:
    name: str
    scope: str
    guard: object
    parent: object
    frame: int
    factors: frozenset
    output: Rows
    eq: Rows
    ine: Rows
    depth: int


class Store:
    """Own immutable parsed rows. Repeated views retain row identity, not copies."""
    __slots__ = ('request', 'identity', 'experts', 'factors', 'views', 'pages',
                 'common', 'templates', 'entries', 'stats', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('immutable checked store')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('immutable checked store')

    def __init__(self, root, *, expected_manifest, expected_request, work):
        work.tick()
        m, total = read(root, 'manifest.json', POLICY['header_bytes'], expected_manifest)
        require(type(m) is dict and set(m) == {'schema','request','policy','experts','factors','members','views','roles'}
                and m['schema'] == SCHEMA and m['request'] == sha(expected_request)
                and m['policy'] == dict(POLICY), 'store manifest/request/policy')
        e = m['experts']; require(type(e) is int and 2 <= e <= 8, 'store expert capacity')
        self.request, self.identity, self.experts = expected_request, expected_manifest, e
        scopes = {'common'} | {f'expert/{i}' for i in range(e)}
        registry = m['factors']
        require(type(registry) is list and 1 <= len(registry) <= POLICY['factors'], 'store factor capacity')
        names = set(); factors = []
        for f in registry:
            work.tick()
            require(type(f) is list and len(f) == 4, 'factor fields')
            name, kind, scope, owner = f
            require(type(name) is str and 0 < len(name) <= 200 and name not in names
                    and kind in ('continuous','binary') and type(scope) is str and scope in scopes
                    and type(owner) is str and 0 < len(owner) <= 48, 'factor identity/kind/owner')
            names.add(name); factors.append(tuple(f))
        self.factors = tuple(factors)
        refs = m['members']; require(type(refs) is list and len(refs) <= POLICY['members'], 'store member count')
        require(all(type(r) is dict and set(r) == {'file','bytes','sha256'}
                    and type(r['sha256']) is str for r in refs), 'member inventory fields')
        require([r['sha256'] for r in refs] == sorted(set(r['sha256'] for r in refs)), 'member inventory order/alias')
        require(all(type(r['bytes']) is int and 0 < r['bytes'] <= POLICY['member_bytes'] for r in refs), 'row member byte capacity')
        require(total+sum(r['bytes'] for r in refs) <= POLICY['total_bytes'], 'store total byte capacity')
        pages = {}; page_kind = {}; row_names = set(); nnz = 0; used_factors = set(); files = {'manifest.json'}
        for ref in refs:
            work.tick()
            require(set(ref) == {'file','bytes','sha256'} and type(ref['bytes']) is int
                    and 0 < ref['bytes'] <= POLICY['member_bytes']
                    and ref['file'] == f"rows/{sha(ref['sha256'])}.json", 'row member reference')
            total += ref['bytes']; require(total <= POLICY['total_bytes'], 'store total byte capacity')
            page, size = read(root, ref['file'], POLICY['member_bytes'], ref['sha256'])
            require(size == ref['bytes'] and type(page) is dict and set(page) == {'kind','rows'} and page['kind'] in ('output','eq','ine'), 'row member fields/length')
            require(type(page['rows']) is list and 1 <= len(page['rows']) <= POLICY['page_rows'], 'row page capacity')
            parsed = []
            for raw in page['rows']:
                work.visit('row_visits')
                require(type(raw) is list and len(raw) == 3, 'row fields')
                name, rhs, entries = raw
                require(type(name) is str and 0 < len(name) <= 200 and name not in row_names, 'unique row identity')
                row_names.add(name); require(len(row_names) <= POLICY['rows'], 'unique row capacity')
                require(type(entries) is list, 'row coefficient type')
                nnz += len(entries); require(nnz <= POLICY['nnz'], 'unique coefficient capacity')
                coefficients = []; last = -1
                for item in entries:
                    work.visit('coefficient_visits')
                    require(type(item) is list and len(item) == 2, 'coefficient fields')
                    j, value = item
                    require(type(j) is int and last < j < len(factors), 'canonical coefficient column')
                    q = rational(value); require(q != 0, 'explicit zero coefficient')
                    coefficients.append((j,q)); used_factors.add(j); last = j
                parsed.append(Row(name, rational(rhs), tuple(coefficients)))
            pages[ref['sha256']] = tuple(parsed); page_kind[ref['sha256']] = page['kind']; files.add(ref['file'])
        actual = set()
        for p in Path(root).rglob('*'):
            work.tick(); require(not p.is_symlink(), 'store symlink inventory')
            if p.is_file(): actual.add(p.relative_to(root).as_posix())
        require(actual == files, 'complete store member inventory')
        views = {}; referenced = set(); assigned = set()
        require(type(m['views']) is list and 1 <= len(m['views']) <= POLICY['views'], 'store view capacity')
        for v in m['views']:
            work.tick()
            require(type(v) is dict and set(v) == {'name','scope','guard','parent','frame','add_factors','output','eq','ine'}, 'view fields')
            name, scope = v['name'], v['scope']
            require(type(name) is str and 0 < len(name) <= 200 and name not in views
                    and type(scope) is str and scope in scopes, 'view name/scope')
            require(v['parent'] is None or type(v['parent']) is str, 'view parent type')
            parent = None if v['parent'] is None else views.get(v['parent'])
            require(v['parent'] is None and not views or parent is not None, 'view parent order/root')
            guard = v['guard']
            require(guard is None or type(guard) is list and len(guard) == 2
                    and all(type(i) is int for i in guard) and 0 <= guard[0] < guard[1] < e, 'view guard')
            guard = None if guard is None else tuple(guard)
            require(type(v['frame']) is int and (parent is None or parent.frame == v['frame']), 'view frame')
            added = v['add_factors']
            require(type(added) is list and all(type(i) is int and 0 <= i < len(factors) for i in added)
                    and added == sorted(set(added)) and not assigned.intersection(added), 'factor addition identity/alias')
            require(all(factors[i][2] == scope for i in added), 'factor addition scope')
            active = frozenset(added) | (parent.factors if parent else frozenset())
            require(0 < len(active) <= POLICY['active'], 'active factor capacity')
            depth = parent.depth+1 if parent else 1; require(depth <= POLICY['depth'], 'view depth capacity')
            if parent:
                require(parent.scope == scope or parent.scope == 'common' and parent.guard is None,
                        'expert-private inheritance')
                require(parent.guard is None or guard == parent.guard, 'guard fact promoted to larger domain')
            if guard is not None:
                require(scope == 'common' and parent is not None and not added and not v['eq']
                        and v['output'] == m['views'][list(views).index(parent.name)]['output'], 'guard changes common state')
            sequences = {}
            for kind in ('output','eq','ine'):
                ids = v[kind]
                require(type(ids) is list and all(type(x) is str for x in ids) and len(ids) == len(set(ids))
                        and all(x in pages and page_kind[x] == kind for x in ids), 'view row coverage/type')
                inherited = getattr(parent, kind) if parent and kind != 'output' else None
                local_pages = tuple(pages[x] for x in ids)
                seq = Rows(inherited, local_pages, (inherited.count if inherited else 0)+sum(map(len,local_pages)))
                require(seq.count <= POLICY['view_rows'], 'view row capacity')
                seen = set()
                for row in seq:
                    work.visit('row_visits')
                    require(row.name not in seen, 'duplicate inherited row coverage'); seen.add(row.name)
                    for j,_ in row.coefficients:
                        work.visit('coefficient_visits')
                        require(j in active and factors[j][2] in ('common',scope), 'private factor contamination')
                referenced.update(ids); sequences[kind] = seq
            require(sequences['output'].count > 0 and sum(x.count for x in sequences.values()) <= POLICY['view_rows'], 'view row capacity')
            views[name] = View(name,scope,guard,parent,v['frame'],active,**sequences,depth=depth)
            assigned.update(added)
        require(assigned == set(range(len(factors))) and referenced == set(pages), 'unassigned factors/unused row members')
        roles = m['roles']; require(type(roles) is dict and set(roles) == {'common','templates','entries'}, 'store role fields')
        require(type(roles['common']) is str and roles['common'] in views and type(roles['templates']) is list and len(roles['templates']) == e
                and all(type(n) is str and n in views for n in roles['templates']), 'common/expert role inventory')
        common = views[roles['common']]
        require(common.scope == 'common' and common.guard is None, 'unconditional common role')
        templates = tuple(views[n] for n in roles['templates'])
        for i,v in enumerate(templates):
            require(v.scope == f'expert/{i}' and v.guard is None and common.factors <= v.factors, 'expert role binding')
            for k in ('eq','ine'):
                for _ in getattr(v,k).suffix(getattr(common,k)): work.tick()
        require(type(roles['entries']) is list
                and all(type(x) is dict and set(x) == {'pair','view'} for x in roles['entries'])
                and [x['pair'] for x in roles['entries']] == [list(p) for p in combinations(range(e),2)], 'all unordered pair roles')
        entries = {}
        for r in roles['entries']:
            require(type(r['view']) is str and r['view'] in views, 'entry role fields')
            v = views[r['view']]; p = tuple(r['pair'])
            require(v.parent is common and v.guard == p and v.factors == common.factors, 'pair guard role binding')
            entries[p] = v
        self.views = MappingProxyType(views); self.pages = MappingProxyType(pages)
        self.common, self.templates, self.entries = common, templates, MappingProxyType(entries)
        self.stats = MappingProxyType(dict(factors=len(factors), views=len(views), row_members=len(pages),
            unique_rows=len(row_names), unique_nnz=nnz, referenced_factors=len(used_factors), stored_bytes=total))
        work.tick(); self._sealed = True

    def pair(self, pair):
        p = tuple(pair); require(p in self.entries, 'registered pair required')
        a,b = (self.templates[i] for i in p); entry = self.entries[p]
        active = a.factors | b.factors
        require(len(active) <= POLICY['active'], 'pair factor capacity')
        return entry,a,b,active

    def constraints(self, pair, kind):
        require(kind in ('eq','ine'), 'constraint kind')
        entry,a,b,_ = self.pair(pair)
        yield from getattr(entry,kind)
        yield from getattr(a,kind).suffix(getattr(self.common,kind))
        yield from getattr(b,kind).suffix(getattr(self.common,kind))


def evaluate(store, pair, *, left, right, offset, equality, inequality, work):
    """Exact original-row residual; no LP expansion or numerical solver."""
    work.tick(); entry,a,b,active = store.pair(pair)
    require(type(left) is list and type(right) is list and len(left) == a.output.count
            and len(right) == b.output.count, 'objective row coverage')
    residual = {j:F(0) for j in active}; lower = rational(offset)
    for view,weights in ((a,left),(b,right)):
        for row,w in zip(view.output,weights):
            work.visit('row_visits'); w = rational(w); lower = work.exact(lower+work.exact(w*row.rhs))
            for j,v in row.coefficients:
                work.visit('coefficient_visits'); residual[j] = work.exact(residual[j]+work.exact(w*v))
    for kind,values in (('eq',equality),('ine',inequality)):
        needed = (getattr(entry,kind).count + getattr(a,kind).count
                  + getattr(b,kind).count - 2*getattr(store.common,kind).count)
        require(type(values) is list and len(values) == needed, 'dual row coverage')
        for row,v in zip(store.constraints(pair,kind),values):
            work.visit('row_visits'); v = rational(v)
            require(kind != 'ine' or v <= 0, 'inequality dual sign')
            lower = work.exact(lower+work.exact(v*row.rhs))
            for j,c in row.coefficients:
                work.visit('coefficient_visits'); residual[j] = work.exact(residual[j]-work.exact(v*c))
    for j in sorted(residual): lower = work.exact(lower-abs(residual[j]))
    work.tick()
    return lower, tuple((j,residual[j]) for j in sorted(residual))


def check_candidate(store, query, candidate, *, expected_query, work):
    """A received dual/claimed bound is never trusted, even with a valid hash."""
    work.tick()
    require(type(query) is dict and set(query) == {'store','request','pair','left','right','offset'}
            and type(candidate) is dict and set(candidate) == {'query','equality','inequality','lower'},
            'query/candidate fields')
    # Capture caller-owned lists before any arithmetic or observer can change
    # them. This is bounded private input, not a hash of live mutable aliases.
    for values in (query['left'],query['right'],candidate['equality'],candidate['inequality']):
        require(type(values) is list and len(values) <= 3*POLICY['view_rows']
                and all(type(v) is str and len(v) <= 2470 for v in values), 'candidate vector capacity/type')
    require(type(query['pair']) is list and len(query['pair']) == 2
            and all(type(i) is int for i in query['pair']), 'query pair type')
    require(all(type(v) is str and len(v) <= 2470 for v in
                (query['store'],query['request'],query['offset'],candidate['query'],candidate['lower'])),
            'candidate scalar capacity/type')
    estimated = sum(sum(len(v)+3 for v in values) for values in
                    (query['left'],query['right'],candidate['equality'],candidate['inequality']))+10000
    require(estimated <= POLICY['header_bytes'], 'candidate capture capacity')
    query, candidate = decode(compact(query)), decode(compact(candidate))
    work.tick()
    require(identity(query) == sha(expected_query)
            and query['store'] == store.identity and query['request'] == store.request,
            'query external binding')
    require(candidate['query'] == expected_query, 'candidate query binding')
    bound, residual = evaluate(store, query['pair'], left=query['left'], right=query['right'],
        offset=query['offset'], equality=candidate['equality'], inequality=candidate['inequality'], work=work)
    require(rational(candidate['lower']) == bound, 'candidate claimed bound')
    work.tick(); return bound, residual
