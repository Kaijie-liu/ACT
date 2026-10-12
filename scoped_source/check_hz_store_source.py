"""Exact, row-consuming source checks; no producer, NumPy, solver or old parser.

Primitive calls assume their parent was checked. ``check_source`` closes that
ordering and the complete finite declaration inventory. It proves inclusion of
the declared real operators, not native floating-point execution or output SAFE.
"""
from fractions import Fraction as F
from itertools import combinations
import math

from scoped_source.hz_request_store import (
    Row, compact, decode, digest, rational, require, sha,
)

SCHEMA = 'REQUEST_STORE_SOURCE_V1'


def binary(q):
    try:
        f = float(q)
        require(math.isfinite(f) and F(f) == q, 'source target not exact binary64')
    except (OverflowError, ValueError):
        raise ValueError('source target not exact binary64') from None
    return q


def capture(doc, expected, work):
    raw = compact(doc)
    require(len(raw) <= 4*1024*1024, 'source declaration byte capacity')
    require(digest(raw) == sha(expected), 'source declaration identity')
    result = decode(raw); work.tick()
    return result


def polynomial(row, work):
    work.visit('row_visits'); work.exact(row.rhs)
    result = {}
    for j, q in row.coefficients:
        work.visit('coefficient_visits'); result[j] = work.exact(q)
    return row.rhs, result


def combine(rows, weights, bias, work):
    constant = work.exact(bias); coeff = {}
    for j, w in weights:
        require(type(j) is int and 0 <= j < len(rows), 'operator coordinate')
        w = work.exact(w)
        c, values = polynomial(rows[j], work)
        constant = work.exact(constant + work.exact(w*c))
        for k, v in values.items():
            coeff[k] = work.exact(coeff.get(k,F(0)) + work.exact(w*v))
    return constant, {k:v for k,v in coeff.items() if v}


def distance(a, b, work):
    result = F(0)
    for j in sorted(set(a) | set(b)):
        work.visit('coefficient_visits')
        result = work.exact(result + abs(work.exact(a.get(j,F(0))-b.get(j,F(0)))))
    return result


class Allocation:
    def __init__(self, store, source, target, work):
        require(target.parent is source and target.frame == source.frame,
                'source parent/frame')
        require(target.eq.parent is source.eq and target.ine.parent is source.ine,
                'source inherited constraints')
        self.ids = iter(sorted(target.factors-source.factors))
        self.store, self.source, self.target, self.work = store, source, target, work
        self.used = []
        self.basis = None

    def take(self, label, kind='continuous'):
        self.work.tick()
        j = next(self.ids, None)
        require(j is not None and self.store.factors[j] ==
                (f'{self.target.name}/{label}',kind,self.target.scope,self.target.name),
                'source factor allocation/kind/owner')
        self.used.append(j)
        return j

    def done(self):
        require(next(self.ids, None) is None, 'extra source factor')

    def begin_enclosure(self):
        self.basis = self.source.factors | frozenset(self.used)


def enclosed(reference, rows, kind, allocation, work):
    """Check new rows only. Original row coordinates remain common globally."""
    require(len(reference) == len(rows), 'source new row inventory')
    for i, ((c, coeff), row) in enumerate(zip(reference, rows)):
        work.visit('row_visits')
        require(row.name == f'{allocation.target.name}/{kind}/{i}', 'source row identity')
        binary(row.rhs)
        actual = dict(row.coefficients)
        for q in actual.values(): binary(q)
        # Enclosure columns may not be repurposed as original coordinates.
        original = allocation.basis
        require(original is not None and set(coeff) <= original, 'source reference factor scope')
        nominal = {j:q for j,q in actual.items() if j in original}
        coefficient_error = distance(coeff, nominal, work)
        if kind == 'ine':
            require(actual == nominal and row.rhs >= work.exact(c+coefficient_error),
                    'inward source inequality')
        else:
            needed = work.exact(abs(work.exact(c-row.rhs))+coefficient_error)
            remainder = {j:q for j,q in actual.items() if j not in original}
            if needed:
                j = allocation.take(f'lift/{kind}/{i}')
                require(set(remainder) == {j} and remainder[j] >= needed,
                        'missing/inward source enclosure')
            else:
                require(not remainder, 'unnecessary source enclosure column')


def check_input(store, lower, upper, work):
    view = store.views['input']; rows = tuple(view.output)
    require(view.parent is None and view.scope == 'common' and view.guard is None
            and view.eq.count == view.ine.count == 0,
            'input unguarded unconstrained root')
    require(len(lower) == len(upper) == len(rows) and 1 <= len(rows) <= 4096
            and view.factors == frozenset(range(len(rows))), 'input coordinate inventory')
    for i,(row,l,u) in enumerate(zip(rows,lower,upper)):
        work.visit('row_visits'); l,u = rational(l),rational(u)
        require(l <= u and store.factors[i] == (f'input/c/{i}','continuous','common','input'),
                'input source/factor identity')
        require(row.name == f'input/output/{i}', 'input row identity')
        c = binary(row.rhs); coeff = dict(row.coefficients)
        require(set(coeff) <= {i}, 'input coordinate map')
        radius = binary(coeff.get(i,F(0)))
        require(radius >= 0 and work.exact(c-radius) <= l and work.exact(c+radius) >= u,
                'input enclosure inward')
    return view


def check_affine(store, source, target, layer, work):
    allocation = Allocation(store, source, target, work)
    require(set(layer) == {'kind','rows'} and layer['kind'] == 'affine', 'affine fields')
    params = layer['rows']; rows = tuple(source.output); actual = tuple(target.output)
    require(type(params) is list and 1 <= len(params) <= 4096 and len(params) == len(actual)
            and len(rows) <= 4096, 'affine output inventory')
    require(target.eq.count == source.eq.count and target.ine.count == source.ine.count,
            'affine added constraints')
    reference = []; nnz = 0
    for i,(param,out) in enumerate(zip(params,actual)):
        require(type(param) is list and len(param) == 2 and type(param[1]) is list,
                'affine parameter fields')
        bias = binary(rational(param[0])); weights = []; last = -1
        for item in param[1]:
            require(type(item) is list and len(item) == 2, 'affine coefficient fields')
            j,w = item; require(type(j) is int and last < j < len(rows), 'affine parameter order')
            w = binary(rational(w)); require(w != 0, 'explicit zero operator coefficient')
            last = j; weights.append((j,w)); nnz += 1
            require(nnz <= 262144, 'affine parameter nnz capacity')
        c, expected = combine(rows,weights,bias,work)
        nominal_c = binary(out.rhs)
        nominal = {j:binary(q) for j,q in out.coefficients if j in source.factors}
        needed = work.exact(abs(work.exact(c-nominal_c))+distance(expected,nominal,work))
        ref = dict(nominal)
        if needed: ref[allocation.take(f'error/{i}')] = needed
        reference.append((nominal_c,ref))
    allocation.begin_enclosure()
    enclosed(reference,actual,'output',allocation,work)
    allocation.done()


def check_relu(store, source, target, work):
    allocation = Allocation(store,source,target,work)
    outputs, eq, negative, positive = [],[],[],[]
    branches = dict(active=0,inactive=0,unstable=0)
    for i,row in enumerate(source.output):
        c, coeff = polynomial(row,work)
        radius = F(0)
        for v in coeff.values(): radius = work.exact(radius+abs(v))
        lo,hi = work.exact(c-radius),work.exact(c+radius)
        branch = 'active' if lo >= 0 else 'inactive' if hi <= 0 else 'unstable'
        branches[branch] += 1
        if branch == 'active': outputs.append((c,coeff))
        elif branch == 'inactive': outputs.append((F(0),{}))
        else:
            u = allocation.take(f'negative/{i}'); v = allocation.take(f'positive/{i}')
            z = allocation.take(f'sign/{i}','binary')
            outputs.append((work.exact(hi/2),{v:work.exact(-hi/2)}))
            values = {j:-q for j,q in coeff.items()}
            values.update({u:work.exact(lo/2),v:work.exact(-hi/2),z:work.exact(lo/2)})
            eq.append((work.exact(c-hi/2),values))
            negative.append((F(0),{u:F(-1),z:F(-1)}))
            positive.append((F(0),{v:F(-1),z:F(1)}))
    allocation.begin_enclosure()
    enclosed(outputs,tuple(target.output),'output',allocation,work)
    enclosed(eq,tuple(target.eq.suffix(source.eq)),'eq',allocation,work)
    enclosed(negative+positive,tuple(target.ine.suffix(source.ine)),'ine',allocation,work)
    allocation.done()
    return branches


def check_common(store, initial, router, work):
    target = store.common; allocation = Allocation(store,router,target,work)
    require(target.scope == 'common' and target.guard is None
            and target.eq.count == router.eq.count and target.ine.count == router.ine.count
            and target.output.count == initial.output.count
            and all(a is b for a,b in zip(target.output,initial.output)), 'common input restoration')
    allocation.done()


def check_guard(store, router, pair, work):
    source,target = store.common,store.entries[pair]
    allocation = Allocation(store,source,target,work)
    require(target.output.count == source.output.count
            and all(a is b for a,b in zip(source.output,target.output))
            and target.eq.count == source.eq.count, 'guard changes common state')
    router_rows = tuple(router.output)
    require(len(router_rows) == store.experts, 'router output inventory')
    reference = []
    for a in pair:
        for outside in range(store.experts):
            if outside in pair: continue
            c, values = combine(router_rows,[(outside,F(1)),(a,F(-1))],F(0),work)
            reference.append((-c,values))
    allocation.begin_enclosure()
    enclosed(reference,tuple(target.ine.suffix(source.ine)),'ine',allocation,work)
    allocation.done()


def check_source(store, declaration, *, expected_source, work):
    doc = capture(declaration,expected_source,work)
    require(store.request == expected_source and set(doc) == {'schema','lower','upper','router','experts'}
            and doc['schema'] == SCHEMA, 'source declaration/store binding')
    require(type(doc['experts']) is list and len(doc['experts']) == store.experts,
            'source expert inventory')
    initial = check_input(store,doc['lower'],doc['upper'],work)
    consumed = ['input']; reports = []; width = None
    def network(layers, parent, prefix, scope):
        require(type(layers) is list and 1 <= len(layers) <= 16, 'source layer inventory')
        for i,layer in enumerate(layers):
            work.tick(); name = f'{prefix}/{i}'; require(name in store.views, 'missing source layer')
            target = store.views[name]
            require(target.scope == scope and target.guard is None, 'source layer scope')
            require(type(layer) is dict, 'source layer type')
            if layer.get('kind') == 'affine': check_affine(store,parent,target,layer,work)
            else:
                require(layer == {'kind':'relu'}, 'unsupported/extra source parameter')
                reports.append(dict(view=name,**check_relu(store,parent,target,work)))
            consumed.append(name); parent = target
        return parent
    router = network(doc['router'],initial,'router','common')
    require(router.output.count == store.experts, 'router class count')
    check_common(store,initial,router,work); consumed.append(store.common.name)
    for i,layers in enumerate(doc['experts']):
        v = network(layers,store.common,f'expert{i}',f'expert/{i}')
        require(v is store.templates[i], 'source expert terminal binding')
        if width is None: width = v.output.count
        require(v.output.count == width and width >= 2, 'source expert property dimensions')
    for p in combinations(range(store.experts),2):
        check_guard(store,router,p,work); consumed.append(store.entries[p].name)
    require(consumed == list(store.views), 'complete source view order/inventory')
    work.tick()
    return dict(status='CHECKED_DECLARED_SOURCE_STORE', source=expected_source,
                store=store.identity, views=len(consumed), relu=reports,
                experts=store.experts, classes=width,
                complete_output_proof=False, native_float_proof=False)
