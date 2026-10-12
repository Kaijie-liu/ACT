"""Untrusted captured-source lowering into a coefficient forest.

No solve. Independent acceptance is in check_source_dag. Native affine workspaces
contain output matrices only; inherited constraints remain in the forest.
"""
from fractions import Fraction as F
from itertools import combinations
import time

from scoped_source.hz_request_store import POLICY, compact, decode, identity, require
from scoped_source.hz_store_source import nearest, upward
from scoped_source.tensor_rows import TensorRows

SCHEMA = 'SOURCE_COEFFICIENT_FOREST_V1'


class Builder:
    schema = SCHEMA

    def __init__(self, reader, work):
        self.reader, self.work = reader, work
        self.factors, self.nodes, self.views = [], [], []
        self.norms = []
        self.local_nnz = self.reference_count = 0
        self.native_seconds = 0.
        self.native_coordinates = 0

    def factor(self, name, kind, scope, owner):
        self.work.visit('row_visits')
        require(len(self.factors) < POLICY['factors'], 'factor capacity')
        j = len(self.factors); self.factors.append([name, kind, scope, owner]); return j

    def local_norm(self, local):
        n = F(0)
        for _, q in local: n = self.work.exact(n+abs(q))
        return n

    def document_extra(self): return {}

    def node(self, owner, local, refs=()):
        self.work.visit('row_visits')
        require(len(self.nodes) < POLICY['rows'], 'node capacity')
        local = tuple(sorted((j, q) for j, q in local.items() if q))
        self.local_nnz += len(local); self.reference_count += len(refs)
        require(self.local_nnz<=POLICY['nnz'] and self.reference_count<=65536,'forest local/reference capacity')
        n = self.local_norm(local)
        # A sound triangle bound; equals L1 only for nonoverlapping terms.
        for p, s in refs: n = self.work.exact(n+abs(s)*self.norms[p])
        j = len(self.nodes); self.nodes.append([owner, local, tuple(refs)])
        self.norms.append(n); return j

    def expand(self, node):
        result = {}; stack = [(node, F(1))]
        while stack:
            j, w = stack.pop(); self.work.visit('row_visits')
            _, local, refs = self.nodes[j]
            for k, q in local:
                self.work.visit('coefficient_visits')
                result[k] = self.work.exact(result.get(k, F(0))+w*q)
            stack.extend((p, self.work.exact(w*s)) for p, s in refs)
        return {j: q for j, q in result.items() if q}

    def view(self, name, scope, parent, op, first, output, eq=(), ine=()):
        require(len(self.views) < POLICY['views'], 'view capacity')
        v = dict(name=name, scope=scope, parent=parent, op=op,
                 factors=[first, len(self.factors)], output=list(output), eq=list(eq), ine=list(ine))
        self.views.append(v); return len(self.views)-1

    def input(self):
        out = []; owner = 0
        for i, (lo, hi) in enumerate(self.reader.inputs):
            c = nearest(self.work.exact((lo+hi)/2)); r = upward(max(c-lo, hi-c))
            f = self.factor(f'input/c/{i}', 'continuous', -1, owner)
            out.append((c, self.node(owner, {f: r})))
        return self.view('input', -1, None, 'input', 0, out)

    def live_parent(self, parent):
        import numpy as np
        import scipy.sparse as sp
        from act.back_end.solver.solver_hz import SparseHZono
        rows = self.views[parent]['output']; n = len(self.factors)
        ids = [[j for j, f in enumerate(self.factors) if f[1] == kind]
               for kind in ('continuous', 'binary')]
        maps = [{j: k for k, j in enumerate(v)} for v in ids]
        ds, cs, ps = [[], []], [[], []], [[0], [0]]
        for _, node in rows:
            # Parent expansion is bounded and temporary. Never expands all constraints.
            d = self.expand(node)
            for j, q in sorted(d.items()):
                z = 0 if j in maps[0] else 1
                ds[z].append(float(q)); cs[z].append(maps[z][j]); self.native_coordinates += 1
            for z in (0, 1): ps[z].append(len(ds[z]))
        matrices = [sp.csr_matrix((ds[z], cs[z], ps[z]), shape=(len(rows), len(ids[z]))) for z in (0, 1)]
        return SparseHZono(np.array([float(c) for c, _ in rows]), *matrices,
            sp.csr_matrix((0, len(ids[0]))), sp.csr_matrix((0, len(ids[1]))),
            np.empty(0), frame_id=1, exact=False), ids

    def affine(self, parent, layer, stream):
        import numpy as np
        import scipy.sparse as sp
        from act.back_end.solver.solver_hz import sparse_hz_linear
        owner = len(self.views); first = len(self.factors)
        scope = -1 if layer.graph == 'router' else int(layer.graph[6:])
        name = f'{layer.graph}/{layer.index}'; rows = self.views[parent]['output']
        begin = time.monotonic(); live, ids = self.live_parent(parent)
        self.native_seconds += time.monotonic()-begin
        pending = []; diagonal = rows == self.views[0]['output']
        for i in range(layer.outputs):
            source = next(stream)
            require((source.graph, source.layer, source.row) == (layer.graph, layer.index, i), 'source row order')
            begin = time.monotonic()
            w = sp.csr_matrix(np.asarray([float(v) for v in source.coefficients]).reshape(1, -1))
            self.native_coordinates += len(source.coefficients)
            nominal = sparse_hz_linear(live, w, np.array([float(source.bias)]))
            self.native_seconds += time.monotonic()-begin
            nc = F(float(nominal.c[0])); nd = {}
            for mat, fids in ((nominal.Gc, ids[0]), (nominal.Gb, ids[1])):
                for k, v in zip(mat.indices, mat.data):
                    q = F(float(v))
                    if q: nd[fids[int(k)]] = q
            c = source.bias; expected = {}; error = F(0)
            if diagonal:
                allowed = set()
                for j, weight in enumerate(source.coefficients):
                    self.work.visit('coefficient_visits')
                    pc, pn = rows[j]; local = self.nodes[pn][1]
                    require(not self.nodes[pn][2] and len(local) <= 1, 'diagonal source basis')
                    if pc and weight: c = self.work.exact(c+weight*pc)
                    if local:
                        k, r = local[0]; allowed.add(k)
                        q = self.work.exact(weight*r)
                        error = self.work.exact(error+abs(q-nd.get(k, F(0))))
                require(set(nd) <= allowed, 'nominal diagonal columns')
            else:
                for j, weight in enumerate(source.coefficients):
                    self.work.visit('coefficient_visits')
                    if not weight: continue
                    pc, pn = rows[j]; c = self.work.exact(c+weight*pc)
                    for k, v in self.expand(pn).items():
                        expected[k] = self.work.exact(expected.get(k, F(0))+weight*v)
                for k in set(expected)|set(nd):
                    error = self.work.exact(error+abs(expected.get(k,F(0))-nd.get(k,F(0))))
            error = self.work.exact(error+abs(c-nc))
            delta = F(0)
            if error:
                f = self.factor(f'{name}/error/{i}', 'continuous', scope, owner)
                rounded = nearest(error); delta = abs(error-rounded)
                if rounded: nd[f] = rounded
            pending.append([nc, self.node(owner, nd), delta])
        out = []
        for i, (c, node, delta) in enumerate(pending):
            if delta:
                f=self.factor(f'{name}/lift/output/{i}', 'continuous', scope, owner); q=upward(delta)
                self.nodes[node][1]+=((f,q),); self.norms[node]=self.work.exact(self.norms[node]+q)
                self.local_nnz+=1; require(self.local_nnz<=POLICY['nnz'],'forest local/reference capacity')
            out.append((c, node))
        return self.view(name, scope, parent, 'Linear', first, out)

    def relu(self, parent, layer):
        owner = len(self.views); first = len(self.factors); name = f'{layer.graph}/{layer.index}'
        scope = -1 if layer.graph=='router' else int(layer.graph[6:])
        outputs = []; eqs = []; neg = []; pos = []
        for i, (c, pn) in enumerate(self.views[parent]['output']):
            self.work.visit('row_visits'); r = self.norms[pn]
            lo, hi = self.work.exact(c-r), self.work.exact(c+r)
            if lo >= 0: outputs.append(('alias', c, pn))
            elif hi <= 0: outputs.append(('new', F(0), {}, ()))
            else:
                u, v, z = [self.factor(f'{name}/{label}/{i}', kind, scope, owner)
                           for label, kind in (('negative','continuous'),('positive','continuous'),('sign','binary'))]
                a, b = self.work.exact(lo/2), self.work.exact(hi/2)
                outputs.append(('new', b, {v:-b}, ()))
                eqs.append(('new', self.work.exact(c-b), {u:a,v:-b,z:a}, ((pn,F(-1)),)))
                neg.append(('new',F(0),{u:F(-1),z:F(-1)},()))
                pos.append(('new',F(0),{v:F(-1),z:F(1)},()))
        def enclose(kind, seq):
            result=[]
            for i, item in enumerate(seq):
                if item[0]=='alias': result.append((item[1],item[2])); continue
                _, c, d, refs = item; nc=nearest(c); nd={j:nearest(q) for j,q in d.items()}
                error = self.work.exact(abs(c-nc)+sum(abs(q-nd[j]) for j,q in d.items()))
                if error: nd[self.factor(f'{name}/lift/{kind}/{i}','continuous',scope,owner)]=upward(error)
                result.append((nc,self.node(owner,nd,refs)))
            return result
        out=enclose('output',outputs); eq=enclose('eq',eqs); ine=enclose('ine',neg+pos)
        return self.view(name,scope,parent,'ReLU',first,out,eq,ine)

    def guard(self, owner, other, selected):
        rc,rn=other; sc,sn=selected; d=self.expand(rn)
        for k,q in self.expand(sn).items(): d[k]=self.work.exact(d.get(k,F(0))-q)
        nd={k:nearest(q) for k,q in d.items()}; delta=sum(abs(q-nd[k]) for k,q in d.items())
        rhs=upward(self.work.exact(sc-rc+delta))
        return rhs,self.node(owner,nd)

    def finish(self, cut):
        initial=self.input(); stream=iter(self.reader.rows()); common=None; router=None
        ends={}; parent=None; previous=None; stopped=False
        for layer in self.reader.plan:
            if stopped: break
            if layer.graph != previous:
                if previous == 'router':
                    router=parent
                    common=self.view('common',-1,router,'common',len(self.factors),self.views[initial]['output'])
                parent=initial if layer.graph=='router' else common; previous=layer.graph
            if layer.kind == 'Flatten':
                parent=self.view(f'{layer.graph}/{layer.index}',-1 if layer.graph=='router' else int(layer.graph[6:]),parent,
                                 'Flatten',len(self.factors),self.views[parent]['output'])
            elif layer.kind == 'Linear': parent=self.affine(parent,layer,stream)
            elif layer.kind == 'ReLU': parent=self.relu(parent,layer)
            else: raise ValueError('unsupported operator')
            ends[layer.graph]=parent
            if cut == [layer.graph, layer.index]: stopped=True
        if cut is not None: require(stopped,'prefix not found')
        for _ in stream: self.work.tick()  # source identity requires complete suffix intake
        entries=[]
        if cut is None:
            for a,b in combinations(range(self.reader.request['experts']),2):
                owner=len(self.views); inequalities=[]
                for selected in (a,b):
                    for j in range(self.reader.request['experts']):
                        if j in (a,b): continue
                        inequalities.append(self.guard(owner,self.views[router]['output'][j],
                                                       self.views[router]['output'][selected]))
                v=self.view(f'pair/{a}-{b}',-1,common,'guard',len(self.factors),
                            self.views[common]['output'],ine=inequalities)
                entries.append([a,b,v])
        roles=dict(input=initial,router=router,common=common,
                   experts=[ends.get(f'expert{i}') if cut is None else None for i in range(self.reader.request['experts'])],
                   pairs=entries,prefix=parent if cut is not None else None)
        doc=dict(schema=self.schema,source=self.reader.receipt()['source_sha256'],request=identity(self.reader.request),
                 frame=1,cut=cut,policy=dict(POLICY),factors=self.factors,
                 nodes=[[o,[[j,str(q)] for j,q in local],[[j,str(q)] for j,q in refs]] for o,local,refs in self.nodes],
                 views=[{**v,**{k:[[str(c),n] for c,n in v[k]] for k in ('output','eq','ine')}} for v in self.views],
                 roles=roles,source_receipt=self.reader.receipt())
        doc.update(self.document_extra())
        require(sum(len(n[1]) for n in self.nodes)<=POLICY['nnz'] and sum(len(n[2]) for n in self.nodes)<=65536,
                'forest local/reference capacity')
        raw=compact(doc); require(len(raw)<=POLICY['total_bytes'],'forest byte capacity'); self.work.tick()
        return raw,dict(native_seconds=self.native_seconds,native_coordinates=self.native_coordinates)


def build(root, source, *, work, cut=None):
    cut=decode(compact(cut))
    reader=TensorRows(root,source,deadline=work.deadline)
    require(reader.request['experts']<=8,'source forest expert capacity')
    return Builder(reader,work).finish(cut)
