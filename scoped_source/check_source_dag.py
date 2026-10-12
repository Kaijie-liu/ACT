"""Independent stdlib source/affine/ReLU forest checker. No producer or solver.

Checks a captured real-arithmetic declaration's enclosure, not execution floats,
route feasibility, softmax/output properties or the live checkpoint capture.
"""
from fractions import Fraction as F
from itertools import combinations
import math
from types import MappingProxyType

from scoped_source.hz_request_store import POLICY, compact, decode, digest, identity, rational, require, sha
from scoped_source.tensor_rows import TensorRows


def binary(q):
    f=float(q); require(math.isfinite(f) and F(f)==q,'binary64 stored coefficient'); return q


def nearest(q):
    f=float(q); require(math.isfinite(f),'binary64 rounding overflow'); return F(f)


def upper(q):
    f=float(q); require(math.isfinite(f),'outward overflow')
    if F(f)<q: f=math.nextafter(f,math.inf)
    require(math.isfinite(f),'outward overflow'); return F(f)


class Checker:
    def __init__(self, raw, root, source, expected, work, cut):
        self.work=work; work.tick()
        require(type(raw) is bytes and len(raw)<=POLICY['total_bytes'] and digest(raw)==sha(expected),'forest identity/bytes')
        d=decode(raw)
        require(compact(d)==raw and type(d) is dict and set(d)=={
            'schema','source','request','frame','cut','policy','factors','nodes','views','roles','source_receipt'},'forest canonical schema')
        require(d['schema']=='SOURCE_COEFFICIENT_FOREST_V1' and d['source']==sha(source)
            and d['frame']==1 and type(d['frame']) is int and d['cut']==cut and d['policy']==dict(POLICY),'source/frame/mode/policy')
        require(cut is None or type(cut) is list and len(cut)==2 and type(cut[0]) is str and type(cut[1]) is int,'explicit prefix')
        self.reader=TensorRows(root,source,deadline=work.deadline)
        require(self.reader.request['experts']<=8 and d['request']==identity(self.reader.request),'expert/request binding')
        require(type(d['factors']) is list and 0<len(d['factors'])<=POLICY['factors'],'factor inventory')
        self.factors=tuple(tuple(f) for f in d['factors'])
        require(all(len(f)==4 and type(f[0]) is str and len(f[0])<=200 and f[1] in ('continuous','binary')
            and type(f[2]) is int and -1<=f[2]<self.reader.request['experts'] and type(f[3]) is int for f in self.factors), 'factor fields')
        require(len({f[0] for f in self.factors})==len(self.factors),'factor unique names')
        require(type(d['views']) is list and 0<len(d['views'])<=POLICY['views'],'view inventory')
        self.views=[]
        for v in d['views']:
            work.visit('row_visits')
            require(type(v) is dict and set(v)=={'name','scope','parent','op','factors','output','eq','ine'},'view fields')
            require(type(v['name']) is str and type(v['scope']) is int and
                (v['parent'] is None or type(v['parent']) is int and 0<=v['parent']<len(self.views)), 'view parent/name')
            require(type(v['factors']) is list and len(v['factors'])==2 and all(type(j) is int for j in v['factors']), 'factor interval')
            rows={}
            for kind in ('output','eq','ine'):
                require(type(v[kind]) is list and len(v[kind])<=POLICY['view_rows'],'root inventory')
                vals=[]
                for row in v[kind]:
                    work.visit('row_visits'); require(type(row) is list and len(row)==2 and type(row[1]) is int,'root fields')
                    vals.append((binary(rational(row[0])),row[1]))
                rows[kind]=tuple(vals)
            self.views.append({**v,**rows})
        require(len({v['name'] for v in self.views})==len(self.views),'unique view names')
        require(sum(len(v[k]) for v in self.views for k in ('output','eq','ine'))<=POLICY['rows'],'root capacity')
        require(type(d['nodes']) is list and 0<len(d['nodes'])<=POLICY['rows'],'node inventory')
        self.nodes=[]; self.norms=[]; self.local_norms=[]; nnz=0; refs=0; active=set(); depths=[]
        for ni,n in enumerate(d['nodes']):
            work.visit('row_visits')
            require(type(n) is list and len(n)==3 and type(n[0]) is int and 0<=n[0]<len(self.views),'node fields')
            parsed=[]; norm=F(0)
            for z,size in ((1,len(self.factors)),(2,ni)):
                require(type(n[z]) is list,'node entries'); entries=[]; last=-1
                for item in n[z]:
                    work.visit('coefficient_visits')
                    require(type(item) is list and len(item)==2 and type(item[0]) is int and last<item[0]<size,'entry order/range')
                    j,q=item[0],rational(item[1]); require(q!=0,'zero stored entry')
                    if z==1:
                        binary(q); norm=work.exact(norm+abs(q)); active.add(j)
                    entries.append((j,q)); last=j
                parsed.append(tuple(entries))
            nnz+=len(parsed[0]); refs+=len(parsed[1])
            require(nnz<=POLICY['nnz'] and refs<=65536 and len(active)<=POLICY['active'],'local/reference/active capacity')
            depth=1+max((depths[j] for j,_ in parsed[1]),default=0)
            require(depth<=POLICY['depth'],'node depth'); depths.append(depth)
            self.nodes.append((n[0],parsed[0],parsed[1])); self.local_norms.append(norm); self.norms.append(None)
        self.fid=self.vid=self.nid=0; self.receipt=d['source_receipt']; self.roles=d['roles']; self.cut=d['cut']
        self.stats=dict(nodes=len(self.nodes),local_nnz=nnz,references=refs,factors=len(self.factors),
                        views=len(self.views),forest_bytes=len(raw))
        self.visible=[]; self.depths=[]

    def factor(self,name,kind,scope,owner):
        self.work.visit('row_visits')
        require(self.fid<len(self.factors) and self.factors[self.fid]==(name,kind,scope,owner),'factor allocation/ownership')
        j=self.fid; self.fid+=1; return j

    def view(self,name,scope,parent,op):
        self.work.visit('row_visits'); i=self.vid; self.vid+=1
        require(i<len(self.views),'missing operator view'); v=self.views[i]
        require((v['name'],v['scope'],v['parent'],v['op'],v['factors'][0])==(name,scope,parent,op,self.fid),'view source/ancestry/order')
        depth=1+(self.depths[parent] if parent is not None else 0)
        require(depth<=POLICY['depth'],'view depth'); self.depths.append(depth)
        return i,v

    def done(self,i):
        v=self.views[i]; require(v['factors'][1]==self.fid,'extra/missing layer factors')
        parent=v['parent']; inherited=self.visible[parent] if parent is not None else frozenset()
        self.visible.append(inherited|frozenset(range(v['factors'][0],self.fid)))

    def node(self,owner,ni,*,refs=()):
        require(type(ni) is int and ni==self.nid and ni<len(self.nodes),'node order/inventory')
        self.nid+=1; o,local,actual_refs=self.nodes[ni]
        require(o==owner and actual_refs==refs,'coefficient reference/owner')
        self.norms[ni]=self.local_norms[ni]
        for p,s in refs:
            require(self.norms[p] is not None,'unchecked parent')
            self.norms[ni]=self.work.exact(self.norms[ni]+abs(s)*self.norms[p])
        return dict(local)

    def expand(self,ni):
        require(self.norms[ni] is not None,'source uses unchecked row'); d={}; todo=[(ni,F(1))]
        while todo:
            n,w=todo.pop(); self.work.visit('row_visits')
            for j,q in self.nodes[n][1]:
                self.work.visit('coefficient_visits'); d[j]=self.work.exact(d.get(j,F(0))+w*q)
            todo.extend((p,self.work.exact(w*s)) for p,s in self.nodes[n][2])
        return {j:q for j,q in d.items() if q}

    def initial(self):
        i,v=self.view('input',-1,None,'input'); require(not v['eq'] and not v['ine'] and len(v['output'])==len(self.reader.inputs),'input roster')
        for k,((lo,hi),(c,n)) in enumerate(zip(self.reader.inputs,v['output'])):
            f=self.factor(f'input/c/{k}','continuous',-1,i); d=self.node(i,n)
            expected_c=nearest(self.work.exact((lo+hi)/2)); r=upper(max(expected_c-lo,hi-expected_c))
            require(c==expected_c and d==({f:r} if r else {}) and c-r<=lo<=hi<=c+r,'input containment/identity')
        self.done(i); return i

    def alias(self,name,scope,parent,op,outputs):
        i,v=self.view(name,scope,parent,op)
        require(v['output']==outputs and not v['eq'] and not v['ine'],'alias rows/constraints')
        self.done(i); return i

    def affine(self,parent,layer,stream):
        scope=-1 if layer.graph=='router' else int(layer.graph[6:]); name=f'{layer.graph}/{layer.index}'
        i,v=self.view(name,scope,parent,'Linear'); basis=self.fid
        require(len(v['output'])==layer.outputs and not v['eq'] and not v['ine'],'affine output/constraint roster')
        parent_rows=self.views[parent]['output']; diagonal=parent_rows==self.views[0]['output']; pending=[]
        for k,(c,n) in enumerate(v['output']):
            d=self.node(i,n); nominal={j:q for j,q in d.items() if j<basis}
            require(set(nominal)<=self.visible[parent],'private factor visibility')
            row=next(stream); require((row.graph,row.layer,row.row)==(layer.graph,layer.index,k),'affine source order')
            center=row.bias; err=F(0); exact={}
            if diagonal:
                coords=set()
                for j,w in enumerate(row.coefficients):
                    self.work.visit('coefficient_visits'); pc,pn=parent_rows[j]; entries=self.nodes[pn][1]
                    require(not self.nodes[pn][2] and len(entries)<=1,'checked diagonal input')
                    if pc and w: center=self.work.exact(center+w*pc)
                    if entries:
                        f,r=entries[0]; coords.add(f)
                        expected=self.work.exact(w*r)
                        err=self.work.exact(err+abs(expected-nominal.get(f,F(0))))
                require(set(nominal)<=coords,'diagonal extra columns')
            else:
                for j,w in enumerate(row.coefficients):
                    self.work.visit('coefficient_visits')
                    if not w: continue
                    pc,pn=parent_rows[j]; center=self.work.exact(center+w*pc)
                    for f,q in self.expand(pn).items(): exact[f]=self.work.exact(exact.get(f,F(0))+w*q)
                for f in set(exact)|set(nominal): err=self.work.exact(err+abs(exact.get(f,F(0))-nominal.get(f,F(0))))
            err=self.work.exact(err+abs(center-c)); expected={}; delta=F(0)
            tail={j:q for j,q in d.items() if j>=basis}
            require(len(tail)<=2,'affine sparse compensation tail')
            if err:
                f=self.factor(f'{name}/error/{k}','continuous',scope,i); q=nearest(err); delta=abs(err-q)
                if q: expected[f]=q
            pending.append((tail,expected,delta))
        for k,(d,expected,delta) in enumerate(pending):
            if delta: expected[self.factor(f'{name}/lift/output/{k}','continuous',scope,i)]=upper(delta)
            require(d==expected,'affine compensation/extra coefficient')
        self.done(i); return i

    def relu(self,parent,layer):
        name=f'{layer.graph}/{layer.index}'; scope=-1 if layer.graph=='router' else int(layer.graph[6:])
        i,v=self.view(name,scope,parent,'ReLU')
        expected={'output':[],'eq':[],'ine':[]}; negative=[]; positive=[]
        for k,(c,n) in enumerate(self.views[parent]['output']):
            self.work.visit('row_visits'); r=self.norms[n]; require(r is not None,'unchecked ReLU range')
            lo,hi=self.work.exact(c-r),self.work.exact(c+r)
            if lo>=0: expected['output'].append(('alias',c,n))
            elif hi<=0: expected['output'].append(('new',F(0),{},()))
            else:
                fs=[self.factor(f'{name}/{label}/{k}',kind,scope,i) for label,kind in
                    (('negative','continuous'),('positive','continuous'),('sign','binary'))]
                u,p,z=fs; low,high=self.work.exact(lo/2),self.work.exact(hi/2)
                expected['output'].append(('new',high,{p:-high},()))
                expected['eq'].append(('new',self.work.exact(c-high),{u:low,p:-high,z:low},((n,F(-1)),)))
                negative.append(('new',F(0),{u:F(-1),z:F(-1)},()))
                positive.append(('new',F(0),{p:F(-1),z:F(1)},()))
        expected['ine']=negative+positive
        for kind in ('output','eq','ine'):
            require(len(v[kind])==len(expected[kind]),'ReLU all row obligations')
            for k,(actual,item) in enumerate(zip(v[kind],expected[kind])):
                c,n=actual
                if item[0]=='alias': require(actual==item[1:],'active complete parent alias'); continue
                _,rc,rd,refs=item; d=self.node(i,n,refs=refs); nc=nearest(rc); nd={j:nearest(q) for j,q in rd.items()}
                err=self.work.exact(abs(rc-nc)+sum(abs(q-nd[j]) for j,q in rd.items()))
                if err: nd[self.factor(f'{name}/lift/{kind}/{k}','continuous',scope,i)]=upper(err)
                require(c==nc and d=={j:q for j,q in nd.items() if q},'ReLU rational construction/lift/rhs')
                # Every reference is the full checked parent; every local factor is fresh.
                require(all(j>=v['factors'][0] for j in d),'ReLU local parent overlap')
        self.done(i); return i

    def run(self):
        initial=self.initial(); stream=iter(self.reader.rows()); parent=None; previous=None
        common=router=None; ends={}; stopped=False
        for layer in self.reader.plan:
            if stopped: break
            if layer.graph!=previous:
                if previous=='router':
                    router=parent; common=self.alias('common',-1,router,'common',self.views[initial]['output'])
                parent=initial if layer.graph=='router' else common; previous=layer.graph
            if layer.kind=='Flatten':
                scope=-1 if layer.graph=='router' else int(layer.graph[6:])
                parent=self.alias(f'{layer.graph}/{layer.index}',scope,parent,'Flatten',self.views[parent]['output'])
            elif layer.kind=='Linear': parent=self.affine(parent,layer,stream)
            elif layer.kind=='ReLU': parent=self.relu(parent,layer)
            else: raise ValueError('unsupported source operator')
            ends[layer.graph]=parent
            if self.cut==[layer.graph,layer.index]: stopped=True
        require(self.cut is None or stopped,'missing prefix endpoint')
        for _ in stream: self.work.tick()
        require(self.reader.receipt()==self.receipt,'complete independent tensor receipt')
        pairs=[]
        if self.cut is None:
            for a,b in combinations(range(self.reader.request['experts']),2):
                i,v=self.view(f'pair/{a}-{b}',-1,common,'guard'); expected=[]
                for selected in (a,b):
                    for j in range(self.reader.request['experts']):
                        if j in (a,b): continue
                        c,n=self.views[router]['output'][j]; sc,sn=self.views[router]['output'][selected]; d=self.expand(n)
                        for k,q in self.expand(sn).items(): d[k]=self.work.exact(d.get(k,F(0))-q)
                        nd={k:nearest(q) for k,q in d.items()}; err=sum(abs(q-nd[k]) for k,q in d.items())
                        expected.append((upper(self.work.exact(sc-c+err)),{k:q for k,q in nd.items() if q}))
                require(v['output']==self.views[common]['output'] and not v['eq'] and len(v['ine'])==len(expected),'guard roster')
                for (c,n),(rc,rd) in zip(v['ine'],expected): require(c==rc and self.node(i,n)==rd,'guard construction/sign')
                self.done(i); pairs.append([a,b,i])
        roles=dict(input=initial,router=router,common=common,experts=[ends.get(f'expert{i}') if self.cut is None else None
            for i in range(self.reader.request['experts'])],pairs=pairs,prefix=parent if self.cut is not None else None)
        require(self.roles==roles and self.fid==len(self.factors) and self.nid==len(self.nodes) and self.vid==len(self.views),
                'complete factor/node/view/role inventory')
        self.work.tick()
        return dict(status='CHECKED_SOURCE_FOREST' if self.cut is None else 'CHECKED_SOURCE_PREFIX',
            source=self.reader.receipt()['source_sha256'],request=identity(self.reader.request),cut=self.cut,
            stats=self.stats,source_receipt=self.reader.receipt(),work=dict(self.work.counts),
            guarantee='captured real source enclosure; no output property or deployed floating-point proof',full_request_certificates=0)


def check(raw, root, source, *, expected, work, cut=None):
    return Checker(raw,root,source,expected,work,cut).run()
