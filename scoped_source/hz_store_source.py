"""Untrusted finite source producer; acceptance is in check_hz_store_source.

Uses actual sparse_hz_linear for the nominal, then compensates exact rational
residuals. Only final rows are stored; ancestors are immutable references.
"""
from fractions import Fraction as F
from itertools import combinations
import math
from types import SimpleNamespace
import time

from scoped_source.hz_request_store import Row, Rows, identity, require, compact, decode, rational
from scoped_source.hz_store_write import Writer
from scoped_source.hz_store_sparse import instantiate, kind_maps


def nearest(q):
    f=float(q); require(math.isfinite(f), 'producer binary64 overflow'); return F(f)


def upward(q):
    f=float(q); require(math.isfinite(f), 'producer binary64 overflow')
    if F(f) < q: f=math.nextafter(f,math.inf)
    require(math.isfinite(f), 'producer outward overflow'); return F(f)


def plus(rows, weights, bias, work):
    c=work.exact(F(bias)); coeff={}
    for j,w in weights:
        w=work.exact(F(w)); c=work.exact(c+work.exact(w*rows[j].rhs))
        for k,q in rows[j].coefficients:
            coeff[k]=work.exact(coeff.get(k,F(0))+work.exact(w*q))
    return c,{k:q for k,q in coeff.items() if q}


def raw(row):
    return [row.name,str(row.rhs),[[j,str(q)] for j,q in row.coefficients]]


def admit(doc, work):
    require(set(doc)=={'schema','lower','upper','router','experts'}
            and doc['schema']=='REQUEST_STORE_SOURCE_V1', 'producer declaration fields')
    require(type(doc['lower']) is list and type(doc['upper']) is list
            and 1<=len(doc['lower'])==len(doc['upper'])<=4096, 'producer input capacity')
    for l,u in zip(doc['lower'],doc['upper']):
        work.tick(); require(rational(l)<=rational(u), 'producer input bounds')
    require(type(doc['experts']) is list and 2<=len(doc['experts'])<=8, 'producer expert capacity')
    for layers in [doc['router'],*doc['experts']]:
        width=len(doc['lower'])
        require(type(layers) is list and 1<=len(layers)<=16, 'producer layer capacity')
        for layer in layers:
            work.tick()
            if layer=={'kind':'relu'}: continue
            require(type(layer) is dict and set(layer)=={'kind','rows'} and layer['kind']=='affine'
                    and type(layer['rows']) is list and 1<=len(layer['rows'])<=4096, 'producer operator capacity')
            nnz=0
            for row in layer['rows']:
                require(type(row) is list and len(row)==2 and type(row[1]) is list, 'producer operator fields')
                q=rational(row[0]); require(nearest(q)==q, 'producer nonbinary parameter')
                last=-1
                for item in row[1]:
                    work.visit('coefficient_visits')
                    require(type(item) is list and len(item)==2, 'producer parameter fields')
                    j,value=item; require(type(j) is int and last<j<width, 'producer parameter coordinate')
                    q=rational(value); require(q!=0 and nearest(q)==q, 'producer nonbinary parameter')
                    last=j; nnz+=1; require(nnz<=262144, 'producer parameter nnz capacity')
            width=len(layer['rows'])


class Builder:
    def __init__(self, root, doc, work, page_rows):
        self.work=work; self.factors=[]; self.views={}; self.refs={}; self.native_seconds=0.
        self.serialization_seconds=0.
        self.writer=Writer(root,identity(doc),len(doc['experts']),self.factors,
                           work=work,page_rows=page_rows)

    def factor(self, name, label, scope, kind='continuous'):
        self.work.tick(); j=len(self.factors)
        self.factors.append([f'{name}/{label}',kind,scope,name]); return j

    def publish(self, name, scope, parent, fresh, outputs, eq=(), ine=(), guard=None, output_refs=None):
        def seq(kind, rows, inherited):
            values=tuple(Row(f'{name}/{kind}/{i}',F(c),tuple(sorted((j,F(q)) for j,q in d.items() if q)))
                         for i,(c,d) in enumerate(rows))
            started=time.monotonic(); refs=self.writer.rows(kind,map(raw,values))
            self.serialization_seconds+=time.monotonic()-started
            return Rows(inherited,(values,) if values else (),
                        (inherited.count if inherited else 0)+len(values)), refs
        if output_refs is None:
            out,orefs=seq('output',outputs,None)
        else:
            out=self.views[output_refs].output; orefs=self.refs[output_refs]
        equal,erefs=seq('eq',eq,parent.eq if parent else None)
        upper,irefs=seq('ine',ine,parent.ine if parent else None)
        state=SimpleNamespace(name=name,scope=scope,frame=1,guard=guard,parent=parent,
            factors=(parent.factors if parent else frozenset())|frozenset(fresh),
            output=out,eq=equal,ine=upper)
        started=time.monotonic()
        self.writer.view(name,scope=scope,guard=guard,parent=parent.name if parent else None,
            add_factors=fresh,output=orefs,eq=erefs,ine=irefs)
        self.serialization_seconds+=time.monotonic()-started
        self.views[name]=state; self.refs[name]=orefs; return state

    def lift(self, name, scope, output, eq, ine):
        """Round only new rows; all output lifts precede all equality lifts."""
        result=[]
        for kind,rows in (('output',output),('eq',eq),('ine',ine)):
            rounded=[]
            for i,(c,d) in enumerate(rows):
                self.work.tick(); nc=nearest(c); nd={j:nearest(q) for j,q in d.items()}
                delta=F(0)
                for j,q in d.items(): delta=self.work.exact(delta+abs(self.work.exact(q-nd[j])))
                if kind=='ine': nc=upward(self.work.exact(c+delta))
                else:
                    delta=self.work.exact(delta+abs(self.work.exact(c-nc)))
                    if delta: nd[self.factor(name,f'lift/{kind}/{i}',scope)]=upward(delta)
                rounded.append((nc,{j:q for j,q in nd.items() if q}))
            result.append(rounded)
        return result

    def input(self, lo, hi):
        outputs=[]
        for i,(l,u) in enumerate(zip(lo,hi)):
            self.work.tick(); l,u=self.work.exact(F(l)),self.work.exact(F(u))
            c=nearest(self.work.exact(self.work.exact(l+u)/2))
            radius=upward(max(self.work.exact(c-l),self.work.exact(u-c)))
            j=self.factor('input',f'c/{i}','common')
            outputs.append((c,{j:radius} if radius else {}))
        return self.publish('input','common',None,range(len(outputs)),outputs)

    def affine(self, parent, layer, name, scope):
        import numpy as np
        import scipy.sparse as sp
        from act.back_end.solver.solver_hz import sparse_hz_linear
        begin=time.monotonic(); live,cids,bids=instantiate(self.factors,parent,self.work)
        data=[]; cc=[]; ptr=[0]; bias=[]
        for c,entries in layer['rows']:
            self.work.tick(); bias.append(float(F(c)))
            for j,q in entries: cc.append(j); data.append(float(F(q)))
            ptr.append(len(data))
        matrix=sp.csr_matrix((data,cc,ptr),shape=(len(bias),parent.output.count))
        nominal=sparse_hz_linear(live,matrix,np.asarray(bias,dtype=np.float64))
        self.native_seconds+=time.monotonic()-begin
        reference=[]; source=tuple(parent.output); start=len(self.factors)
        for i,(c,entries) in enumerate(layer['rows']):
            self.work.tick(); exact_c,exact_d=plus(source,entries,c,self.work); nc=F(float(nominal.c[i])); nd={}
            for m,ids in ((nominal.Gc,cids),(nominal.Gb,bids)):
                for k in range(m.indptr[i],m.indptr[i+1]):
                    q=F(float(m.data[k]))
                    if q: nd[ids[m.indices[k]]]=q
            needed=abs(self.work.exact(exact_c-nc))
            for j in set(exact_d)|set(nd):
                needed=self.work.exact(needed+abs(self.work.exact(exact_d.get(j,F(0))-nd.get(j,F(0)))))
            if needed: nd[self.factor(name,f'error/{i}',scope)]=needed
            reference.append((nc,nd))
        output,eq,ine=self.lift(name,scope,reference,[],[])
        del nominal,live,matrix
        return self.publish(name,scope,parent,range(start,len(self.factors)),output,eq,ine)

    def relu(self, parent, name, scope):
        start=len(self.factors); outputs=[]; eq=[]; negative=[]; positive=[]
        for i,row in enumerate(parent.output):
            self.work.tick(); d=dict(row.coefficients); radius=F(0)
            for q in d.values(): radius=self.work.exact(radius+abs(q))
            lo,hi=self.work.exact(row.rhs-radius),self.work.exact(row.rhs+radius)
            if lo>=0: outputs.append((row.rhs,d))
            elif hi<=0: outputs.append((F(0),{}))
            else:
                u=self.factor(name,f'negative/{i}',scope); v=self.factor(name,f'positive/{i}',scope)
                z=self.factor(name,f'sign/{i}',scope,'binary')
                h2,l2=self.work.exact(hi/2),self.work.exact(lo/2)
                outputs.append((h2,{v:-h2}))
                ed={j:-q for j,q in d.items()}; ed.update({u:l2,v:-h2,z:l2})
                eq.append((self.work.exact(row.rhs-h2),ed))
                negative.append((F(0),{u:F(-1),z:F(-1)}))
                positive.append((F(0),{v:F(-1),z:F(1)}))
        output,equal,upper=self.lift(name,scope,outputs,eq,negative+positive)
        return self.publish(name,scope,parent,range(start,len(self.factors)),output,equal,upper)

    def network(self, layers, parent, prefix, scope):
        for i,layer in enumerate(layers):
            name=f'{prefix}/{i}'
            parent=self.affine(parent,layer,name,scope) if layer['kind']=='affine' else self.relu(parent,name,scope)
        return parent


def build(root, doc, *, work, page_rows=64):
    start=time.monotonic(); payload=compact(doc)
    require(len(payload) <= 4*1024*1024, 'producer declaration capacity')
    doc=decode(payload); work.tick(); admit(doc,work); b=Builder(root,doc,work,page_rows)
    initial=b.input(doc['lower'],doc['upper'])
    router=b.network(doc['router'],initial,'router','common')
    base=b.publish('common','common',router,[],[],output_refs='input')
    experts=[b.network(layers,base,f'expert{i}',f'expert/{i}') for i,layers in enumerate(doc['experts'])]
    entries=[]; router_rows=tuple(router.output)
    for pair in combinations(range(len(experts)),2):
        name=f'entry{pair[0]}-{pair[1]}'; rows=[]
        for a in pair:
            for outside in range(len(experts)):
                if outside not in pair:
                    c,d=plus(router_rows,[(outside,F(1)),(a,F(-1))],F(0),work); rows.append((-c,d))
        _,_,rounded=b.lift(name,'common',[],[],rows)
        state=b.publish(name,'common',base,[],[],ine=rounded,guard=list(pair),output_refs='common')
        entries.append(dict(pair=list(pair),view=state.name))
    writing=time.monotonic()
    h=b.writer.finish(dict(common='common',templates=[v.name for v in experts],entries=entries))
    b.serialization_seconds+=time.monotonic()-writing
    work.tick()
    return h,dict(generation_seconds=time.monotonic()-start,
                  native_workspace_and_affine_seconds=b.native_seconds,
                  source_member_serialization_seconds=b.serialization_seconds)
