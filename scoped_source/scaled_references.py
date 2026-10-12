"""Opt-in V3 captured source: C=(I-R)^-1 L D, not rounded W*radius.

The input scales stay on distinct factors. Auxiliary error/ReLU factors have
unit scale. The separate checker reconstructs the enclosure from raw tensors.
"""
from fractions import Fraction as F

from scoped_source.affine_references import Builder as Base
from scoped_source.hz_request_store import POLICY, compact, decode, require
from scoped_source.hz_store_source import nearest, upward
from scoped_source.tensor_rows import TensorRows

SCHEMA='SOURCE_SCALED_REFERENCE_FOREST_V3'


class Builder(Base):
    schema=SCHEMA

    def __init__(self,reader,work):
        super().__init__(reader,work); self.scales=[]

    def factor(self,name,kind,scope,owner):
        j=super().factor(name,kind,scope,owner); self.scales.append(F(1)); return j

    def local_norm(self,local):
        groups={}
        for j,q in local:
            d=self.scales[j]; groups[d]=self.work.exact(groups.get(d,F(0))+abs(q))
        value=F(0)
        for d,v in groups.items(): value=self.work.exact(value+self.work.exact(d*v))
        return value

    def document_extra(self): return dict(scales=[str(q) for q in self.scales])

    def input(self):
        out=[]
        for i,(lo,hi) in enumerate(self.reader.inputs):
            c=nearest(self.work.exact((lo+hi)/2)); r=upward(max(c-lo,hi-c))
            j=self.factor(f'input/c/{i}','continuous',-1,0); self.scales[j]=r
            out.append((c,self.node(0,{j:F(1)})))
        return self.view('input',-1,None,'input',0,out)

    def affine(self,parent,layer,stream):
        rows=self.views[parent]['output']
        if rows!=self.views[0]['output']: return super().affine(parent,layer,stream)
        owner=len(self.views); first=len(self.factors); name=f'{layer.graph}/{layer.index}'
        scope=-1 if layer.graph=='router' else int(layer.graph[6:]); pending=[]
        for k in range(layer.outputs):
            row=next(stream)
            require((row.graph,row.layer,row.row)==(layer.graph,layer.index,k),'source row order')
            center=row.bias; local={}
            for (pc,pn),weight in zip(rows,row.coefficients,strict=True):
                self.work.visit('coefficient_visits')
                entries=self.nodes[pn][1]
                require(not self.nodes[pn][2] and len(entries)==1 and entries[0][1]==1,'unit input basis')
                if pc and weight: center=self.work.exact(center+weight*pc)
                if weight: local[entries[0][0]]=weight
            c=nearest(center); err=self.work.exact(abs(center-c)); delta=F(0)
            if err:
                f=self.factor(f'{name}/error/{k}','continuous',scope,owner)
                rounded=nearest(err); delta=abs(err-rounded)
                if rounded: local[f]=rounded
            pending.append((c,self.node(owner,local),delta))
        out=[]
        for k,(c,n,delta) in enumerate(pending):
            if delta:
                f=self.factor(f'{name}/lift/output/{k}','continuous',scope,owner); q=upward(delta)
                self.nodes[n][1]+=((f,q),); self.norms[n]=self.work.exact(self.norms[n]+q)
                self.local_nnz+=1; require(self.local_nnz<=POLICY['nnz'],'forest local/reference capacity')
            out.append((c,n))
        return self.view(name,scope,parent,'Linear',first,out)


def build(root,source,*,work,cut=None):
    cut=decode(compact(cut)); reader=TensorRows(root,source,deadline=work.deadline)
    require(reader.request['experts']<=8,'source forest expert capacity')
    return Builder(reader,work).finish(cut)
