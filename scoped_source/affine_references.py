"""Opt-in V2 source lowering: later affine coefficients remain references.

This changes the enclosure relative to V1, not the captured network. First
diagonal affine compensation is unchanged. Acceptance is in the separate
check_affine_references module; no complete endpoint execution is enabled here.
"""
from fractions import Fraction as F

from scoped_source.hz_request_store import POLICY, compact, decode, require
from scoped_source.hz_store_source import nearest, upward
from scoped_source.source_dag import Builder as Base
from scoped_source.tensor_rows import TensorRows

SCHEMA='SOURCE_AFFINE_REFERENCE_FOREST_V2'


class Builder(Base):
    schema=SCHEMA

    def affine(self,parent,layer,stream):
        rows=self.views[parent]['output']
        if rows==self.views[0]['output']: return super().affine(parent,layer,stream)
        owner=len(self.views); first=len(self.factors); name=f'{layer.graph}/{layer.index}'
        scope=-1 if layer.graph=='router' else int(layer.graph[6:]); pending=[]
        for k in range(layer.outputs):
            source=next(stream)
            require((source.graph,source.layer,source.row)==(layer.graph,layer.index,k),'source row order')
            center=source.bias; refs={}
            for (pc,pn),weight in zip(rows,source.coefficients,strict=True):
                self.work.visit('coefficient_visits')
                if not weight: continue
                # Centers are independent of node identity and cannot be merged away.
                center=self.work.exact(center+weight*pc)
                refs[pn]=self.work.exact(refs.get(pn,F(0))+weight)
            c=nearest(center); err=self.work.exact(abs(center-c)); local={}; delta=F(0)
            if err:
                fid=self.factor(f'{name}/error/{k}','continuous',scope,owner)
                q=nearest(err); delta=abs(err-q)
                if q: local[fid]=q
            node=self.node(owner,local,tuple(sorted((n,q) for n,q in refs.items() if q)))
            pending.append((c,node,delta))
        output=[]
        for k,(c,n,delta) in enumerate(pending):
            if delta:
                fid=self.factor(f'{name}/lift/output/{k}','continuous',scope,owner); q=upward(delta)
                self.nodes[n][1]+=((fid,q),); self.norms[n]=self.work.exact(self.norms[n]+q)
                self.local_nnz+=1; require(self.local_nnz<=POLICY['nnz'],'forest local/reference capacity')
            output.append((c,n))
        return self.view(name,scope,parent,'Linear',first,output)

    def guard(self,owner,other,selected):
        rc,rn=other; sc,sn=selected
        refs={} if rn==sn else {rn:F(1),sn:F(-1)}
        return upward(self.work.exact(sc-rc)),self.node(owner,{},tuple(sorted(refs.items())))


def build(root,source,*,work,cut=None):
    cut=decode(compact(cut)); reader=TensorRows(root,source,deadline=work.deadline)
    require(reader.request['experts']<=8,'source forest expert capacity')
    return Builder(reader,work).finish(cut)
