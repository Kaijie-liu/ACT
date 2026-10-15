"""Opt-in shared first-affine construction; old V3 enclosure equations."""
from fractions import Fraction as F

from scoped_source.scaled_references import Builder as Base
from scoped_source.hz_request_store import POLICY, require
from scoped_source.hz_store_source import nearest, upward
from scoped_source.shared_source import Snapshot, SourceLocal
from scoped_source.shared_format import encode


class Builder(Base):
    def affine(self, parent, layer, stream):
        rows=self.views[parent]['output']
        if rows!=self.views[0]['output']: return super().affine(parent,layer,stream)
        owner=len(self.views); first=len(self.factors); name=f'{layer.graph}/{layer.index}'
        scope=-1 if layer.graph=='router' else int(layer.graph[6:]); pending=[]
        for k in range(layer.outputs):
            source=next(stream); key=(layer.graph,layer.index,k)
            require((source.graph,source.layer,source.row)==key, 'shared source order')
            packed=self.reader.row(key); center=source.bias
            for j,((pc,pn),weight) in enumerate(zip(rows,source.coefficients,strict=True)):
                self.work.visit('coefficient_visits')
                require(self.nodes[pn][1]==((j,F(1)),) and not self.nodes[pn][2], 'contiguous checked input basis')
                if pc and weight: center=self.work.exact(center+weight*pc)
            c=nearest(center); err=self.work.exact(abs(center-c)); delta=F(0); tail=[]
            if err:
                f=self.factor(f'{name}/error/{k}','continuous',scope,owner)
                q=nearest(err); delta=abs(err-q)
                if q: tail.append((f,q))
            local=SourceLocal(packed,tuple(tail),self.work)
            self.work.visit('row_visits'); self.local_nnz+=len(local)
            require(len(self.nodes)<POLICY['rows'] and self.local_nnz<=POLICY['nnz'], 'unchanged source node/local cap')
            n=len(self.nodes); self.nodes.append([owner,local,()]); self.norms.append(self.local_norm(local))
            pending.append((c,n,delta))
        out=[]
        for k,(c,n,delta) in enumerate(pending):
            if delta:
                f=self.factor(f'{name}/lift/output/{k}','continuous',scope,owner); q=upward(delta)
                self.nodes[n][1]+=((f,q),); self.norms[n]=self.work.exact(self.norms[n]+q)
                self.local_nnz+=1; require(self.local_nnz<=POLICY['nnz'], 'source lift capacity')
            out.append((c,n))
        return self.view(name,scope,parent,'Linear',first,out)

    def serialize(self, roles, cut):
        packet=encode(self,roles,cut)
        return packet,dict(native_seconds=0.,native_coordinates=0,packed_source_bytes=self.reader.packed_bytes,
                           forest_bytes=packet.size,pages=len(packet.pages))


def build(root, source, *, work, enabled=False):
    require(enabled is True, 'shared source is opt-in')
    return Builder(Snapshot(root,source,work),work).finish(None)
