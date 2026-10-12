"""CPU candidate operator for checked general affine-reference forests.

No optimizer or accepting terminal. C=(I-R)^-1 L, without expanding C. CPU
float64 is approximate; controls compare to exact expansion of the same V2.
"""
import torch

from scoped_source.check_affine_references import CheckedAffineForest
from scoped_source.hz_request_store import require


class Operator:
    def __init__(self,forest,work):
        require(type(forest) is CheckedAffineForest and getattr(forest,'_sealed',False) is True
                and forest.complete,'complete V2 checked source required')
        require(torch.get_num_threads()==1,'one CPU thread')
        self.work=work; self.shape=(len(forest.nodes),len(forest.factors)); rr=[]; cc=[]; vv=[]
        depths=[]; edges={}
        for n,(_,local,refs) in enumerate(forest.nodes):
            work.visit('row_visits')
            depth=1+max((depths[p] for p,_ in refs),default=0); depths.append(depth)
            for j,q in local:
                work.visit('coefficient_visits'); rr.append(n); cc.append(j); vv.append(float(q))
            for p,s in refs:
                work.visit('coefficient_visits'); edges.setdefault(depth,[]).append((n,p,float(s)))
        ix=lambda x:torch.tensor(x,dtype=torch.int64,device='cpu')
        val=lambda x:torch.tensor(x,dtype=torch.float64,device='cpu')
        self.local=torch.sparse_coo_tensor(ix([rr,cc]),val(vv),self.shape,device='cpu').coalesce()
        self.transpose=self.local.transpose(0,1)  # May allocate another native value bank.
        self.levels=tuple((ix([n for n,p,s in edges[d]]),ix([p for n,p,s in edges[d]]),
                           val([s for n,p,s in edges[d]])[:,None]) for d in sorted(edges))
        require(bool(torch.isfinite(self.local.values()).all()) and
                all(bool(torch.isfinite(s).all()) for _,_,s in self.levels),'finite candidate coefficients')
        self.stats=dict(nodes=self.shape[0],factors=self.shape[1],local_nnz=len(vv),
                        references=sum(len(v) for v in edges.values()),levels=len(self.levels),
                        expanded_entries=0,device='cpu',dtype='float64')
        work.tick()

    def validate(self,v,rows):
        self.work.tick()
        require(type(v) is torch.Tensor and v.device.type=='cpu' and v.dtype==torch.float64
                and v.ndim==2 and v.shape[0]==rows and 0<v.shape[1]<=8,'bounded CPU float64 tile')
        require(bool(torch.isfinite(v).all()),'finite candidate tile')

    def forward(self,x):
        self.validate(x,self.shape[1]); values=torch.sparse.mm(self.local,x)
        for children,parents,scale in self.levels:
            self.work.visit('row_visits'); values.index_add_(0,children,values.index_select(0,parents)*scale)
        self.work.tick(); require(bool(torch.isfinite(values).all()),'finite forward result'); return values

    def adjoint(self,alpha):
        self.validate(alpha,self.shape[0]); values=alpha.clone()
        for children,parents,scale in reversed(self.levels):
            self.work.visit('row_visits'); values.index_add_(0,parents,values.index_select(0,children)*scale)
        result=torch.sparse.mm(self.transpose,values)
        self.work.tick(); require(bool(torch.isfinite(result).all()),'finite adjoint result'); return result
