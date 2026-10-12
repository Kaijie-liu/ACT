"""Optional HybridZ scaled-reference multi-objective CPU candidate path.

Local coefficient bank and DAG references stay factored. Numerical candidates
never authorize acceptance; the stdlib session checks both proposals exactly.
No CUDA execution or default-path replacement is enabled by this component.
"""
from fractions import Fraction as F
import math
import time

import torch

from scoped_source.check_scaled_references import checked
from scoped_source.hz_request_store import require,identity,compact
from scoped_source.scaled_endpoints import Session

ALGORITHM='CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3'


class Operator:
    def __init__(self,forest,work):
        checked(forest,work); require(torch.get_num_threads()==1,'one CPU thread')
        self.work=work; self.shape=(len(forest.nodes),len(forest.factors)); self.forest=forest
        rr=[]; cc=[]; vv=[]; depths=[]; edges={}
        for n,(_,local,refs) in enumerate(forest.nodes):
            work.visit('row_visits'); depth=1+max((depths[p] for p,_ in refs),default=0); depths.append(depth)
            for j,q in local:
                work.visit('coefficient_visits'); rr.append(n); cc.append(j); vv.append(float(q))
            for p,s in refs:
                work.visit('coefficient_visits'); edges.setdefault(depth,[]).append((n,p,float(s)))
        scales=[]
        for q in forest.scales: work.visit('coefficient_visits'); scales.append(float(q))
        self.scales=self.val(scales)[:,None]
        self.local=torch.sparse_coo_tensor(self.ix([rr,cc]),self.val(vv),self.shape,device='cpu').coalesce()
        self.transpose=self.local.transpose(0,1)
        self.levels=tuple((self.ix([n for n,p,s in edges[d]]),self.ix([p for n,p,s in edges[d]]),
                           self.val([s for n,p,s in edges[d]])[:,None]) for d in sorted(edges))
        self.stats=dict(algorithm=ALGORITHM,device='cpu',dtype='float64',nodes=self.shape[0],
                        factors=self.shape[1],local_nnz=len(vv),references=sum(map(len,edges.values())),
                        levels=len(self.levels),expanded_constraint_entries=0)
        work.tick()

    def ix(self,x): return torch.tensor(x,dtype=torch.int64,device='cpu')

    def val(self,x):
        result=torch.tensor(x,dtype=torch.float64,device='cpu')
        require(bool(torch.isfinite(result).all()),'finite native input'); self.work.tick(); return result

    def validate(self,x,rows):
        self.work.tick(); require(type(x) is torch.Tensor and x.device.type=='cpu' and x.dtype==torch.float64
            and x.ndim==2 and x.shape[0]==rows and 0<x.shape[1]<=8,'bounded CPU float64 tile')
        require(bool(torch.isfinite(x).all()),'finite native tile')

    def forward(self,x):
        self.validate(x,self.shape[1]); result=torch.sparse.mm(self.local,self.scales*x)
        for children,parents,scale in self.levels:
            self.work.visit('row_visits'); result.index_add_(0,children,result.index_select(0,parents)*scale)
        self.work.tick(); require(bool(torch.isfinite(result).all()),'finite native forward'); return result

    def adjoint(self,alpha):
        self.validate(alpha,self.shape[0]); values=alpha.clone()
        for children,parents,scale in reversed(self.levels):
            self.work.visit('row_visits'); values.index_add_(0,parents,values.index_select(0,children)*scale)
        result=self.scales*torch.sparse.mm(self.transpose,values)
        self.work.tick(); require(bool(torch.isfinite(result).all()),'finite native adjoint'); return result

    def propose(self,tasks,roots):
        k=len(tasks); require(0<k<=8 and len(roots)==k and len({t.pair for t in tasks})==1,'one bounded pair tile')
        alpha=torch.zeros((self.shape[0],k),dtype=torch.float64,device='cpu'); offsets=[]
        for col,(d,weights) in enumerate(roots):
            self.work.visit('row_visits'); offsets.append(float(d))
            for n,w in weights.items(): self.work.visit('coefficient_visits'); alpha[n,col]=float(w)
        d=self.val(offsets); groups=[]
        for kind in ('ine','eq'):
            ix=[]; rhs=[]
            for row in self.forest.rows(tasks[0].pair,kind):
                self.work.visit('row_visits'); c,n=self.forest.row(row); ix.append(n)
                self.work.visit('coefficient_visits'); rhs.append(float(c))
            z=torch.zeros((len(rhs),k),dtype=torch.float64,device='cpu')
            groups.append(dict(kind=kind,ix=self.ix(ix),rhs=self.val(rhs)[:,None],z=z,best=z.clone()))
        best=torch.full((k,),-math.inf,dtype=torch.float64,device='cpu')
        with torch.no_grad():
            for step in range(129):
                self.work.visit('row_visits'); a=alpha.clone(); value=d.clone()
                for g in groups:
                    require(bool(torch.isfinite(g['z']).all()),'finite dual iterate')
                    a.index_add_(0,g['ix'],-g['z']); value+=(g['rhs']*g['z']).sum(0)
                residual=self.adjoint(a); value-=residual.abs().sum(0)
                require(bool(torch.isfinite(value).all()),'finite candidate value')
                good=value>best; best=torch.where(good,value,best)
                for g in groups: g['best'][:,good]=g['z'][:,good]
                if step==128: break
                points=self.forward(-torch.sign(residual)); rate=.125/math.sqrt(step+1)
                for g in groups:
                    z=g['z']+rate*(g['rhs']-points.index_select(0,g['ix']))
                    g['z']=torch.minimum(z,torch.zeros_like(z,device='cpu')) if g['kind']=='ine' else z
        records=[]
        for col,task in enumerate(tasks):
            self.work.visit('row_visits'); candidate=dict(query=task.query_id,lower='0')
            for g in groups:
                values=[]
                for v in g['best'][:,col].tolist():
                    self.work.visit('coefficient_visits'); values.append(str(F(v)))
                candidate['inequality' if g['kind']=='ine' else 'equality']=values
            self.work.visit('row_visits')
            zero=dict(query=task.query_id,lower='0',equality=['0']*len(candidate['equality']),
                      inequality=['0']*len(candidate['inequality']))
            records.append(dict(slot=task.slot,proposed=candidate,zero=zero))
        self.work.tick(); return records


def produce(forest,doc,*,width=8,work,event=None,missing_slots=()):
    started=time.monotonic(); session=Session(forest,doc,expected=identity(doc),work=work,width=width)
    planning=time.monotonic()-started; start=time.monotonic(); native=Operator(forest,work)
    preparation=time.monotonic()-start; receipts=[]; proposal=exact=objective=encoding=0.
    missing=frozenset(missing_slots); require(missing<={t.slot for t in session.tasks},'partial slot binding')
    for i in range(session.tile_count):
        start=time.monotonic(); tasks,roots=session.prepare(i); objective+=time.monotonic()-start
        start=time.monotonic(); records=native.propose(tasks,roots); proposal+=time.monotonic()-start
        for record in records:
            if record['slot'] in missing: record['proposed']=record['zero']=None
        start=time.monotonic()
        # Preserve RAW proposals before an exact-check timeout or exception.
        # Publication is not an accepting receipt, and is fully budgeted.
        raw=compact(records); require(len(raw)<=4*2**20,'raw candidate publication cap')
        if event: event(i,raw)  # Immutable bytes also consumed by the checker.
        encoding+=time.monotonic()-start; work.tick()
        start=time.monotonic(); receipts.extend(session.accept(raw)); exact+=time.monotonic()-start
    start=time.monotonic(); result=session.finish(receipts); proof=session.proof(); terminal=time.monotonic()-start
    return proof,result,dict(native.stats,planning_seconds=planning,preparation_seconds=preparation,
        objective_seconds=objective,proposal_seconds=proposal,exact_check_seconds=exact,
        partial_publish_seconds=encoding,terminal_encode_seconds=terminal,tiles=session.tile_count,
        endpoints=len(session.tasks),candidates=2*len(session.tasks))
