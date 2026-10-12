"""Opt-in HybridZ support on a checked coefficient-reference forest.

One local sparse bank plus a reference scatter, not an expanded constraint
matrix. CPU candidates only in this version; exact acceptance is independent.
"""
from fractions import Fraction as F
import math
import time

import torch

from scoped_source.hz_request_store import compact, require
from scoped_source.forest_endpoints import (
    CheckedForest, PROOF, contract, objective_roots, evaluate_tile,
)

ALGORITHM='CPU_REFERENCE_PROJECTED_SUBGRADIENT_128_V1'


class Native:
    def __init__(self,forest,work):
        contract(forest)
        require(torch.get_num_threads()==1, 'one CPU thread')
        self.forest=forest; self.work=work; started=time.monotonic()
        rr=[]; cc=[]; vv=[]; children=[]; parents=[]
        for n,(_,local,refs) in enumerate(forest.nodes):
            work.visit('row_visits')
            for j,q in local:
                work.visit('coefficient_visits'); rr.append(n); cc.append(j); vv.append(float(q))
            if refs:
                require(len(refs)==1 and refs[0][1]==-1 and not forest.nodes[refs[0][0]][2], 'one-level reference')
                children.append(n); parents.append(refs[0][0])
        self.shape=(len(forest.nodes),len(forest.factors))
        self.local=torch.sparse_coo_tensor(self.indices([rr,cc]),self.tensor(vv),self.shape,
                                          dtype=torch.float64,device='cpu').coalesce()
        self.transpose=self.local.transpose(0,1)  # index/value view; no second coefficient conversion
        self.children=self.indices(children); self.parents=self.indices(parents)
        self.stats=dict(algorithm=ALGORITHM,device='cpu',dtype='float64',local_nnz=len(vv),
                        references=len(children),coefficient_conversions=len(vv),expanded_constraint_entries=0,
                        preparation_seconds=time.monotonic()-started,proposal_seconds=0.,
                        exact_selection_seconds=0.,tiles=0,columns=0)

    def indices(self,values): return torch.tensor(values,dtype=torch.int64,device='cpu')

    def tensor(self,values):
        r=torch.tensor(values,dtype=torch.float64,device='cpu')
        require(bool(torch.isfinite(r).all()), 'finite CPU candidate input'); self.work.tick(); return r

    def forward(self,x):
        self.work.tick(); v=torch.sparse.mm(self.local,x)
        if len(self.children): v.index_add_(0,self.children,-v.index_select(0,self.parents))
        return v

    def adjoint(self,alpha):
        self.work.tick(); a=alpha.clone()
        if len(self.children): a.index_add_(0,self.parents,-alpha.index_select(0,self.children))
        return torch.sparse.mm(self.transpose,a)

    def propose(self,tasks):
        started=time.monotonic(); k=len(tasks)
        require(0<k<=8 and len({t.pair for t in tasks})==1, 'bounded one-pair tile')
        a=torch.zeros((self.shape[0],k),dtype=torch.float64,device='cpu'); dd=[]
        for i,t in enumerate(tasks):
            d,weights=objective_roots(self.forest,t,self.work); dd.append(float(d))
            for n,w in weights.items(): a[n,i]=float(w)
        d=self.tensor(dd); groups=[]
        for kind in ('ine','eq'):
            rows=self.forest.rows(tasks[0].pair,kind); vals=[self.forest.row(r) for r in rows]
            ix=self.indices([n for _,n in vals]); rhs=self.tensor([float(c) for c,_ in vals])[:,None]
            z=torch.zeros((len(rows),k),dtype=torch.float64,device='cpu')
            groups.append(dict(kind=kind,ix=ix,rhs=rhs,z=z,best=z.clone()))
        best=torch.full((k,),-math.inf,dtype=torch.float64,device='cpu')
        with torch.no_grad():
            for step in range(129):
                self.work.visit('row_visits'); alpha=a.clone(); value=d.clone()
                for g in groups:
                    require(bool(torch.isfinite(g['z']).all()), 'finite dual iterate')
                    alpha.index_add_(0,g['ix'],-g['z']); value+=(g['rhs']*g['z']).sum(0)
                r=self.adjoint(alpha); value-=r.abs().sum(0)
                require(bool(torch.isfinite(r).all() and torch.isfinite(value).all()), 'finite candidate iteration')
                good=value>best; best=torch.where(good,value,best)
                for g in groups: g['best'][:,good]=g['z'][:,good]
                if step==128: break
                v=self.forward(-torch.sign(r)); rate=.125/math.sqrt(step+1)
                for g in groups:
                    z=g['z']+rate*(g['rhs']-v.index_select(0,g['ix']))
                    g['z']=torch.minimum(z,torch.zeros_like(z,device='cpu')) if g['kind']=='ine' else z
        proposed=[]; zeros=[]
        for col,t in enumerate(tasks):
            c=dict(query=t.query_id,lower='0')
            for g in groups:
                c['inequality' if g['kind']=='ine' else 'equality']=[str(F(v)) for v in g['best'][:,col].tolist()]
            proposed.append(c)
            zeros.append(dict(query=t.query_id,lower='0',equality=['0']*len(c['equality']),inequality=['0']*len(c['inequality'])))
        self.stats['proposal_seconds']+=time.monotonic()-started; started=time.monotonic()
        bounds,_=evaluate_tile(self.forest,tasks,proposed,self.work)
        fallback,_=evaluate_tile(self.forest,tasks,zeros,self.work)
        records=[]
        for t,p,z,b,f in zip(tasks,proposed,zeros,bounds,fallback):
            c=z if f>b else p; c['lower']=str(max(b,f)); records.append(dict(slot=t.slot,candidate=c))
        self.stats['exact_selection_seconds']+=time.monotonic()-started
        self.stats['tiles']+=1; self.stats['columns']+=k; self.work.tick(); return records


def produce(forest,inventory,*,width,work,missing_slots=(),event=None):
    require(inventory.forest==forest.identity, 'plan source binding')
    missing=frozenset(missing_slots); require(missing<={t.slot for t in inventory.tasks}, 'partial slot binding')
    native=Native(forest,work); tiles=[]; total=0; encoding=0.
    for tasks in inventory.tiles(width):
        records=native.propose(tasks)
        for r in records:
            if r['slot'] in missing: r['candidate']=None
        begin=time.monotonic(); raw=compact(records); total+=len(raw)
        require(len(raw)<=4*2**20 and total<=16*2**20-8192, 'bounded producer proof bytes')
        encoding+=time.monotonic()-begin; work.tick(); tiles.append(records)
        if event: event(len(tiles)-1,records)
        work.tick()
    proof=dict(schema=PROOF,contract=inventory.contract,width=width,tiles=tiles)
    begin=time.monotonic(); size=len(compact(proof)); encoding+=time.monotonic()-begin
    require(size<=16*2**20, 'bounded final proof bytes'); work.tick()
    return proof,dict(native.stats,capacity_encoding_seconds=encoding,proof_bytes=size)
