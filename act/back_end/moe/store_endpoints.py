"""Device-selectable tiled support on the original request-owned HybridZ rows.

Common/private sparse matrices persist once per request. Pair guards are local;
only objective columns and dual iterates are tiled. No joint HZ or LP export.
"""
from fractions import Fraction as F
import math
from pathlib import Path
import time

import torch

from scoped_source.hz_request_store import compact, digest, require
from scoped_source.check_store_endpoints import PROOF, ALGORITHM, objectives, evaluate_tile
from scoped_proof.io import save
from act.back_end.moe.store_device import Runtime


def write(path, value):
    return save(path,value)


class Native:
    def __init__(self, store, work, *, device='cpu', cuda_context=None,
                 expected_context_sha256=None, contract=None):
        require(torch.get_num_threads()==1, 'one CPU thread required')
        self.store,self.work=store,work; self.width=len(store.factors)
        self.runtime=Runtime(device,cuda_context,expected_context_sha256,store=store.identity,
                             request=store.request,contract=contract,work=work)
        self.device=device
        self.stats=dict(matrix_constructions=0,coefficient_conversions=0,
                        proposal_tiles=0,proposal_columns=0,iterations_per_tile=128,
                        native_preparation_seconds=0.,proposal_seconds=0.,exact_selection_seconds=0.)
        started=time.monotonic()
        self.common=self.block(store.common.eq,store.common.ine)
        self.experts=tuple(self.block(v.eq.suffix(store.common.eq),v.ine.suffix(store.common.ine))
                           for v in store.templates)
        self.stats['native_preparation_seconds']+=time.monotonic()-started

    def tensor(self, values):
        result=torch.tensor(values,dtype=torch.float64,device='cpu')
        require(bool(torch.isfinite(result).all()), 'finite CPU conversion')
        self.work.tick(); return result

    def matrix(self, rows):
        rr=[]; cc=[]; data=[]; rhs=[]
        for i,row in enumerate(rows):
            self.work.visit('row_visits'); rhs.append(float(row.rhs))
            for j,q in row.coefficients:
                self.work.visit('coefficient_visits'); rr.append(i); cc.append(j); data.append(float(q))
        ix=torch.tensor([rr,cc],dtype=torch.int64,device='cpu')
        m=torch.sparse_coo_tensor(ix,self.tensor(data),(len(rhs),self.width),
                                 dtype=torch.float64,device='cpu').coalesce()
        self.stats['matrix_constructions']+=1; self.stats['coefficient_conversions']+=len(data)
        return m,m.transpose(0,1).coalesce(),self.tensor(rhs)[:,None]

    def block(self, eq, ine):
        a,at,b=self.matrix(ine); e,et,h=self.matrix(eq)
        a,at,b,e,et,h=self.runtime.move((a,at,b,e,et,h))
        return dict(a=a,at=at,b=b,e=e,et=et,h=h)

    def blocks(self, pair):
        started=time.monotonic(); entry=self.store.entries[pair]
        guard=self.block(entry.eq.suffix(self.store.common.eq),entry.ine.suffix(self.store.common.ine))
        self.stats['native_preparation_seconds']+=time.monotonic()-started
        return (self.common,guard,self.experts[pair[0]],self.experts[pair[1]])

    def propose(self, tasks, blocks):
        start=time.monotonic(); targets=objectives(self.store,tasks,self.work); k=len(tasks)
        c=torch.zeros((self.width,k),dtype=torch.float64,device='cpu')
        for i,(_,coeff) in enumerate(targets):
            for j,v in coeff.items(): c[j,i]=float(v)
        d=self.tensor([float(value) for value,_ in targets])
        require(bool(torch.isfinite(c).all()), 'finite objective conversion')
        c,d=self.runtime.move((c,d))
        states=[]
        for p in blocks:
            y=torch.zeros((len(p['b']),k),dtype=torch.float64,device=self.device)
            z=torch.zeros((len(p['h']),k),dtype=torch.float64,device=self.device)
            states.append(dict(p,y=y,z=z,by=y.clone(),bz=z.clone()))
        best=torch.full((k,),-math.inf,dtype=torch.float64,device=self.device)
        self.runtime.sync(); optimization=time.monotonic()
        with torch.no_grad():
            for step in range(129):
                self.work.tick(); r=c.clone(); value=d.clone()
                for p in states:
                    r-=torch.sparse.mm(p['at'],p['y'])+torch.sparse.mm(p['et'],p['z'])
                    value+=(p['b']*p['y']).sum(0)+(p['h']*p['z']).sum(0)
                value-=r.abs().sum(0)
                require(bool(torch.isfinite(r).all() and torch.isfinite(value).all()), 'finite candidate iteration')
                improved=value>best; best=torch.where(improved,value,best)
                for p in states:
                    require(bool(torch.isfinite(p['y']).all() and torch.isfinite(p['z']).all()), 'finite dual iterate')
                    p['by'][:,improved]=p['y'][:,improved]; p['bz'][:,improved]=p['z'][:,improved]
                if step==128: break
                x=-torch.sign(r); rate=.125/math.sqrt(step+1)
                for p in states:
                    p['y']=torch.minimum(p['y']+rate*(p['b']-torch.sparse.mm(p['a'],x)),
                                         torch.zeros_like(p['y'],device=self.device))
                    p['z']=p['z']+rate*(p['h']-torch.sparse.mm(p['e'],x))
        self.runtime.sync()
        self.runtime.costs['optimization_sync_seconds']+=time.monotonic()-optimization
        readbacks=[self.runtime.read(p['by'],p['bz']) for p in states]
        proposed=[]; zeros=[]
        for i,task in enumerate(tasks):
            candidate=dict(query=task.query_id,equality=[],inequality=[],lower='0')
            for ys,zs in readbacks:
                candidate['equality'].extend(str(F(v)) for v in zs[i])
                candidate['inequality'].extend(str(F(v)) for v in ys[i])
            proposed.append(candidate)
            zeros.append(dict(query=task.query_id,equality=['0']*len(candidate['equality']),
                              inequality=['0']*len(candidate['inequality']),lower='0'))
        self.stats['proposal_seconds']+=time.monotonic()-start
        start=time.monotonic()
        bounds,_=evaluate_tile(self.store,tasks,proposed,self.work)
        fallback,_=evaluate_tile(self.store,tasks,zeros,self.work)
        result=[]
        for task,p,z,b,f in zip(tasks,proposed,zeros,bounds,fallback):
            chosen=z if f>b else p; chosen['lower']=str(max(b,f))
            result.append(dict(slot=task.slot,candidate=chosen))
        self.stats['exact_selection_seconds']+=time.monotonic()-start
        self.stats['proposal_tiles']+=1; self.stats['proposal_columns']+=k
        self.work.tick(); return result


def produce(root, store, plan, *, tile_width, work, missing_slots=(), event=None,
            device='cpu',cuda_context=None,expected_context_sha256=None):
    api_start=time.monotonic()
    missing_slots=frozenset(missing_slots)
    require(missing_slots<={t.slot for t in plan.tasks}, 'partial endpoint identity')
    root=Path(root); root.mkdir(exist_ok=False)
    native=Native(store,work,device=device,cuda_context=cuda_context,
                  expected_context_sha256=expected_context_sha256,contract=plan.contract)
    refs=[]; prior=None; blocks=None; serialization=0.; total=0
    for i,tasks in enumerate(plan.tiles(tile_width)):
        if tasks[0].pair!=prior:
            prior=tasks[0].pair; blocks=native.blocks(prior)
        records=native.propose(tasks,blocks); work.tick(); start=time.monotonic()
        for record in records:
            if record['slot'] in missing_slots: record['candidate']=None
        if event is not None: event('before_tile',i,None)
        raw=compact(records); require(len(raw)<=4*2**20, 'producer tile capacity')
        total+=len(raw); require(total<=64*2**20, 'producer proof capacity')
        name=f'{i:05d}.json'; ref=write(root/name,records); refs.append(dict(file=name,**ref))
        if event is not None: event('tile',i,refs[-1])
        serialization+=time.monotonic()-start; work.tick()
    device_report=native.runtime.report()
    ref=write(root/'manifest.json',dict(schema=PROOF,contract=plan.contract,tile_width=tile_width,tiles=refs))
    work.tick()
    return ref['sha256'],dict(native.stats,algorithm=ALGORITHM if device=='cpu' else
                             'CUDA_BLOCK_PROJECTED_SUBGRADIENT_128_V1',device=device,dtype='float64',
                             device_report=device_report,api_seconds=time.monotonic()-api_start,
                             serialization_seconds=serialization,proof_bytes=total+ref['bytes'])
