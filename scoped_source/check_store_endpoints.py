"""Complete weighted top-2 endpoint checks on immutable source rows (stdlib).

The input declaration, not a model checkpoint, is the source trust boundary.
Binary factors are relaxed to [-1,1]. No producer or numerical solver is used.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

from scoped_source.hz_request_store import (
    Work, Store, compact, decode, digest, identity, rational, require, sha,
)
from scoped_source.check_hz_store_source import check_source, combine

SCHEMA = 'STORE_ENDPOINT_CONTRACT_V1'
PROOF = 'STORE_ENDPOINT_TILES_V1'
THRESHOLD = F(1,10000000)
MAX_VISITS = 96000000
ALGORITHM = 'CPU_BLOCK_PROJECTED_SUBGRADIENT_128_V1'


class EndpointWork(Work):
    """New composition budget, cumulative across source, pairs and tiles."""
    def visit(self, kind, amount=1):
        self.tick(); self.counts[kind] += amount
        require(sum(self.counts.values()) <= MAX_VISITS, 'endpoint composition work capacity')


def load(path, limit, expected=None):
    p=Path(path)
    require(not p.is_symlink() and p.is_file() and p.stat().st_size<=limit,
            'endpoint member absent/capacity/symlink')
    with p.open('rb') as stream: raw=stream.read(limit+1)
    require(len(raw)<=limit, 'endpoint read capacity')
    if expected is not None: require(digest(raw)==sha(expected), 'endpoint member identity')
    value=decode(raw); require(compact(value)==raw, 'endpoint canonical bytes')
    return value


@dataclass(frozen=True)
class Task:
    pair: tuple
    other: int
    weight: F
    left: tuple
    right: tuple
    offset: str
    query_id: str
    slot: str

    def query(self, store):
        return dict(store=store.identity,request=store.request,pair=list(self.pair),
                    left=list(self.left),right=list(self.right),offset=self.offset)


@dataclass(frozen=True)
class Plan:
    contract: str
    mode: str
    gates: tuple
    tasks: tuple
    duties: tuple

    def tiles(self, width):
        require(type(width) is int and 1<=width<=8, 'endpoint tile width')
        for pair,_ in self.gates:
            items=tuple(t for t in self.tasks if t.pair==pair)
            for start in range(0,len(items),width): yield items[start:start+width]


def plan(store, contract, *, expected_contract, declaration, work):
    raw=compact(contract); require(len(raw)<=8192 and digest(raw)==sha(expected_contract),
                                  'external endpoint contract')
    doc=decode(raw); work.tick()
    require(set(doc)=={'schema','store','request','mode','classes','label','margin','semantics','ties'}
            and doc['schema']==SCHEMA and doc['store']==store.identity
            and doc['request']==store.request and doc['semantics']=='selected-softmax-top2'
            and doc['ties']=='ANY_LEGAL_TOPK', 'endpoint source/semantics binding')
    c,y=doc['classes'],doc['label']; margin=rational(doc['margin'])
    require(type(c) is int and 2<=c<=10 and type(y) is int and 0<=y<c
            and margin>=0 and all(v.output.count==c for v in store.templates),
            'complete classification property dimensions')
    require(doc['mode'] in ('checked_source','given_hz'), 'endpoint source mode')
    router=None
    if doc['mode']=='checked_source':
        check_source(store,declaration,expected_source=store.request,work=work)
        router=tuple(store.common.parent.output)
    else: require(declaration is None, 'given HZ cannot claim a source proof')
    gates=[]; tasks=[]; duties=[]
    for pair in combinations(range(store.experts),2):
        work.tick(); gate=(F(0),F(1))
        if router is not None:
            center,coeff=combine(router,[(pair[0],F(1)),(pair[1],F(-1))],F(0),work)
            radius=F(0)
            for v in coeff.values(): radius=work.exact(radius+abs(v))
            lo,hi=work.exact(center-radius),work.exact(center+radius)
            gate=((F(1,2),F(1,2)) if lo==hi==0 else
                  (F(0),F(1,2)) if hi<=0 else (F(1,2),F(1)) if lo>=0 else gate)
        gates.append((pair,gate))
        for k in range(c):
            if k==y: continue
            ids=[]
            for t in sorted(set(gate)):
                q=[F(int(j==y)-int(j==k)) for j in range(c)]
                query=dict(store=store.identity,request=store.request,pair=list(pair),
                    left=[str(t*v) for v in q],right=[str((1-t)*v) for v in q],offset=str(-margin))
                query_id=identity(query)
                slot=identity(dict(contract=expected_contract,query=query_id,other=k,weight=str(t)))
                tasks.append(Task(pair,k,t,tuple(query['left']),tuple(query['right']),query['offset'],query_id,slot))
                ids.append(slot)
            duties.append((pair,k,tuple(ids)))
    require(len(tasks)<=504 and len(set(t.slot for t in tasks))==len(tasks), 'complete endpoint capacity/identity')
    work.tick(); return Plan(expected_contract,doc['mode'],tuple(gates),tuple(tasks),tuple(duties))


def objectives(store, tasks, work):
    """Only objective coefficients, not copies of the base/source matrices."""
    require(0<len(tasks)<=8 and len({t.pair for t in tasks})==1, 'one pair per bounded tile')
    _,a,b,_=store.pair(tasks[0].pair); targets=[]
    for t in tasks:
        coeff={}; d=rational(t.offset)
        for view,weights in ((a,t.left),(b,t.right)):
            require(len(weights)==view.output.count, 'target output dimensions')
            for row,value in zip(view.output,weights):
                work.visit('row_visits'); w=rational(value)
                for j,v in row.coefficients:
                    work.visit('coefficient_visits')
                    if w: coeff[j]=work.exact(coeff.get(j,F(0))+work.exact(w*v))
                if w: d=work.exact(d+work.exact(w*row.rhs))
        targets.append((d,{j:v for j,v in coeff.items() if v}))
    return targets


def evaluate_tile(store, tasks, candidates, work):
    """Read each original constraint row once per tile, exact residual per column.

Zero multipliers need no multiply, but row identities/coefficients are still
visited. Unmentioned factors have exactly zero residual, not dropped relations.
"""
    require(len(tasks)==len(candidates), 'tile candidate count')
    targets=objectives(store,tasks,work); lower=[x[0] for x in targets]; r=[dict(x[1]) for x in targets]
    pair=tasks[0].pair; entry,a,b,_=store.pair(pair)
    for kind in ('eq','ine'):
        count=(getattr(entry,kind).count+getattr(a,kind).count+getattr(b,kind).count
               -2*getattr(store.common,kind).count)
        key='equality' if kind=='eq' else 'inequality'; vectors=[]
        for task,candidate in zip(tasks,candidates):
            require(type(candidate) is dict and set(candidate)=={'query','equality','inequality','lower'}
                    and candidate['query']==task.query_id, 'candidate endpoint binding')
            values=candidate[key]
            require(type(values) is list and len(values)==count, 'candidate full dual inventory')
            vectors.append(values)
        for i,row in enumerate(store.constraints(pair,kind)):
            work.visit('row_visits'); multipliers=[]
            for values in vectors:
                v=rational(values[i]); require(kind!='ine' or v<=0, 'inequality dual sign')
                multipliers.append(v)
            for column,v in enumerate(multipliers):
                if v: lower[column]=work.exact(lower[column]+work.exact(v*row.rhs))
            for j,q in row.coefficients:
                work.visit('coefficient_visits')
                for column,v in enumerate(multipliers):
                    if v: r[column][j]=work.exact(r[column].get(j,F(0))-work.exact(v*q))
    for i,residual in enumerate(r):
        for j in sorted(residual): lower[i]=work.exact(lower[i]-abs(residual[j]))
    work.tick(); return lower,r


def check_tile(store, tasks, records, work):
    require(type(records) is list and len(records)==len(tasks), 'complete tile slots')
    # Bounded private capture before any arithmetic; no live alias re-hashing.
    for record in records:
        require(type(record) is dict and set(record)=={'slot','candidate'}, 'tile record fields')
    raw=compact(records); require(len(raw)<=4*2**20, 'tile byte capacity')
    records=decode(raw); present=[]; candidates=[]; values={}
    for task,record in zip(tasks,records):
        require(record['slot']==task.slot, 'tile slot order/binding')
        if record['candidate'] is None: values[task.slot]=None
        else: present.append(task); candidates.append(record['candidate'])
    if present:
        bounds,_=evaluate_tile(store,present,candidates,work)
        for task,candidate,bound in zip(present,candidates,bounds):
            require(rational(candidate['lower'])==bound, 'candidate claimed exact lower bound')
            values[task.slot]=bound
    work.tick(); return values


def aggregate(plan, values, work):
    require(set(values)=={t.slot for t in plan.tasks}, 'complete request endpoint inventory')
    results=[]
    for pair,k,slots in plan.duties:
        work.tick(); bounds=[values[s] for s in slots]
        lo=None if None in bounds else min(bounds)
        results.append(dict(pair=list(pair),other=k,endpoints=list(slots),
                            lower=None if lo is None else str(lo),positive=lo is not None and lo>THRESHOLD))
    positive=sum(r['positive'] for r in results); missing=sum(v is None for v in values.values())
    status=('CHECKED_POSITIVE_DECLARED_SOURCE' if plan.mode=='checked_source' else 'CHECKED_POSITIVE_GIVEN_HZ_AND_GATE')
    if positive!=len(results): status='UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE'
    return dict(status=status,required=len(results),positive=positive,required_endpoints=len(plan.tasks),
        checked_endpoints=len(values)-missing,missing_endpoints=missing,results=results,
        native_float_proof=False,checkpoint_correspondence_checked=False)


def check_directory(root, *, expected_contract, expected_proof, work):
    root=Path(root); work.tick(); require(not root.is_symlink(), 'endpoint root symlink')
    contract=load(root/'contract.json',8192,expected_contract)
    store=Store(root/'store',expected_manifest=contract['store'],expected_request=contract['request'],work=work)
    declaration=load(root/'source.json',4*2**20,contract['request']) if contract['mode']=='checked_source' else None
    inventory=plan(store,contract,expected_contract=expected_contract,declaration=declaration,work=work)
    proof_root=root/'proof'; require(not proof_root.is_symlink(), 'proof root symlink')
    proof=load(proof_root/'manifest.json',2**20,expected_proof)
    require(type(proof) is dict and set(proof)=={'schema','contract','tile_width','tiles'}
            and proof['schema']==PROOF and proof['contract']==expected_contract, 'proof contract binding')
    tiles=list(inventory.tiles(proof['tile_width'])); refs=proof['tiles']
    require(type(refs) is list and len(refs)==len(tiles), 'complete tile manifest')
    names={'manifest.json'}; values={}; total=0
    for i,(tasks,ref) in enumerate(zip(tiles,refs)):
        work.tick(); name=f'{i:05d}.json'
        require(type(ref) is dict and set(ref)=={'file','sha256','bytes'} and ref['file']==name
                and type(ref['bytes']) is int and 0<ref['bytes']<=4*2**20, 'tile reference')
        total+=ref['bytes']; require(total<=64*2**20, 'proof total byte capacity')
        records=load(proof_root/name,ref['bytes'],ref['sha256'])
        require(len(compact(records))==ref['bytes'], 'tile exact length')
        values.update(check_tile(store,tasks,records,work)); names.add(name)
    require({p.name for p in proof_root.iterdir()}==names and all(p.is_file() and not p.is_symlink()
            for p in proof_root.iterdir()), 'complete proof file inventory')
    work.tick(); return aggregate(inventory,values,work)
