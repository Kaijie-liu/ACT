"""Checked source-reference forest and complete support acceptance; stdlib only.

No eager Store, producer, optimizer or numerical library is used here. A
captured real-arithmetic source is checked, not its live checkpoint capture or
the floating-point execution of that network.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from types import MappingProxyType

from scoped_source.check_source_dag import Checker
from scoped_source.hz_request_store import compact, decode, digest, identity, rational, require, sha

SCHEMA = 'CHECKED_FOREST_ENDPOINT_CONTRACT_V1'
PROOF = 'CHECKED_FOREST_ENDPOINT_PROOF_V1'
THRESHOLD = F(1, 10000000)


def frozen(x):
    if type(x) is dict: return MappingProxyType({k: frozen(v) for k,v in x.items()})
    if type(x) in (tuple,list): return tuple(frozen(v) for v in x)
    return x


class CheckedForest:
    __slots__ = ('identity','source','request','declaration','factors','nodes','views',
                 'roles','complete','receipt','_sealed')

    def __setattr__(self,k,v):
        if getattr(self,'_sealed',False): raise AttributeError('immutable checked forest')
        object.__setattr__(self,k,v)

    def __delattr__(self,k):
        raise AttributeError('immutable checked forest')

    def __init__(self, raw, root, source, *, expected, work, cut=None):
        c=Checker(raw,root,source,expected,work,cut)
        result=c.run()  # No checked handle can escape a partial/failed check.
        self.identity=sha(expected); self.source=sha(source)
        self.request=result['request']; self.declaration=frozen(c.reader.request)
        self.factors=c.factors; self.nodes=tuple(c.nodes)
        self.views=frozen(c.views); self.roles=frozen(c.roles)
        self.complete=c.cut is None; self.receipt=frozen(result)
        # The native one-level operator may ONLY consume this checked shape.
        for _,_,refs in self.nodes:
            work.visit('row_visits')
            require(not refs or len(refs)==1 and refs[0][1]==-1 and not self.nodes[refs[0][0]][2],
                    'source one-level reference structure')
        for v in self.views:
            require(all(not self.nodes[n][2] for _,n in v['output']), 'reference-free output roots')
        self._sealed=True; work.tick()

    def chain(self, vi, stop=None):
        out=[]
        while vi!=stop:
            require(vi is not None, 'source ancestry stops at common')
            out.append(vi); vi=self.views[vi]['parent']
        return tuple(reversed(out))

    def rows(self, pair, kind):
        require(self.complete and kind in ('eq','ine'), 'complete forest constraints')
        require(type(pair) is tuple and len(pair)==2 and all(type(i) is int for i in pair), 'pair type')
        guards={(a,b):v for a,b,v in self.roles['pairs']}
        require(pair in guards, 'tie-legal unordered pair inventory')
        common=self.roles['common']; a,b=pair
        chain=(*self.chain(common),guards[pair],
               *self.chain(self.roles['experts'][a],common),*self.chain(self.roles['experts'][b],common))
        return tuple((vi,kind,k) for vi in chain for k in range(len(self.views[vi][kind])))

    def row(self, row_id):
        vi,kind,k=row_id
        return self.views[vi][kind][k]


def pullback(forest, alpha, work):
    """Exact reverse reference accumulation into ONE global factor residual."""
    require(type(forest) is CheckedForest and getattr(forest,'_sealed',False) is True, 'checked source handle required')
    a=dict(alpha); result={}
    for n in range(len(forest.nodes)-1,-1,-1):
        work.visit('row_visits'); w=a.get(n,F(0))
        if not w: continue
        _,local,refs=forest.nodes[n]
        for j,q in local:
            work.visit('coefficient_visits')
            result[j]=work.exact(result.get(j,F(0))+work.exact(w*q))
        for p,s in refs:
            work.visit('coefficient_visits'); a[p]=work.exact(a.get(p,F(0))+work.exact(w*s))
    return {j:q for j,q in result.items() if q}


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


@dataclass(frozen=True)
class Plan:
    forest: str
    contract: str
    gates: tuple
    tasks: tuple
    duties: tuple

    def tiles(self,width):
        require(type(width) is int and 1<=width<=8, 'bounded endpoint tile')
        for pair,_ in self.gates:
            items=tuple(t for t in self.tasks if t.pair==pair)
            for start in range(0,len(items),width): yield items[start:start+width]


def contract(forest):
    require(type(forest) is CheckedForest and getattr(forest,'_sealed',False) is True and forest.complete,
            'complete checked source required')
    r=forest.declaration
    return dict(schema=SCHEMA,forest=forest.identity,source=forest.source,request=forest.request,
                classes=r['classes'],label=r['label'],margin=r['margin'],gate=r['gate'],
                top_k=r['top_k'],tie_policy=r['tie_policy'])


def plan(forest, doc, *, expected, work):
    raw=compact(doc)
    require(len(raw)<=8192 and digest(raw)==sha(expected), 'endpoint contract identity/capacity')
    require(decode(raw)==contract(forest), 'property/source/semantics binding')
    d=forest.declaration; c=d['classes']; y=d['label']; margin=work.exact(F(d['margin']))
    require(2<=c<=10 and margin>=0, 'bounded complete class inventory')
    router=forest.views[forest.roles['router']]['output']; gates=[]; tasks=[]; duties=[]
    for pair in combinations(range(d['experts']),2):
        a,b=pair; ca,na=router[a]; cb,nb=router[b]
        coeff=pullback(forest,{na:F(1)} if na!=nb else {},work) if na!=nb else {}
        if na!=nb:
            for j,q in pullback(forest,{nb:F(1)},work).items(): coeff[j]=work.exact(coeff.get(j,F(0))-q)
        radius=F(0)
        for q in coeff.values(): radius=work.exact(radius+abs(q))
        center=work.exact(ca-cb); lo=work.exact(center-radius); hi=work.exact(center+radius)
        gate=((F(1,2),F(1,2)) if lo==hi==0 else (F(0),F(1,2)) if hi<=0 else
              (F(1,2),F(1)) if lo>=0 else (F(0),F(1)))
        gates.append((pair,gate))
        for k in range(c):
            if k==y: continue
            slots=[]
            for t in sorted(set(gate)):
                q=[F(int(j==y)-int(j==k)) for j in range(c)]
                left=tuple(str(t*v) for v in q); right=tuple(str((1-t)*v) for v in q)
                query=identity(dict(forest=forest.identity,request=forest.request,pair=list(pair),
                                    left=left,right=right,offset=str(-margin)))
                slot=identity(dict(contract=expected,query=query,other=k,weight=str(t)))
                tasks.append(Task(pair,k,t,left,right,str(-margin),query,slot)); slots.append(slot)
            duties.append((pair,k,tuple(slots)))
    require(len(tasks)<=504 and len({t.slot for t in tasks})==len(tasks), 'complete endpoint capacity')
    work.tick(); return Plan(forest.identity,expected,tuple(gates),tuple(tasks),tuple(duties))


def objective_roots(forest, task, work):
    d=rational(task.offset); alpha={}
    for e,weights in zip(task.pair,(task.left,task.right)):
        rows=forest.views[forest.roles['experts'][e]]['output']
        require(len(weights)==len(rows), 'all output roots')
        for (c,n),w in zip(rows,weights):
            work.visit('row_visits'); w=rational(w)
            if w:
                d=work.exact(d+work.exact(c*w)); alpha[n]=work.exact(alpha.get(n,F(0))+w)
    return d,alpha


def evaluate_tile(forest, tasks, candidates, work):
    require(type(forest) is CheckedForest and getattr(forest,'_sealed',False) is True and forest.complete,
            'complete checked source handle')
    require(0<len(tasks)<=8 and len(tasks)==len(candidates) and len({t.pair for t in tasks})==1,
            'one pair and bounded complete tile')
    # Caller containers cannot change while Work hooks / arithmetic execute.
    raw=compact(candidates); require(len(raw)<=4*2**20, 'candidate tile capacity'); candidates=decode(raw)
    roots=[objective_roots(forest,t,work) for t in tasks]; low=[d for d,_ in roots]; alpha=[a for _,a in roots]
    for t,c in zip(tasks,candidates):
        require(type(c) is dict and set(c)=={'query','equality','inequality','lower'} and c['query']==t.query_id,
                'candidate endpoint binding')
    for kind,key in (('eq','equality'),('ine','inequality')):
        rows=forest.rows(tasks[0].pair,kind)
        require(all(type(c[key]) is list and len(c[key])==len(rows) for c in candidates), 'full original dual roster')
        for i,row in enumerate(rows):
            work.visit('row_visits'); rhs,n=forest.row(row)
            for col,c in enumerate(candidates):
                w=rational(c[key][i]); require(kind!='ine' or w<=0, 'inequality dual sign')
                if w:
                    low[col]=work.exact(low[col]+work.exact(rhs*w))
                    alpha[col][n]=work.exact(alpha[col].get(n,F(0))-w)
    residual=[]
    for i,a in enumerate(alpha):
        r=pullback(forest,a,work); residual.append(r)
        for q in r.values(): low[i]=work.exact(low[i]-abs(q))
    return low,residual


def _check_records(forest, inventory, records, work):
    """Internal bound helper; only check_proof issues a checked terminal."""
    require(inventory.forest==forest.identity, 'plan forest binding')
    raw=compact(records); require(len(raw)<=4*2**20, 'record tile capacity'); records=decode(raw)
    require(type(records) is list and 0<len(records)<=8, 'bounded record list')
    by_slot={t.slot:t for t in inventory.tasks}; selected=[]; candidates=[]; values={}
    for r in records:
        require(type(r) is dict and set(r)=={'slot','candidate'} and r['slot'] in by_slot and r['slot'] not in values,
                'endpoint slot identity/uniqueness')
        values[r['slot']]=None
        if r['candidate'] is not None: selected.append(by_slot[r['slot']]); candidates.append(r['candidate'])
    if selected:
        bounds,_=evaluate_tile(forest,selected,candidates,work)
        for t,c,b in zip(selected,candidates,bounds):
            require(rational(c['lower'])==b, 'claimed exact bound'); values[t.slot]=b
    return values


def _aggregate(inventory, values, work):
    """Untrusted-value aggregation, NOT an acceptance/positive-proof API."""
    require(len(inventory.tasks)>0 and len(inventory.duties)>0, 'nonempty obligation inventory')
    require(set(values)=={t.slot for t in inventory.tasks}, 'complete endpoint inventory')
    rows=[]
    for pair,k,slots in inventory.duties:
        work.tick(); bb=[values[s] for s in slots]; lo=None if None in bb else min(bb)
        rows.append(dict(pair=list(pair),other=k,lower=None if lo is None else str(lo),
                         positive=lo is not None and lo>THRESHOLD))
    missing=sum(v is None for v in values.values()); positive=sum(r['positive'] for r in rows)
    status=('AGGREGATE_POSITIVE_UNVERIFIED_INPUTS' if positive==len(rows) else
            'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE')
    return dict(status=status,required=len(rows),positive=positive,required_endpoints=len(inventory.tasks),
                checked_endpoints=len(values)-missing,missing_endpoints=missing,results=rows,
                native_float_proof=False,checkpoint_correspondence_checked=False)


def check_proof(forest, doc, proof, *, expected_contract, expected_proof, work):
    inventory=plan(forest,doc,expected=expected_contract,work=work)
    raw=compact(proof); require(len(raw)<=16*2**20 and digest(raw)==sha(expected_proof), 'proof identity/capacity')
    proof=decode(raw)
    require(type(proof) is dict and set(proof)=={'schema','contract','width','tiles'} and
            proof['schema']==PROOF and proof['contract']==expected_contract, 'proof fields/binding')
    tiles=list(inventory.tiles(proof['width']))
    require(type(proof['tiles']) is list and len(proof['tiles'])==len(tiles), 'complete tile inventory')
    values={}
    for tasks,records in zip(tiles,proof['tiles']):
        require(type(records) is list and [r.get('slot') for r in records]==[t.slot for t in tasks], 'complete tile order')
        values.update(_check_records(forest,inventory,records,work))
    work.tick(); result=_aggregate(inventory,values,work)
    if result['status']=='AGGREGATE_POSITIVE_UNVERIFIED_INPUTS':
        result['status']='CHECKED_POSITIVE_CAPTURED_REAL_SOURCE'
    return result
