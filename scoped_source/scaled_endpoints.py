"""Check-once objectives and bounded multi-column exact dual consumption.

Standard library only. Raw source and BOTH candidates are independently
checked. A session-owned receipt is not an input to the portable checker.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from types import MappingProxyType

from scoped_source.check_scaled_references import checked
from scoped_source.forest_endpoints import Task,Plan,THRESHOLD,objective_roots,frozen
from scoped_source.hz_request_store import compact,decode,digest,identity,rational,require,sha

SCHEMA='SCALED_FOREST_ENDPOINT_CONTRACT_V3'
PROOF='SCALED_FOREST_ALL_CANDIDATES_V3'


def pullback(forest,columns,work,workspace=None):
    """Unscaled P=L^T(I-R)^(-T); same coefficient visit, exact work per column."""
    checked(forest,work)
    require(0<len(columns)<=8,'bounded adjoint columns')
    from scoped_source.pair_workspace import PairWorkspace
    if type(workspace) is PairWorkspace:
        workspace.bind(forest,work)
        return workspace.pullback(columns)
    alpha=([dict(c) for c in columns] if workspace is None else
           [workspace.mapping('nodes',c) for c in columns])
    out=([{} for _ in columns] if workspace is None else
         [workspace.mapping('factors') for _ in columns])
    for n in range(len(forest.nodes)-1,-1,-1):
        work.visit('row_visits'); weights=[(k,a.get(n,F(0))) for k,a in enumerate(alpha)]
        weights=[(k,v) for k,v in weights if v]
        if not weights: continue
        _,local,refs=forest.nodes[n]
        for j,q in local:
            work.visit('coefficient_visits')
            for k,w in weights: out[k][j]=work.exact(out[k].get(j,F(0))+work.exact(w*q))
        for p,q in refs:
            work.visit('coefficient_visits')
            for k,w in weights: alpha[k][p]=work.exact(alpha[k].get(p,F(0))+work.exact(w*q))
    if workspace is not None:
        for a in alpha: a.close()
    return out


def weighted_norm(forest,residual,work,*,full=True):
    """Abs before scale grouping. Equal scales never identify coordinates."""
    groups={}
    entries=((j,residual.get(j,F(0))) for j in range(len(forest.factors))) if full else residual.items()
    for j,value in entries:
        work.visit('coefficient_visits'); scale=forest.scales[j]
        groups[scale]=work.exact(groups.get(scale,F(0))+abs(value))
    total=F(0)
    for scale,value in groups.items():
        if full: work.visit('row_visits')
        total=work.exact(total+work.exact(scale*value))
    return total


def contract(forest):
    checked(forest,forest.session); d=forest.declaration
    return dict(schema=SCHEMA,forest=forest.identity,source=forest.source,request=forest.request,
                factors=forest.factor_identity,classes=d['classes'],label=d['label'],margin=d['margin'],
                gate=d['gate'],top_k=d['top_k'],tie_policy=d['tie_policy'])


def _plan(forest,doc,expected,work):
    checked(forest,work); raw=compact(doc)
    require(len(raw)<=8192 and digest(raw)==sha(expected) and decode(raw)==contract(forest),'complete contract binding')
    work.visit('row_visits',len(doc)); d=forest.declaration; classes=d['classes']; label=d['label']
    margin=work.exact(F(d['margin'])); require(2<=classes<=10 and margin>=0,'bounded classification properties')
    gates=[]; tasks=[]; duties=[]; router=forest.views[forest.roles['router']]['output']
    for pair in combinations(range(d['experts']),2):
        work.visit('row_visits'); a,b=pair; ca,na=router[a]; cb,nb=router[b]
        alpha={} if na==nb else {na:F(1),nb:F(-1)}
        r=pullback(forest,[alpha],work)[0]; radius=weighted_norm(forest,r,work,full=False)
        center=work.exact(ca-cb); lo=work.exact(center-radius); hi=work.exact(center+radius)
        gate=((F(1,2),F(1,2)) if lo==hi==0 else (F(0),F(1,2)) if hi<=0 else
              (F(1,2),F(1)) if lo>=0 else (F(0),F(1)))
        gates.append((pair,gate))
        for other in range(classes):
            if other==label: continue
            work.visit('row_visits'); slots=[]
            for weight in sorted(set(gate)):
                work.visit('row_visits'); left=[]; right=[]
                for j in range(classes):
                    q=F(int(j==label)-int(j==other))
                    left.append(str(work.exact(weight*q))); right.append(str(work.exact((1-weight)*q)))
                left,right=tuple(left),tuple(right)
                query=identity(dict(forest=forest.identity,request=forest.request,pair=list(pair),
                                    left=left,right=right,offset=str(-margin)))
                slot=identity(dict(contract=expected,query=query,other=other,weight=str(weight)))
                tasks.append(Task(pair,other,weight,left,right,str(-margin),query,slot)); slots.append(slot)
            duties.append((pair,other,tuple(slots)))
    require(len(tasks)<=504 and len({t.slot for t in tasks})==len(tasks),'complete endpoint inventory')
    work.tick(); return Plan(forest.identity,expected,tuple(gates),tuple(tasks),tuple(duties))


@dataclass(frozen=True)
class Receipt:
    record: object


class _JSONTile:
    """Legacy wire format; mathematical consumption below is shared."""
    def __init__(self,records,tasks):
        require(type(records) is list and len(records)==len(tasks),'complete endpoint tile')
        self.records=records; self.tasks=tasks

    def present(self,k):
        record=self.records[k]; task=self.tasks[k]
        require(type(record) is dict and set(record)=={'slot','proposed','zero'} and record['slot']==task.slot,'slot binding/order')
        require((record['proposed'] is None)==(record['zero'] is None),'partial candidate pair')
        return record['proposed'] is not None

    def candidate(self,k,role):
        c=self.records[k][role]
        require(type(c) is dict and set(c)=={'query','equality','inequality','lower'}
                and c['query']==self.tasks[k].query_id,'raw candidate fields/binding')
        rational(c['lower'])

    def dimensions(self,k,role,key,count):
        v=self.records[k][role][key]
        require(type(v) is list and len(v)==count,'full dual roster')

    def weight(self,k,role,key,i): return rational(self.records[k][role][key][i])


class Session:
    """Private complete plan, one live tile; receipts authorized by object identity.

    This is an in-process API trust boundary, not a sandbox against malicious
    Python code changing private memory. Durable inputs never trust receipts.
    """
    def __init__(self,forest,doc,*,expected,work,width=8,retain_raw=True,workspace=None):
        checked(forest,work); require(type(width) is int and 1<=width<=8,'bounded tile width')
        require(type(retain_raw) is bool,'explicit evidence retention mode')
        if workspace is not None:
            from scoped_source.bounded_runtime import Workspace
            from scoped_source.pair_workspace import PairWorkspace,CANDIDATE_WIDTH
            require(type(workspace) in (Workspace,PairWorkspace) and not retain_raw,'optional owned streaming workspace')
            require(type(workspace) is not PairWorkspace or width==CANDIDATE_WIDTH,'scoped candidate width 8')
            workspace.bind(forest,work)
        from scoped_source.pair_workspace import PairWorkspace
        self.__scoped=type(workspace) is PairWorkspace
        self.__workspace=workspace
        self.__retain_raw=retain_raw
        self.__forest=forest; self.__work=work; self.__plan=_plan(forest,doc,expected,work)
        self.__tiles=tuple(self.__plan.tiles(width)); self.__width=width; self.__next=0
        self.__live=None; self.__issued={}; self.__values={}; self.__raw=[]; self.__bytes=0
        self.__closed=False

    @property
    def tasks(self): return self.__plan.tasks

    @property
    def gates(self): return self.__plan.gates

    @property
    def tile_count(self): return len(self.__tiles)

    def _check(self):
        require(not self.__closed,'closed checker session'); checked(self.__forest,self.__work)

    def prepare(self,index):
        self._check(); require(type(index) is int and index==self.__next and self.__live is None
                               and index<len(self.__tiles),'one next private tile')
        tasks=self.__tiles[index]; roots=[]; objectives=[]
        if self.__scoped: self.__workspace.select(tasks[0].pair)
        for task in tasks:
            d,alpha=(objective_roots(self.__forest,task,self.__work) if self.__workspace is None
                     else self.__workspace.objective(task))
            roots.append((d,MappingProxyType(alpha)))
            if not self.__scoped:
                objectives.append(pullback(self.__forest,[alpha],self.__work,self.__workspace)[0])
        self.__live=(tasks,tuple(roots),objectives)
        return tasks,tuple(roots)

    def accept(self,records):
        self._check(); require(self.__live is not None,'prepare before accepting raw candidates')
        require(not self.__scoped,'pair-scoped consumer requires published direct frame')
        raw=records if type(records) is bytes else compact(records)
        require(len(raw)<=4*2**20,'bounded raw candidate tile')
        records=decode(raw); require(compact(records)==raw,'canonical raw candidate bytes')
        return self._consume(_JSONTile(records,self.__live[0]),raw)

    def accept_direct(self,tile):
        self._check(); require(self.__live is not None and self.__workspace is not None,'owned direct tile only')
        from scoped_source.direct_candidates import Tile
        owner=self.__workspace.owner
        require(type(tile) is Tile and tile is owner._tile and tile.owner is owner
                and tile.tasks==self.__live[0] and tile.ordinal==self.__next,'issued direct tile identity')
        return self._consume(tile,tile.raw)

    def _consume(self,access,raw):
        # This private entry is reached only through the two closed accessors.
        tasks,roots,objectives=self.__live; work=self.__work
        self.__bytes+=len(raw); require(self.__bytes<=16*2**20-8192,'bounded cumulative evidence')
        issued=[]; candidate_hash=digest(raw)
        from scoped_source.pair_workspace import CHECK_WIDTH
        width=CHECK_WIDTH if self.__scoped else len(tasks)
        for start in range(0,len(tasks),width):
            stop=min(start+width,len(tasks))
            try:
                block_objectives=objectives[start:stop]
                if self.__scoped:
                    # Sparse descriptors are expanded only AFTER full publication.
                    block_objectives=[]
                    for _,alpha in roots[start:stop]:
                        private=self.__workspace.mapping('nodes',alpha)
                        block_objectives.append(pullback(self.__forest,[private],work,self.__workspace)[0])
                issued.extend(self._consume_block(access,tasks[start:stop],roots[start:stop],
                                                  block_objectives,start,candidate_hash))
                if self.__scoped:
                    self.__workspace.blocks_checked+=1
                    self.__workspace.maximum_block=max(self.__workspace.maximum_block,stop-start)
            finally:
                if self.__scoped: self.__workspace.clear_block()
            work.tick()  # Release cost is inside the original deadline.
        work.tick()
        if self.__retain_raw: self.__raw.append(raw)
        self.__next+=1; self.__live=None
        if self.__workspace is not None: self.__workspace.clear_tile()
        return tuple(issued)

    def _consume_block(self,access,tasks,roots,objectives,start,candidate_hash):
        # Shared mathematical consumer for legacy, dense and scoped paths.
        forest=self.__forest; work=self.__work
        present=[]; alpha=([{} for _ in tasks] if self.__workspace is None else
                           [self.__workspace.mapping('nodes') for _ in tasks])
        dual_rhs=[F(0) for _ in tasks]
        for k,task in enumerate(tasks):
            if not access.present(start+k): continue
            present.append(k)
            for role in ('proposed','zero'):
                work.visit('row_visits',4); access.candidate(start+k,role)
        for kind,key in (('eq','equality'),('ine','inequality')):
            rows=forest.rows(tasks[0].pair,kind)
            for k in present:
                for role in ('proposed','zero'): access.dimensions(start+k,role,key,len(rows))
            for i,row in enumerate(rows):
                work.visit('row_visits')
                rhs,n=(forest.row(row) if self.__workspace is None else self.__workspace.roots.row(row))
                for k in present:
                    for role in ('proposed','zero'):
                        work.visit('coefficient_visits'); w=access.weight(start+k,role,key,i)
                        require(kind!='ine' or w<=0,'inequality dual sign')
                        if role=='zero': require(w==0,'explicit zero must be zero'); continue
                        if w:
                            dual_rhs[k]=work.exact(dual_rhs[k]+work.exact(rhs*w))
                            alpha[k][n]=work.exact(alpha[k].get(n,F(0))-w)
        duals=pullback(forest,alpha,work,self.__workspace) if present else [{} for _ in tasks]
        issued=[]
        for k,task in enumerate(tasks):
            if k not in present:
                self.__values[task.slot]=None; issued.append(None); continue
            work.visit('row_visits'); residual=duals[k]; objective=objectives[k]; d=roots[k][0]
            for j in range(len(forest.factors)):
                work.visit('coefficient_visits')
                residual[j]=work.exact(residual.get(j,F(0))+objective.get(j,F(0)))
            work.visit('row_visits'); proposed=work.exact(work.exact(d+dual_rhs[k])-weighted_norm(forest,residual,work))
            work.visit('row_visits'); zero=work.exact(d-weighted_norm(forest,objective,work))
            selected='zero' if zero>proposed else 'proposed'; value=max(proposed,zero)
            work.visit('row_visits',2)  # Exact bound encoding.
            record=dict(schema='CHECKED_SCALED_ENDPOINT_RECEIPT_V3',forest=forest.identity,
                source=forest.source,request=forest.request,factors=forest.factor_identity,
                contract=self.__plan.contract,slot=task.slot,query=task.query_id,pair=task.pair,
                other=task.other,weight=str(task.weight),raw_tile=candidate_hash,
                proposed=str(proposed),zero=str(zero),selected=selected,lower=str(value),
                ordinal=(self.__next,start+k),policy='EXACT_RESIDUAL_STRICT_GT_1E-7')
            work.visit('row_visits',len(record)); receipt=Receipt(frozen(record)); work.visit('row_visits')
            self.__issued[task.slot]=receipt; self.__values[task.slot]=value; issued.append(receipt)
        return tuple(issued)

    def abort(self):
        """Fail-closed owner cleanup even after Work's deadline; no counter reset."""
        self.__closed=True; self.__live=None; self.__issued.clear(); self.__values.clear(); self.__raw.clear()
        if self.__workspace is not None: self.__workspace.close()

    def finish(self,receipts):
        self._check(); work=self.__work
        require(self.__live is None and self.__next==len(self.__tiles),'all tiles completed before terminal')
        require(type(receipts) in (list,tuple) and len(receipts)==len(self.__plan.tasks),'all endpoint receipts')
        for task,receipt in zip(self.__plan.tasks,receipts,strict=True):
            work.visit('row_visits')
            if self.__values[task.slot] is None: require(receipt is None,'missing receipt cannot be invented')
            else: require(type(receipt) is Receipt and self.__issued.get(task.slot) is receipt,'checker-owned receipt identity/session')
        rows=[]
        for pair,other,slots in self.__plan.duties:
            work.visit('row_visits'); bounds=[self.__values[s] for s in slots]
            lo=None if None in bounds else min(bounds)
            rows.append(dict(pair=list(pair),other=other,lower=None if lo is None else str(lo),positive=lo is not None and lo>THRESHOLD))
        positive=sum(r['positive'] for r in rows); missing=sum(v is None for v in self.__values.values())
        result=dict(status=('CHECKED_POSITIVE_CAPTURED_REAL_SOURCE' if positive==len(rows) else
                            'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE'),
            required=len(rows),positive=positive,required_endpoints=len(self.__plan.tasks),
            checked_endpoints=len(self.__values)-missing,missing_endpoints=missing,results=rows,
            native_float_proof=False,checkpoint_correspondence_checked=False)
        from scoped_source.wire_forest import CheckedWireForest
        if type(self.__forest) is CheckedWireForest: self.__forest.source_access.seal()
        if type(self.__forest).__module__=='scoped_source.check_direct_source':
            from scoped_source.check_direct_source import CheckedDirectForest
            require(type(self.__forest) is CheckedDirectForest,'closed direct type')
            self.__forest.seal()
        work.visit('row_visits',len(result)); work.tick(); self.__closed=True; return result

    def proof(self):
        self.__work.tick(); require(self.__closed,'terminal before final proof serialization')
        require(self.__retain_raw,'streaming session requires immutable tile manifest')
        result=dict(schema=PROOF,contract=self.__plan.contract,width=self.__width,
                    tiles=[decode(raw) for raw in self.__raw])
        self.__work.visit('row_visits',len(result)); require(len(compact(result))<=16*2**20,'full proof cap')
        self.__work.tick(); return result


def check_proof(forest,doc,proof,*,expected_contract,expected_proof,work):
    raw=compact(proof)
    require(len(raw)<=16*2**20 and digest(raw)==sha(expected_proof),'raw proof identity/capacity')
    proof=decode(raw)
    require(type(proof) is dict and set(proof)=={'schema','contract','width','tiles'} and
            proof['schema']==PROOF and proof['contract']==expected_contract,'proof binding/fields')
    session=Session(forest,doc,expected=expected_contract,work=work,width=proof['width'])
    require(type(proof['tiles']) is list and len(proof['tiles'])==session.tile_count,'all raw tiles required')
    receipts=[]
    for i,records in enumerate(proof['tiles']):
        session.prepare(i); receipts.extend(session.accept(records))
    return session.finish(receipts)
