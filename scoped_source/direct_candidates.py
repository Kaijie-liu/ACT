"""Direct immutable candidate frames; no logical JSON replay or byte proxy.

The caller supplies only one bounded tile. Publication is external and charged
before acceptance; the Driver is fail-closed for the entire invocation.
"""
from fractions import Fraction as F
import math
import struct
import time

from scoped_source.hz_request_store import POLICY,require,rational,digest,sha
from scoped_source.bounded_runtime import scalar_size,encode_scalar,decode_scalar,Workspace

PREFIX=struct.Struct('<8s32s32s32sIIIIII')
RECORD=struct.Struct('<32s32sBB6s')
MAGIC=b'HZDTC001'
MAX_TILE=4*2**20
MAX_TOTAL=16*2**20-8192  # Retain the old metadata reserve, not hypothetical JSON size.
RATIONAL=scalar_size(POLICY['stored_bits'])


def _native(raw):
    x=struct.unpack('<d',raw)[0]
    require(math.isfinite(x) and (x!=0 or raw==bytes(8)),'finite canonical binary64')
    return F(x)


class Tile:
    """Accessor over immutable owned bytes; only Driver issues an accepted tile."""
    __slots__=('_width','_eq','_ine','_stride','raw','owner','tasks','ordinal','profile','_sealed')

    def __setattr__(self,name,value):
        require(not getattr(self,'_sealed',False),'immutable candidate accessor')
        object.__setattr__(self,name,value)

    def __delattr__(self,name): raise AttributeError('immutable candidate accessor')

    def __init__(self,raw,owner,forest,contract,tasks,ordinal):
        require(type(raw) is bytes and PREFIX.size<=len(raw)<=MAX_TILE,'immutable bounded frame')
        owner.check(); require(owner.forest is forest,'frame owner forest')
        fields=PREFIX.unpack_from(raw)
        magic,request,source,contract_id,index,count,eq,ine,profile,reserved=fields
        require(magic==MAGIC and request.hex()==forest.request and source.hex()==forest.identity
                and contract_id.hex()==sha(contract) and index==ordinal and count==len(tasks)
                and profile in (0,1) and reserved==0,'frame invocation/header binding')
        require(0<count<=8 and all(t.pair==tasks[0].pair for t in tasks),'one pair tile')
        require(eq==len(forest.rows(tasks[0].pair,'eq')) and ine==len(forest.rows(tasks[0].pair,'ine')),
                'complete frame dual dimensions')
        self._width=8 if profile==0 else RATIONAL; self._eq=eq; self._ine=ine
        self._stride=RECORD.size+2*RATIONAL+2*(eq+ine)*self._width
        require(len(raw)==PREFIX.size+count*self._stride,'exact frame extent')
        self.raw=raw; self.owner=owner; self.tasks=tasks; self.ordinal=ordinal; self.profile=profile
        for k,task in enumerate(tasks):
            owner.work.visit('row_visits'); start=self._start(k)
            slot,query,present,roles,pad=RECORD.unpack_from(raw,start)
            require(slot.hex()==task.slot and query.hex()==task.query_id and present in (0,1)
                    and roles==(3 if present else 0) and pad==bytes(6),'frame slot/query/explicit roles')
            if not present:
                require(not any(raw[start+RECORD.size:start+self._stride]),'missing frame canonical padding')
        self._sealed=True

    def _start(self,k):
        require(type(k) is int and 0<=k<len(self.tasks),'frame row index')
        return PREFIX.size+k*self._stride

    def present(self,k): return bool(RECORD.unpack_from(self.raw,self._start(k))[2])

    def candidate(self,k,role):
        require(role in ('proposed','zero'),'candidate role')
        offset=self._start(k)+RECORD.size+(0 if role=='proposed' else RATIONAL)
        decode_scalar(self.raw[offset:offset+RATIONAL],POLICY['stored_bits'])  # advisory only

    def dimensions(self,k,role,key,count):
        require(count==(self._eq if key=='equality' else self._ine),'frame full roster')

    def weight(self,k,role,key,i):
        count=self._eq if key=='equality' else self._ine
        require(type(i) is int and 0<=i<count,'frame coefficient index')
        j=(0 if role=='proposed' else self._eq+self._ine)+(0 if key=='equality' else self._eq)+i
        offset=self._start(k)+RECORD.size+2*RATIONAL+j*self._width
        raw=self.raw[offset:offset+self._width]
        return _native(raw) if self.profile==0 else decode_scalar(raw,POLICY['stored_bits'])


def encode(owner,forest,contract,tasks,ordinal,records,*,profile=0,remaining=MAX_TOTAL):
    """Preflight exact frame size before fill allocation; no JSON intermediate."""
    owner.check(); require(owner.forest is forest and type(profile) is int and profile in (0,1),'encoder profile/owner')
    require(type(records) is list and 0<len(tasks)<=8 and len(records)==len(tasks),'bounded candidate roster')
    require(type(ordinal) is int and 0<=ordinal<504 and all(t.pair==tasks[0].pair for t in tasks),'tile ordinal/pair')
    eq=len(forest.rows(tasks[0].pair,'eq')); ine=len(forest.rows(tasks[0].pair,'ine'))
    width=8 if profile==0 else RATIONAL; stride=RECORD.size+2*RATIONAL+2*(eq+ine)*width
    size=PREFIX.size+len(tasks)*stride
    require(type(remaining) is int and 0<=remaining<=MAX_TOTAL and size<=min(MAX_TILE,remaining),'typed tile capacity')
    block=owner.allocate(size)
    try:
        block.put(0,PREFIX.pack(MAGIC,bytes.fromhex(forest.request),bytes.fromhex(forest.identity),
                              bytes.fromhex(sha(contract)),ordinal,len(tasks),eq,ine,profile,0))
        for k,(record,task) in enumerate(zip(records,tasks,strict=True)):
            owner.work.visit('row_visits')
            require(type(record) is dict and set(record)=={'slot','proposed','zero'}
                    and record['slot']==task.slot,'encoder slot')
            require((record['proposed'] is None)==(record['zero'] is None),'partial candidate pair')
            present=record['proposed'] is not None; start=PREFIX.size+k*stride
            block.put(start,RECORD.pack(bytes.fromhex(task.slot),bytes.fromhex(task.query_id),int(present),3 if present else 0,bytes(6)))
            if not present: continue
            for ri,role in enumerate(('proposed','zero')):
                c=record[role]
                require(type(c) is dict and set(c)=={'query','lower','equality','inequality'}
                        and c['query']==task.query_id,'encoder candidate binding')
                block.put(start+RECORD.size+ri*RATIONAL,encode_scalar(rational(c['lower']),POLICY['stored_bits']))
                for key,count,base in (('equality',eq,0),('inequality',ine,eq)):
                    require(type(c[key]) is list and len(c[key])==count,'encoder full dual roster')
                    for i,value in enumerate(c[key]):
                        owner.work.visit('coefficient_visits'); q=rational(value)
                        if profile==0:
                            x=float(q); require(math.isfinite(x) and F(x)==q,'native candidate losslessness')
                            raw=struct.pack('<d',x)
                        else: raw=encode_scalar(q,POLICY['stored_bits'])
                        offset=start+RECORD.size+2*RATIONAL+(ri*(eq+ine)+base+i)*width
                        block.put(offset,raw)
        raw=block.seal(); owner.work.tick(); return block,raw
    except BaseException: block.release(); raise


class Driver:
    """Single-owner execution boundary. A failure cannot be retried in-place."""
    def __init__(self,forest,doc,*,expected,work,width=8,capacity=POLICY['total_bytes']):
        self.started=time.monotonic()
        from scoped_source.scaled_endpoints import Session
        self.workspace=Workspace(forest,work,capacity=capacity); self.owner=self.workspace.owner
        self.contract=expected; self.forest=forest; self.work=work; self.closed=False
        self.pending=None; self.ordinal=0; self.receipts=[]; self.total_bytes=0
        self.costs=dict(init_seconds=0.,prepare_seconds=0.,encode_seconds=0.,receive_seconds=0.,
                        publish_seconds=0.,check_seconds=0.,terminal_seconds=0.,release_seconds=0.,
                        publication_attempted_bytes=0,acknowledged_bytes=0)
        try: self.session=Session(forest,doc,expected=expected,work=work,width=width,retain_raw=False,workspace=self.workspace)
        except BaseException: self.workspace.close(); raise
        self.costs['init_seconds']=time.monotonic()-self.started

    def abort(self):
        self.closed=True; self.pending=None; self.receipts.clear()
        self.session.abort(); self.workspace.close()

    def prepare(self):
        start=time.monotonic()
        try:
            require(not self.closed and self.pending is None,'one live invocation tile')
            tasks,roots=self.session.prepare(self.ordinal); self.pending=(tasks,object())
            self.work.tick()
            return tasks,roots,self.pending[1]
        except BaseException: self.abort(); raise
        finally: self.costs['prepare_seconds']+=time.monotonic()-start

    def submit(self,token,records,publish,*,profile=0):
        block=None
        try:
            require(not self.closed and self.pending is not None and token is self.pending[1], 'owner tile token')
            tasks=self.pending[0]; start=time.monotonic()
            try: block,raw=encode(self.owner,self.forest,self.contract,tasks,self.ordinal,records,
                                 profile=profile,remaining=MAX_TOTAL-self.total_bytes)
            finally: self.costs['encode_seconds']+=time.monotonic()-start
            return self._consume_raw(tasks,raw,publish)
        except BaseException: self.abort(); raise
        finally:
            self._release(block)

    def receive(self,token,raw,publish):
        """Portable byte intake. Caller bytes are copied into the owned arena.

        External retained copies are not part of the arena/RSS claim. Partial
        frames are errors, not missing mathematical candidates.
        """
        block=None; start=time.monotonic()
        try:
            require(not self.closed and self.pending is not None and token is self.pending[1],'owner tile token')
            self.owner.check()
            require(type(raw) is bytes and PREFIX.size<=len(raw)<=min(MAX_TILE,MAX_TOTAL-self.total_bytes),'bounded immutable intake')
            block=self.owner.allocate(len(raw)); block.put(0,raw); owned=block.seal()
            self.costs['receive_seconds']+=time.monotonic()-start; start=None
            return self._consume_raw(self.pending[0],owned,publish)
        except BaseException: self.abort(); raise
        finally:
            if start is not None: self.costs['receive_seconds']+=time.monotonic()-start
            self._release(block)

    def _release(self,block):
        start=time.monotonic()
        try:
            if block is not None: block.release()
            if not self.closed: self.work.tick()
        except BaseException: self.abort(); raise
        finally: self.costs['release_seconds']+=time.monotonic()-start

    def _consume_raw(self,tasks,raw,publish):
        self.total_bytes+=len(raw); require(self.total_bytes<=MAX_TOTAL,'cumulative typed evidence ceiling')
        start=time.monotonic()
        try:
            self.work.tick(); h=digest(raw); self.costs['publication_attempted_bytes']+=len(raw)
            result=publish(self.ordinal,raw)
            require(result==(h,len(raw)),'immutable publication acknowledgment'); self.work.tick()
            self.costs['acknowledged_bytes']+=len(raw)
        finally: self.costs['publish_seconds']+=time.monotonic()-start
        start=time.monotonic()
        try:
            tile=Tile(raw,self.owner,self.forest,self.contract,tasks,self.ordinal)
            self.owner._tile=tile
            issued=self.session.accept_direct(tile); self.work.tick()
        finally: self.costs['check_seconds']+=time.monotonic()-start
        self.receipts.extend(issued); self.owner._tile=None; self.pending=None; self.ordinal+=1
        return issued

    def finish(self):
        start=time.monotonic()
        try:
            require(not self.closed and self.pending is None,'complete invocation terminal')
            result=self.session.finish(self.receipts); self.work.tick()
            self.workspace.close(); self.work.tick()  # cleanup belongs to the same budget
            self.closed=True; return result
        except BaseException: self.abort(); raise
        finally: self.costs['terminal_seconds']+=time.monotonic()-start
