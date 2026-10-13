"""One-tile lossless candidate intake and portable all-obligations replay.

No solver/producer imports. Both proposed and zero values go through the same
V3 mathematical checker. Physical and logical identities are distinct.
"""
import hashlib
import struct

from scoped_source.hz_request_store import compact,decode,digest,identity,rational,require,sha
from scoped_source.factored_io import read
from scoped_source.wire_forest import CheckedWireForest,float_exact,unpack_float
from scoped_source.scaled_endpoints import Session,contract,PROOF

MAGIC=b'HZWT0001'
PREFIX=struct.Struct('<8sI')
TILE_CAP=4*2**20
PROOF_CAP=16*2**20
MANIFEST='WIRE_ENDPOINT_MANIFEST_V1'


def binding(forest,tasks):
    require(tasks and all(t.pair==tasks[0].pair for t in tasks),'one pair per tile')
    return dict(source=forest.source,forest=forest.identity,request=forest.request,
                factors=forest.factor_identity,contract=identity(contract(forest)),
                rows=identity({k:forest.rows(tasks[0].pair,k) for k in ('eq','ine')}))


def encode_tile(records,forest,tasks,work):
    work.tick()
    require(type(records) is list and len(records)==len(tasks)<=8,'tile task roster')
    headers=[]; payload=bytearray(); logical_size=2
    counts={k:len(forest.rows(tasks[0].pair,k)) for k in ('eq','ine')}
    for record,task in zip(records,tasks,strict=True):
        require(type(record) is dict and set(record)=={'slot','proposed','zero'} and record['slot']==task.slot,'encode slot')
        require((record['proposed'] is None)==(record['zero'] is None),'both candidate roles')
        if record['proposed'] is None:
            logical_size+=len(compact(record))+1
            headers.append(dict(slot=task.slot,missing=True)); continue
        item=dict(slot=task.slot,query=task.query_id,missing=False,lower={},zero='EXPLICIT_ZERO_V1')
        for role in ('proposed','zero'):
            c=record[role]; require(type(c) is dict and set(c)=={'query','equality','inequality','lower'} and c['query']==task.query_id,'candidate binding')
            rational(c['lower']); item['lower'][role]=c['lower']
            for kind,key in (('eq','equality'),('ine','inequality')):
                require(type(c[key]) is list and len(c[key])==counts[kind],'complete dual vector')
                for value in c[key]:
                    work.visit('coefficient_visits'); q=rational(value)
                    logical_size+=len(value)+3
                    require(logical_size<=TILE_CAP,'logical candidate element capacity')
                    if role=='zero': require(q==0,'explicit zero role')
                    else: payload.extend(struct.pack('<d',float_exact(q)))
        logical_size+=len(compact({k:v for k,v in item.items() if k!='lower'}))+sum(len(v) for v in item['lower'].values())
        require(logical_size<=TILE_CAP,'logical candidate header capacity')
        headers.append(item)
    raw=compact(records); require(len(raw)<=TILE_CAP,'logical tile cap')
    header=compact(dict(schema='WIRE_CANDIDATE_TILE_V1',binding=binding(forest,tasks),counts=counts,records=headers,logical=digest(raw)))
    require(len(header)<=8192,'candidate header cap')
    result=PREFIX.pack(MAGIC,len(header))+header+payload
    work.tick(); require(len(result)<=TILE_CAP,'physical tile cap'); return result


def decode_tile(raw,forest,tasks,work):
    work.tick(); require(type(raw) is bytes and PREFIX.size<=len(raw)<=TILE_CAP,'wire tile bytes')
    magic,size=PREFIX.unpack_from(raw); require(magic==MAGIC and 0<size<=8192 and PREFIX.size+size<=len(raw),'wire tile prefix')
    hr=raw[PREFIX.size:PREFIX.size+size]; h=decode(hr)
    require(type(h) is dict and compact(h)==hr and set(h)=={'schema','binding','counts','records','logical'} and
            h['schema']=='WIRE_CANDIDATE_TILE_V1' and h['binding']==binding(forest,tasks),'wire candidate identity')
    counts={k:len(forest.rows(tasks[0].pair,k)) for k in ('eq','ine')}
    require(h['counts']==counts and type(h['records']) is list and len(h['records'])==len(tasks),'wire candidate roster')
    records=[]; offset=PREFIX.size+size; logical_size=2
    for item,task in zip(h['records'],tasks,strict=True):
        work.tick(); require(type(item) is dict and item.get('slot')==task.slot and type(item.get('missing')) is bool,'wire slot/order')
        if item['missing']:
            require(set(item)=={'slot','missing'},'missing candidate fields')
            record=dict(slot=task.slot,proposed=None,zero=None)
        else:
            require(set(item)=={'slot','query','missing','lower','zero'} and item['query']==task.query_id and
                    item['zero']=='EXPLICIT_ZERO_V1' and type(item['lower']) is dict and
                    set(item['lower'])=={'proposed','zero'},'wire both role descriptors')
            # Minimum old JSON size is charged BEFORE expanding zero descriptors.
            require(8*(counts['eq']+counts['ine'])+logical_size<TILE_CAP,'zero logical expansion cap')
            proposed=dict(query=task.query_id,lower=item['lower']['proposed'])
            zero=dict(query=task.query_id,lower=item['lower']['zero'])
            rational(proposed['lower']); rational(zero['lower'])
            for kind,key in (('eq','equality'),('ine','inequality')):
                count=counts[kind]; require(offset+8*count<=len(raw),'truncated proposed vector')
                vals=[]
                for j in range(count):
                    work.visit('coefficient_visits'); value=str(unpack_float(struct.unpack_from('<d',raw,offset)[0])); offset+=8
                    logical_size+=len(value)+7
                    require(logical_size<=TILE_CAP,'logical candidate expansion cap'); vals.append(value)
                proposed[key]=vals; zero[key]=['0']*count
            record=dict(slot=task.slot,proposed=proposed,zero=zero)
        # Running allowance above bounds vectors before allocation. Exact old
        # bytes below remain the authoritative limit, not this conservative sum.
        require(logical_size<=TILE_CAP,'logical decoded tile cap'); records.append(record)
    require(offset==len(raw),'extra wire payload'); logical=compact(records)
    require(len(logical)<=TILE_CAP and digest(logical)==sha(h['logical']),'logical candidate identity')
    work.tick(); return logical


class ProofHash:
    """Incremental canonical old proof hash/cap; retain no tile bodies."""
    def __init__(self,contract_hash,width,work):
        require(type(width) is int and 1<=width<=8,'proof width')
        self.work=work; self.h=hashlib.sha256(); self.size=0; self.count=0; self.width=width
        self.add(b'{"contract":'+compact(sha(contract_hash))+b',"schema":'+compact(PROOF)+b',"tiles":[')

    def add(self,raw):
        self.work.tick(); self.size+=len(raw); require(self.size<=PROOF_CAP,'logical proof capacity'); self.h.update(raw)

    def tile(self,raw):
        if self.count: self.add(b',')
        self.add(raw); self.count+=1

    def finish(self):
        self.add(b'],"width":'+compact(self.width)+b'}'); return self.h.hexdigest(),self.size


def check_manifest(root,manifest,*,expected,work):
    """Complete replay from only package-local files. Stored status is not used."""
    work.tick(); require(type(manifest) is bytes,'bounded durable manifest bytes required'); raw=manifest
    require(len(raw)<=2**20 and digest(raw)==sha(expected),'wire manifest identity/cap')
    m=decode(raw)
    require(compact(m)==raw and type(m) is dict and set(m)=={'schema','source','forest','contract','width','tiles','logical_proof','logical_bytes'} and
            m['schema']==MANIFEST,'wire manifest fields')
    def member(ref,cap):
        require(type(ref) is dict and set(ref)=={'file','bytes','sha256'} and type(ref['bytes']) is int and 0<ref['bytes']<=cap,'manifest member')
        work.tick(); value=read(root,ref['file'],cap,sha(ref['sha256'])); work.tick()
        require(len(value)==ref['bytes'],'manifest member length'); return value
    require(type(m['source']) is dict and set(m['source'])=={'directory','sha256'} and m['source']['directory']=='source','package-local source')
    from pathlib import Path
    from scoped_source.hz_request_store import POLICY
    f=CheckedWireForest(member(m['forest'],POLICY['total_bytes']),Path(root)/'source',m['source']['sha256'],
                        expected=m['forest']['sha256'],work=work)
    c=decode(member(m['contract'],8192)); require(c==contract(f),'portable full contract')
    s=Session(f,c,expected=identity(c),work=work,width=m['width'],retain_raw=False)
    require(type(m['tiles']) is list and len(m['tiles'])==s.tile_count and
            len({r['file'] for r in m['tiles']})==s.tile_count,'all unique tile references')
    h=ProofHash(identity(c),m['width'],work); receipts=[]
    for i,ref in enumerate(m['tiles']):
        tasks,_=s.prepare(i); logical=decode_tile(member(ref,TILE_CAP),f,tasks,work)
        h.tile(logical); receipts.extend(s.accept(logical))
        del logical
    ph,size=h.finish()
    require(ph==sha(m['logical_proof']) and type(m['logical_bytes']) is int and size==m['logical_bytes'],'complete logical proof identity')
    result=s.finish(receipts); work.tick(); return result
