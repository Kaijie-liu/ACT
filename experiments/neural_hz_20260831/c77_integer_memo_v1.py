"""Exact protocol4/5 integer sharing; no constructors or mutable deduplication.

Original memo entries keep explicit original indices. Only extra immutable
built-in int aliases are introduced. Opaque byte payloads pass through intact.
"""
from collections import Counter
import hashlib
import pickletools
import struct
import numpy as np

LIMIT=2**24
FRAME_SIZE=65536
BLOBS={b'B':4,b'C':1,b'\x8e':8,b'\x96':8,b'X':4,b'\x8c':1,b'\x8d':8}


class Framer:
    def __init__(self, output):
        self.output=output;self.pending=bytearray();self.hash=hashlib.sha256();self.bytes=0
    def raw(self, value):
        self.hash.update(value);self.bytes+=len(value)
        if self.output is not None:self.output.write(value)
    def flush(self):
        if self.pending:
            self.raw(b'\x95'+struct.pack('<Q',len(self.pending)))
            self.raw(self.pending);self.pending.clear()
    def atom(self, value):
        if len(value)>FRAME_SIZE:raise ValueError('large payload needs streaming boundary')
        if len(self.pending)+len(value)>FRAME_SIZE:self.flush()
        self.pending.extend(value)
    def integer_rows(self, rows):
        for at in range(0,len(rows),13100):
            self.atom(memoryview(rows[at:at+13100]).cast('B'))
    def payload(self, source, header, size):
        if size+len(header)<=FRAME_SIZE:
            self.atom(header+exact(source,size));return
        self.flush();self.raw(header)
        while size:
            chunk=exact(source,min(size,1024**2));self.raw(chunk);size-=len(chunk)


def exact(source,n):
    value=source.read(n)
    if len(value)!=n:raise ValueError('incomplete pickle payload')
    return value


def rewrite(source,output,*,original_memo_count,pool,enabled=False,observe=None):
    if not enabled:return None
    if (type(original_memo_count) is not int or not 0<=original_memo_count<2**31
            or source.tell()!=0 or not hasattr(source,'peek')):
        raise ValueError('complete source memo population and buffered initial stream required')
    pool.charge('c77_exact_uint32_scalar_table',LIMIT)
    memo=np.zeros(LIMIT,np.uint32);next_memo=original_memo_count;old_memo=0
    writer=Framer(output);counts=Counter();batches=singletons=hits=news=payload_bytes=0
    prefix=exact(source,2)
    if prefix not in (b'\x80\x04',b'\x80\x05'):raise ValueError('protocol4/5 source required')
    writer.raw(prefix);counts['PROTO']=1;pool.charge('c77_other_opcode',9)
    dtype5=np.dtype([('opcode','u1'),('value','<i4')])
    dtype3=np.dtype([('opcode','u1'),('value','<u2')])

    def scalar(value):
        nonlocal next_memo,hits,news,singletons
        pool.charge('c77_scalar_integer',32);singletons+=1
        idx=int(memo[value])
        if idx:
            writer.atom(b'j'+struct.pack('<I',idx));hits+=1
        else:
            if next_memo>=2**32-1:raise ValueError('integer memo index overflow')
            # Zero is the empty-table sentinel. Reserve index0 if the original
            # stream has an empty memo; no original GET may reference it then.
            if next_memo==0:next_memo=1
            idx=next_memo;next_memo+=1;memo[value]=idx;news+=1
            writer.atom(b'J'+struct.pack('<i',value)+b'r'+struct.pack('<I',idx))

    while True:
        start=source.tell();code=exact(source,1)
        op=pickletools.code2op.get(code.decode('latin1'))
        if op is None:raise ValueError('unsupported pickle opcode')
        name=op.name
        if code in (b'J',b'M'):
            source.seek(-1,1);peek=source.peek(FRAME_SIZE)
            width=5 if code==b'J' else 3
            rows=np.frombuffer(peek,dtype=dtype5 if width==5 else dtype3,count=len(peek)//width)
            if len(rows):
                v=rows['value'];eligible=(rows['opcode']==code[0])&(v>=257)&(v<LIMIT)
                bad=np.flatnonzero(~eligible)
                count=int(bad[0]) if len(bad) else len(v)
                if count>1:
                    breaks=np.flatnonzero(v[1:count]<=v[:count-1])
                    if len(breaks):count=int(breaks[0])+1
                if count>1:
                    pool.charge('c77_integer_batch_header',32)
                    pool.charge('c77_complete_batched_integer',16*count)
                    values=v[:count].astype(np.int32,copy=False)
                    ids=memo[values];missing=ids==0;first=np.flatnonzero(missing);n=len(first)
                    if next_memo==0:next_memo=1
                    if next_memo+n>=2**32:raise ValueError('integer memo index overflow')
                    ids[first]=np.arange(next_memo,next_memo+n,dtype=np.uint32)
                    memo[values[first]]=ids[first];next_memo+=n
                    positions=np.arange(count)+np.cumsum(missing)-missing
                    encoded=np.empty((count+n,5),np.uint8)
                    encoded[positions,0]=np.where(missing,ord('J'),ord('j'))
                    payload=np.where(missing,values,ids).astype('<u4')
                    encoded[positions,1:]=payload.view(np.uint8).reshape(-1,4)
                    encoded[positions[first]+1,0]=ord('r')
                    encoded[positions[first]+1,1:]=ids[first].astype('<u4').view(np.uint8).reshape(-1,4)
                    writer.integer_rows(encoded)
                    source.seek(width*count,1);counts[name]+=count;batches+=1
                    hits+=count-n;news+=n
                    if observe is not None and batches%1024==0:
                        observe(dict(event='integer_prefixes',batches=batches,source_offset=source.tell(),
                                     new_integer_memos=news,reused_integer_memos=hits,work=pool.used))
                    continue
            raw=exact(source,width);value=int.from_bytes(raw[1:],'little',signed=width==5)
            counts[name]+=1
            if 257<=value<LIMIT:scalar(value)
            else:pool.charge('c77_other_opcode',8+width-1);writer.atom(raw)
            continue
        counts[name]+=1
        if code in BLOBS:
            head=exact(source,BLOBS[code]);n=int.from_bytes(head,'little')
            pool.charge('c77_other_opcode',8+len(head))
            pool.charge('c77_unchanged_payload_read_write',2*((n+7)//8))
            writer.payload(source,code+head,n);payload_bytes+=n
        elif code==b'\x95':
            exact(source,8);pool.charge('c77_other_opcode',16)
        elif code==b'\x94':
            pool.charge('c77_other_opcode',8)
            if old_memo>=original_memo_count:raise ValueError('original memo population exceeded')
            writer.atom(b'r'+struct.pack('<I',old_memo));old_memo+=1
        elif code in (b'q',b'r',b'p'):
            raise ValueError('original explicit PUT is outside the registered memo-only source')
        else:
            arg=op.arg.reader(source) if op.arg is not None else None
            end=source.tell();source.seek(start);raw=exact(source,end-start)
            pool.charge('c77_other_opcode',8+len(raw)-1)
            if code in (b'h',b'j',b'g') and (type(arg) is not int or not 0<=arg<old_memo):
                raise ValueError('original GET is not an already defined original memo')
            if code==b'\x80':raise ValueError('second protocol header')
            writer.atom(raw)
        if code==b'.':break
    if old_memo!=original_memo_count or source.read(1):raise ValueError('incomplete memo population or trailing bytes')
    writer.flush()
    return dict(schema='c77_complete_integer_memo_wire_transform_v1',
        source_opcode_counts=dict(counts),original_memo_count=old_memo,
        new_integer_memos=news,reused_integer_memos=hits,integer_batches=batches,integer_singletons=singletons,
        scalar_table_entries=LIMIT,scalar_table_bytes=memo.nbytes,
        unchanged_byte_and_unicode_payload_bytes=payload_bytes,
        output_sha256=writer.hash.hexdigest(),output_bytes=writer.bytes,
        original_mutable_memo_indices_preserved=True,only_extra_immutable_int_aliases=True,
        all_noninteger_payloads_preserved=True,constructors_executed=False,
        complete_restore_claim=False,formal_gain=0)
