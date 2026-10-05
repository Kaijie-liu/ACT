"""Source-hash-bound opcode census, NEVER an unpickler or HZ admission proof.

Opaque byte/string payloads are hashed through a64KiB reusable buffer. Only
small opcode arguments and bounded exact integer-frequency metadata persist.
No reducer/global/object constructor is executed. This does not prove pickle
stack semantics or reveal objects encoded inside an opaque payload.
"""
from collections import Counter
import hashlib
import pickletools
import struct

CHUNK=65536
BULK={'BINBYTES':(4,False),'SHORT_BINBYTES':(1,False),'BINBYTES8':(8,False),
    'BYTEARRAY8':(8,False),'BINSTRING':(4,True),'SHORT_BINSTRING':(1,False),
    'BINUNICODE':(4,False),'SHORT_BINUNICODE':(1,False),'BINUNICODE8':(8,False)}
INTEGERS={'INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4'}


class Reader:
    def __init__(self,file):
        self.file=file;self.offset=0;self.h=hashlib.sha256();self.buffer=bytearray(CHUNK)
        self.largest_read=0
    def read(self,n):
        if type(n) is not int or not 0<=n<=CHUNK:raise ValueError('unbounded metadata read')
        data=self.file.read(n)
        if len(data)!=n:raise ValueError('truncated pickle argument')
        self.offset+=n;self.h.update(data);self.largest_read=max(self.largest_read,n);return data
    def readline(self):
        data=self.file.readline(CHUNK+1)
        if len(data)>CHUNK or not data.endswith(b'\n'):raise ValueError('unbounded or truncated pickle line')
        self.offset+=len(data);self.h.update(data);self.largest_read=max(self.largest_read,len(data));return data
    def skip(self,n):
        if type(n) is not int or n<0:raise ValueError('negative opaque payload')
        view=memoryview(self.buffer)
        while n:
            count=min(n,CHUNK);got=self.file.readinto(view[:count])
            if got!=count:raise ValueError('truncated opaque payload')
            self.h.update(view[:count]);self.offset+=count;n-=count
            self.largest_read=max(self.largest_read,count)


def census(file,*,expected_sha256,expected_bytes,pool,observe=None,enabled=False):
    if not enabled:return None
    if (type(expected_sha256) is not str or len(expected_sha256)!=64
            or type(expected_bytes) is not int or expected_bytes<=0 or file.tell()!=0):
        raise ValueError('independent complete artifact identity required')
    pool.charge('stream_census_header',512)
    r=Reader(file);counts=Counter();payloads=Counter();integers={};frame_end=None;frames=0;frame_bytes=0
    integer_occurrences=0;protocols=[];maximum_argument=0;op_count=0
    while True:
        if frame_end is not None and r.offset==frame_end:frame_end=None
        pool.charge('stream_census_opcode',8)
        op=pickletools.code2op.get(chr(r.read(1)[0]))
        if op is None:raise ValueError('unknown pickle opcode')
        name=op.name;counts[name]+=1;op_count+=1;argument=None
        if name in BULK:
            width,signed=BULK[name];length=int.from_bytes(r.read(width),'little',signed=signed)
            if length<0 or r.offset+length>expected_bytes:raise ValueError('opaque payload outside artifact')
            maximum_argument=max(maximum_argument,length);payloads[name]+=length;r.skip(length)
        elif name in ('LONG1','LONG4'):
            length=int.from_bytes(r.read(1 if name=='LONG1' else 4),'little',signed=name=='LONG4')
            if not 0<=length<=128:raise ValueError('integer argument outside registered non-extreme domain')
            pool.charge('stream_census_long_bytes',length)
            argument=int.from_bytes(r.read(length),'little',signed=True)
        elif op.arg is not None:argument=op.arg.reader(r)
        if name in INTEGERS and type(argument) is int:
            pool.charge('stream_census_integer_frequency',12)
            if argument not in integers:
                pool.charge('stream_census_new_integer_key',16)
                if len(integers)>=1_000_000:raise MemoryError('exact integer metadata cap')
                integers[argument]=0
            integers[argument]+=1;integer_occurrences+=1
        if name=='PROTO':
            if not 0<=argument<=5:raise ValueError('unregistered pickle protocol')
            protocols.append(argument)
        if name=='FRAME':
            if frame_end is not None:raise ValueError('nested/incomplete pickle frame')
            frame_end=r.offset+argument;frames+=1;frame_bytes+=argument
            if frame_end>expected_bytes:raise ValueError('frame outside complete artifact')
        elif frame_end is not None and r.offset>frame_end:raise ValueError('opcode crosses pickle frame')
        if observe and op_count%262144==0:
            observe(dict(event='nonexecuting_opcode_progress',complete=False,opcodes=op_count,
                byte_offset=r.offset,integer_occurrences=integer_occurrences,distinct_integer_values=len(integers),
                charged_work=pool.used,formal_gain=0))
        if name=='STOP':break
    if (r.offset!=expected_bytes or (frame_end is not None and r.offset!=frame_end)
            or file.read(1) or r.h.hexdigest()!=expected_sha256):
        raise ValueError('incomplete/changed pickle image or trailing payload')
    pool.charge('stream_census_complete_integer_summary',8*len(integers)+256)
    small={v:n for v,n in integers.items() if -5<=v<=256}
    outside_distinct=len(integers)-len(small);outside_occurrences=integer_occurrences-sum(small.values())
    bins=Counter()
    for value,count in integers.items():
        if -5<=value<=256:continue
        label='1' if count==1 else '2' if count==2 else '3' if count==3 else '4-7' if count<8 else '8-15' if count<16 else '16+'
        bins[label]+=1
    return dict(schema='c42_complete_nonexecuting_pickle_opcode_census_v1',complete=True,
        artifact_sha256=r.h.hexdigest(),artifact_bytes=r.offset,protocols=protocols,
        opcode_count=sum(counts.values()),opcode_counts=dict(sorted(counts.items())),
        opaque_payload_bytes_by_opcode=dict(sorted(payloads.items())),opaque_payload_bytes=sum(payloads.values()),
        largest_opaque_argument_bytes=maximum_argument,largest_stream_read_bytes=r.largest_read,
        frames=frames,total_framed_bytes=frame_bytes,integer_literal_occurrences=integer_occurrences,
        distinct_integer_literal_values=len(integers),small_integer_range=[-5,256],
        small_range_integer_occurrences=sum(small.values()),small_range_distinct_values=len(small),
        outside_small_range_integer_occurrences=outside_occurrences,outside_small_range_distinct_values=outside_distinct,
        outside_small_range_repeated_occurrences=outside_occurrences-outside_distinct,
        outside_small_range_value_multiplicity_histogram=dict(sorted(bins.items())),
        integer_object_identity_or_safe_interning_not_proved=True,opaque_inner_objects_not_inspected=True,
        pickle_stack_semantics_not_proved=True,unpickler_or_reducer_executed=False,
        new_HZ_constructed=False,solver_executed=False,formal_gain=0,
        diagnostic_work=pool.used,work_parts=dict(pool.parts))
