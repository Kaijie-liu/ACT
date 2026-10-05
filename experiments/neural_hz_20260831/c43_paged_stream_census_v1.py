"""Same nonexecuting stream lexer; exact exported integer statistics in bounded pages."""
from collections import Counter
import pickletools
from experiments.neural_hz_20260831.c42_pickle_stream_census_v1 import Reader,BULK,INTEGERS
from experiments.neural_hz_20260831.c43_paged_integer_summary_v1 import Summary


def census(file,*,expected_sha256,expected_bytes,pool,observe=None,enabled=False):
    if not enabled:return None
    if (type(expected_sha256) is not str or len(expected_sha256)!=64
            or type(expected_bytes) is not int or expected_bytes<=0 or file.tell()!=0):
        raise ValueError('independent complete artifact identity required')
    pool.charge('stream_census_header',512)
    r=Reader(file);counts=Counter();payloads=Counter();integers=Summary(pool);frame_end=None;frames=0;frame_bytes=0
    protocols=[];maximum_argument=0;op_count=0
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
        if name in INTEGERS and type(argument) is int:integers.add(argument)
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
                byte_offset=r.offset,integer_occurrences=integers.occurrences,distinct_integer_values=integers.distinct,
                charged_work=pool.used,formal_gain=0))
        if name=='STOP':break
    if (r.offset!=expected_bytes or (frame_end is not None and r.offset!=frame_end)
            or file.read(1) or r.h.hexdigest()!=expected_sha256):
        raise ValueError('incomplete/changed pickle image or trailing payload')
    summary=integers.report()
    return dict(schema='c43_complete_paged_nonexecuting_pickle_census_v1',complete=True,
        artifact_sha256=r.h.hexdigest(),artifact_bytes=r.offset,protocols=protocols,
        opcode_count=sum(counts.values()),opcode_counts=dict(sorted(counts.items())),
        opaque_payload_bytes_by_opcode=dict(sorted(payloads.items())),opaque_payload_bytes=sum(payloads.values()),
        largest_opaque_argument_bytes=maximum_argument,largest_stream_read_bytes=r.largest_read,
        frames=frames,total_framed_bytes=frame_bytes,**summary,
        integer_object_identity_or_safe_interning_not_proved=True,opaque_inner_objects_not_inspected=True,
        pickle_stack_semantics_not_proved=True,unpickler_or_reducer_executed=False,
        new_HZ_constructed=False,solver_executed=False,formal_gain=0,
        diagnostic_work=pool.used,work_parts=dict(pool.parts))
