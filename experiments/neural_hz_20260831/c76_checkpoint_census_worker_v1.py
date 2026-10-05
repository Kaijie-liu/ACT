"""Read every pickle opcode without executing any archive constructor."""
from collections import Counter
import heapq
import json
from pathlib import Path
import pickletools
import resource
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c76_checkpoint_census_20260913_v1'
SOURCE=EXP/'results/c74_native_binding_20260913_v2/relu78.pickle'
SHA='5586296e7fccef66956dff042d29ca93b999d3aafc1a0285cea17863b089743f'
INTEGER_OPS={'INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4'}


def census(pool,result,emit):
    if _sha256(SOURCE)!=SHA:raise ValueError('complete original checkpoint changed')
    pool.charge('c76_integer_bitset_initialization',2**21)
    seen=bytearray(2**21);counts=Counter();strings=Counter();blobs=Counter();largest=[]
    integers=Counter();lo=hi=None;unique=total=0;other_strings=0
    try:
        with SOURCE.open('rb') as stream:
            for op,arg,offset in pickletools.genops(stream):
                pool.charge('c76_complete_opcode_census',8)
                name=op.name;counts[name]+=1;total+=1
                if name in INTEGER_OPS and type(arg) is int:
                    pool.charge('c76_integer_population',8)
                    integers['total']+=1
                    lo=arg if lo is None else min(lo,arg);hi=arg if hi is None else max(hi,arg)
                    if -5<=arg<=256:integers['small_cached_range']+=1
                    if 0<=arg<2**24:
                        integers['in_profile_range']+=1
                        at=arg>>3;bit=1<<(arg&7)
                        if not seen[at]&bit:seen[at]|=bit;unique+=1
                    else:integers['outside_profile_range']+=1
                elif type(arg) in (bytes,bytearray):
                    n=len(arg);pool.charge('c76_serialized_byte_payload', (n+7)//8)
                    blobs[name+'_count']+=1;blobs[name+'_bytes']+=n
                    heapq.heappush(largest,(n,int(offset),name))
                    if len(largest)>16:heapq.heappop(largest)
                elif type(arg) is str:
                    if len(arg)<=128 and (arg in strings or len(strings)<8192):strings[arg]+=1
                    else:other_strings+=1
                if total%500_000==0:
                    emit(dict(event='complete_stream_prefix',opcodes=total,byte_offset=int(offset),
                              integer_occurrences=integers['total'],profile_distinct_integers=unique))
            if counts['STOP']!=1 or stream.tell()!=SOURCE.stat().st_size or stream.read(1):
                raise ValueError('not one complete pickle stream')
        if _sha256(SOURCE)!=SHA:raise ValueError('complete original file mutated during read-only census')
        return dict(complete_stream=True,archive_sha256=SHA,file_bytes=SOURCE.stat().st_size,
            complete_file_hash_passes=2,total_opcodes=total,opcodes=dict(counts),
            integer_counts=dict(integers),profile_distinct_integers=unique,integer_min=lo,integer_max=hi,
            profile_range=[0,2**24],integer_bitset_bytes=len(seen),byte_payloads=dict(blobs),
            largest_byte_payloads=sorted(largest,reverse=True),short_strings=dict(strings),
            other_strings=other_strings,constructors_executed=False,complete_restore_claim=False,formal_gain=0)
    finally:
        result['last_prefix']=dict(opcodes=total,integer_counts=dict(integers),
            profile_distinct_integers=unique,opcode_counts=dict(counts),diagnostic_work=pool.used)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    def alarm(signum,frame):raise TimeoutError('registered45s census deadline')
    signal.signal(signal.SIGALRM,alarm)
    pool=WorkPool(256_000_000);started=time.monotonic();result=dict(completed=False,formal_gain=0)
    with (RUN/'events.jsonl').open('x') as log:
        def emit(value):
            log.write(json.dumps(dict(value,worker_elapsed_s=time.monotonic()-started))+'\n');log.flush()
        try:
            signal.setitimer(signal.ITIMER_REAL,45)
            data,stats=measured(lambda:census(pool,result,emit),observe=lambda s:result.update(measurement=s))
            result.update(completed=True,data=data)
        except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            result.update(wall_s=time.monotonic()-started,diagnostic_work=pool.used,work_parts=pool.parts)
            _atomic_exclusive_json(RUN/'result.json',result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('data','last_prefix')}),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
