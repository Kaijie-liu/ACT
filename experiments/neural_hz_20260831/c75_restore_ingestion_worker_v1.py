"""Bounded observation of unchanged complete C74 native decode/root traversal."""
from dataclasses import asdict
import faulthandler
import json
from pathlib import Path
import resource
import signal
import sys
import time
import tracemalloc
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import rss_bytes
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c75_restore_ingestion_20260913_v1'
SOURCE=EXP/'results/c74_native_binding_20260913_v2/relu78.pickle'
SHA='5586296e7fccef66956dff042d29ca93b999d3aafc1a0285cea17863b089743f'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);start=time.monotonic();held={}
    result=dict(completed=False,formal_gain=0,diagnostic_only=True,new_native_or_solver_executed=False)
    def timeout(signum,frame):raise TimeoutError('registered45s soft diagnostic deadline')
    signal.signal(signal.SIGALRM,timeout)
    with (RUN/'events.jsonl').open('x') as log,(RUN/'stacks.log').open('x') as stacks:
        def emit(name,**values):
            current,peak=tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0,0)
            event=dict(event=name,elapsed_s=time.monotonic()-start,rss_bytes=rss_bytes(),
                lifetime_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                traced_current_bytes=current,traced_peak_bytes=peak,
                tracer_metadata_bytes=tracemalloc.get_tracemalloc_memory() if tracemalloc.is_tracing() else 0,**values)
            log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(event),flush=True)
            result['last_completed_boundary']=name
        def check():
            entry=rss_bytes()
            emit('before_complete_authenticated_decode',entry_rss_bytes=entry)
            with SOURCE.open('rb') as handle:
                saved,decoder=load(handle,expected_sha256=SHA,pool=pool,enabled=True)
            held['complete_checkpoint']=saved
            result['decoder']=decoder
            emit('after_complete_authenticated_decode',decoder=decoder,root_keys=len(saved))
            if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-entry>1024**3
                    or sum((tracemalloc.get_traced_memory()[1],tracemalloc.get_tracemalloc_memory()))>1024**3):
                raise MemoryError('complete checkpoint decode alone exceeds original1GiB gate')
            pool.charge('c75_complete_original_root_fingerprint_bound',64_000_000)
            emit('before_complete_root_collect')
            roots=collect(SimpleNamespace(),{'complete_checkpoint':saved})
            held['complete_registered_roots']=roots
            emit('after_complete_root_collect',numeric_roots=len(roots.numeric),
                python_shallow_bytes=roots.python_shallow_bytes,unique_objects=roots.unique_objects)
            emit('before_complete_numeric_owner_layout')
            layout=roots.measure()
            result['numeric_layout']=asdict(layout)
            emit('after_complete_numeric_owner_layout',layout=asdict(layout))
            if layout.resident_entries>64_000_000:raise MemoryError('complete numeric entry cap')
            return dict(complete_input_retained=True,root_fingerprint=roots.fingerprint)
        try:
            signal.setitimer(signal.ITIMER_REAL,45)
            faulthandler.dump_traceback_later(10,repeat=True,file=stacks)
            data,_=measured(check,observe=lambda s:result.update(measurement=s))
            result.update(completed=True,data=data)
        except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);faulthandler.cancel_dump_traceback_later()
            result.update(wall_s=time.monotonic()-start,diagnostic_work=pool.used,work_parts=pool.parts,
                complete_decoded_input_still_retained='complete_checkpoint' in held)
            _atomic_exclusive_json(RUN/'result.json',result);print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
