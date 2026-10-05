"""Complete frozen preflight, exact conversion, or whole native restore."""
from contextlib import nullcontext
from dataclasses import asdict
import faulthandler
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c77_integer_memo_v1 import rewrite
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import rss_bytes
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c74_native_binding_v1 import admit_source,admit_native
from experiments.neural_hz_20260831.c73_outer_query_v1 import GuardedLocalSpliceJournal
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan
from experiments.neural_hz_20260831.c70_native_proof_v1 import verify_inverse
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c77_integer_memo_20260913_v1'
OLD=EXP/'results/c74_native_binding_20260913_v2'
SOURCE=OLD/'relu78.pickle'
SHA='5586296e7fccef66956dff042d29ca93b999d3aafc1a0285cea17863b089743f'
CENSUS=EXP/'results/c76_checkpoint_census_20260913_v1/result.json'
CENSUS_SHA='9eb6ad9a06a188c165c9ce465bd6d5d3e708778f1f6c746bf032de447760d6a4'
SOURCE_PROOF='6e0a339f08fd858ca45683a6e7cdf5a34b48e2012994889897d3290d33f0ea78'
TRANSFER_PROOF='09756b7f918bb911c0312d126100d32e04a0ed8ce59c8ed496a41157aaf9059d'


def transform(stage,pool,result,emit):
    if _sha256(CENSUS)!=CENSUS_SHA:raise ValueError('complete census changed')
    c=json.loads(CENSUS.read_text())
    if not (c['completed'] and c['data']['complete_stream'] and c['data']['archive_sha256']==SHA):
        raise ValueError('complete original census required')
    if _sha256(SOURCE)!=SHA:raise ValueError('complete actual checkpoint changed')
    emit(dict(event='before_complete_wire_transform',source_bytes=SOURCE.stat().st_size))
    manager=(RUN/'checkpoint.pickle').open('xb') if stage=='conversion' else nullcontext(None)
    with SOURCE.open('rb',buffering=65536) as source,manager as output:
        report=rewrite(source,output,original_memo_count=c['data']['opcodes']['MEMOIZE'],
            pool=pool,enabled=True,observe=emit)
    result['wire_report']=report
    if report['source_opcode_counts']!=c['data']['opcodes']:raise ValueError('complete opcode census differs')
    if _sha256(SOURCE)!=SHA:raise ValueError('read-only complete original checkpoint drift')
    if stage=='conversion':
        pre=json.loads((RUN/'preflight_result.json').read_text())
        if (not pre['completed'] or pre['data']['wire_report']!=report
                or pre['diagnostic_work']!=pool.used or pre['work_parts']!=pool.parts
                or _sha256(RUN/'checkpoint.pickle')!=report['output_sha256']
                or (RUN/'checkpoint.pickle').stat().st_size!=report['output_bytes']):
            raise ValueError('complete actual conversion differs from preregistered preflight')
    emit(dict(event='complete_wire_transform_passed',output_bytes=report['output_bytes'],work=pool.used))
    return dict(wire_report=report,original_archive_sha256=SHA,original_archive_bytes=SOURCE.stat().st_size,
        complete_source_hash_passes=2,complete_output_file_hash_passes=int(stage=='conversion'),
        complete_output_stream_hashed=True,full_input_read_only=True,full_CPU_work_256M_claim=False,
        constructors_executed=False,complete_restore_claim=False,formal_gain=0)


def restore(pool,result,held,emit,entry):
    original=json.loads((OLD/'qualification.json').read_text())
    converted=json.loads((RUN/'conversion_result.json').read_text())
    if not original['passed'] or not converted['completed']:raise ValueError('complete original and conversion gates required')
    wire=converted['data']['wire_report'];archive=RUN/'checkpoint.pickle'
    emit(dict(event='before_complete_authenticated_decode'))
    with archive.open('rb') as stream:
        saved,decoder=load(stream,expected_sha256=wire['output_sha256'],pool=pool,enabled=True)
    held['complete_checkpoint']=saved;result['decoder']=decoder
    emit(dict(event='after_complete_authenticated_decode',decoder=decoder,root_keys=len(saved)))
    if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-entry>1024**3
            or tracemalloc.get_traced_memory()[1]+tracemalloc.get_tracemalloc_memory()>1024**3):
        raise MemoryError('complete integer-shared checkpoint decode exceeds original1GiB gate')
    envelope=int(original['whole_live_state']['resident_entries'])+decoder['copied_numeric_entries']
    pool.charge('c77_complete_restored_LIVE_traversal',envelope+1024)
    emit(dict(event='before_complete_root_collect',entry_envelope=envelope))
    roots=collect(SimpleNamespace(),{'complete_native_checkpoint':saved});held['complete_registered_roots']=roots
    emit(dict(event='after_complete_root_collect',numeric_roots=len(roots.numeric),
        python_shallow_bytes=roots.python_shallow_bytes,unique_objects=roots.unique_objects))
    layout=roots.measure();result['numeric_layout']=asdict(layout)
    emit(dict(event='after_complete_numeric_owner_layout',layout=asdict(layout)))
    if layout.resident_entries>min(64_000_000,envelope):raise MemoryError('complete restored LIVE entries')
    f=saved['native_fields']
    emit(dict(event='before_complete_source_binding'))
    source,_=admit_source(f['source_fields'],f['source_proof_bytes'],expected_sha256=SOURCE_PROOF,enabled=True)
    held['source_binding']=source
    emit(dict(event='after_complete_source_binding',authentication=source.authentication))
    native,_=admit_native(enabled=True,source=source,hz=saved['post_relu_hz'],
        lineage=GuardedLocalSpliceJournal(**f['journal']),events=f['events'],
        actual_phase_image=f['actual_phase_image'],transfer_proof_bytes=f['transfer_proof_bytes'],
        expected_transfer_sha256=TRANSFER_PROOF,construction_report=f['construction_report'])
    held['native_binding']=native
    emit(dict(event='after_complete_native_binding',authentication=native.authentication))
    proof=json.loads(f['transfer_proof_bytes'])['complete_component_proof']
    plans=[Plan(**{**p,'tail':tuple(p['tail'])}) for p in proof['plans']]
    emit(dict(event='before_complete_exact_inverse',unit_plans=len(plans)))
    inverse=verify_inverse(source,native.hz,native.lineage,plans,pool=pool)
    emit(dict(event='after_complete_exact_inverse',inverse=inverse))
    if saved['hz_cache'][78] is not native.hz:raise ValueError('complete original native cache alias differs')
    if _sha256(SOURCE)!=SHA:raise ValueError('complete original archive mutated')
    auth=source.authentication+native.authentication
    return dict(complete_input=asdict(layout),decoder=decoder,inverse=inverse,
        source_authentication=source.authentication,native_authentication=native.authentication,
        separate_authentication_work=sum(v['work'] for v in auth),
        complete_restored_native_bound=True,complete_checkpoint_strongly_retained=True,
        original_archive_sha256=SHA,transformed_archive_sha256=wire['output_sha256'],
        full_CPU_work_256M_claim=False,concrete_witness=False,formal_gain=0)


def main():
    stage=sys.argv[1]
    if stage not in ('preflight','conversion','restore'):raise ValueError('explicit stage required')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);started=time.monotonic();held={}
    result=dict(completed=False,stage=stage,formal_gain=0,new_native_or_solver_executed=False)
    with (RUN/(stage+'_events.jsonl')).open('x') as log,(RUN/(stage+'_stacks.log')).open('x') as stacks:
        def emit(value):
            current,peak=tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0,0)
            event=dict(value,worker_elapsed_s=time.monotonic()-started,rss_bytes=rss_bytes(),
                lifetime_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                traced_current_bytes=current,traced_peak_bytes=peak,
                tracer_metadata_bytes=tracemalloc.get_tracemalloc_memory() if tracemalloc.is_tracing() else 0)
            log.write(json.dumps(event)+'\n');log.flush();result['last_completed_boundary']=value['event']
        try:
            faulthandler.dump_traceback_later(10,repeat=True,file=stacks)
            entry=rss_bytes()
            build=(lambda:restore(pool,result,held,emit,entry)) if stage=='restore' else (lambda:transform(stage,pool,result,emit))
            data,_=measured(build,observe=lambda s:result.update(measurement=s))
            result.update(completed=True,data=data)
        except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
        finally:
            faulthandler.cancel_dump_traceback_later()
            result.update(wall_s=time.monotonic()-started,diagnostic_work=pool.used,work_parts=pool.parts,
                complete_decoded_input_still_retained='complete_checkpoint' in held)
            _atomic_exclusive_json(RUN/(stage+'_result.json'),result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('data','wire_report')}),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
