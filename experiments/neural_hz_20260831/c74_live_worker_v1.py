"""One fresh original-network native first-write transaction; no terminal solve."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v1 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed as c5_installed
from experiments.neural_hz_20260831 import c74_live_runtime_v1 as runtime
from experiments.neural_hz_20260831.c32_native_slot_audit_v1 import verify as verify_slots
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import plain_entry
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import caller_roots
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import reference_subset
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c74_native_binding_20260913_v1'


class QualifiedStop(BaseException):pass


def main():
    directory=Path(sys.argv[1]).resolve()
    if directory!=DIRECTORY:raise ValueError('not preregistered native transaction directory')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((directory/'preregistered.json').read_text())
    started=time.monotonic();incoming={};retained=[]
    record=dict(schema='c74_actual_native_splice_qualification_v1',passed=False,formal_gain=0,
        fresh_original_network=True,archived_HZ_loaded_or_substituted=False,terminal_solve_executed=False,
        source_sha256=freeze['source_sha256'],provenance=freeze['provenance'],islands=[])
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
            raise ValueError('frozen source/library/artifact drift')
        anchored=json.loads((directory/'live_inputs.json').read_text())
        raw=(directory/'source_proof.json').read_bytes();transfer=(directory/'transfer_proof.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=anchored['source_proof_sha256'] or _sha256(directory/'transfer_proof.json')!=anchored['transfer_proof_sha256']:
            raise ValueError('independent textual proof inputs changed')
        transfer_sha=anchored['transfer_proof_sha256']
        if _sha256(directory/'native_bound.json')!=anchored['native_bound_sha256']:
            raise ValueError('complete new coupled bound changed')
        bound=json.loads((directory/'native_bound.json').read_text())
        if not bound['fits']:raise ValueError('complete new bound rejected')
        with (directory/'events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(dict(elapsed_s=time.monotonic()-started,**event))+'\n');stream.flush()
                if event['event'].startswith('c74_') or event['event'] in ('c9_live_constructed','c9_native_relu_constructed'):
                    print(json.dumps(event),flush=True)

            def before(state):
                tf=state['tf'];plain_entry(tf)
                extra={**caller_roots(),'apply_args':state['apply_args'],'apply_kwargs':state['apply_kwargs'],
                    'current_expression':state['expression'],'source_proof':raw,'complete_splice_transfer_proof':transfer}
                roots=collect(tf,extra);incoming[id(state)]=(extra,roots.fingerprint)
                entry=roots.measure()
                emit(dict(event='c74_live_entry',layer=state['layer'].id,roots=len(roots.numeric),
                    bytes=entry.resident_bytes,entries=entry.resident_entries))

            def ready(state):
                extra,fingerprint=incoming[id(state)];state['lifted'].validate()
                if collect(state['tf'],extra).fingerprint!=fingerprint:
                    raise ValueError('fresh source generation/binding mutated incoming roots')
                retained.append(state)
                record['islands'].append(dict(layer=state['layer'].id,binding=state['source_binding'],
                    construction=state['construction'],report=state['lifted'].report,incoming_roots_unchanged=True))

            def consumed(state,fact):
                tf,layer=state['tf'],state['layer'];new=state['lifted']
                if len(retained)!=1 or layer.id!=78 or layer.kind!='RELU':
                    raise ValueError('diagnostic target is not unique registered native phase')
                actual=tf._sparse_hz_cache.get(layer.id)
                if actual is not new.hz or layer.id in tf._sparse_affine_expr_cache or state['native_block_calls']!=1:
                    raise ValueError('actual first-write HZ publication missing/conflicted')
                bounds=state['apply_args'][0] if state['apply_args'] else state['apply_kwargs']['input_bounds']
                check=verify_slots(state,bounds)
                if (new.construction_report['generation_and_incremental_whole_work']>bound['whole']
                        or new.construction_report['generation_and_incremental_branch_work']>bound['branch']):
                    raise ValueError('actual new complete native work exceeds preregistered bound')
                record.update(post_relu=check,actual_native_construction=state['consumer_construction'],
                    actual_first_write=new.construction_report,actual_splice_authentication=state['phase_binding'])
                emit(dict(event='c74_actual_phase_and_complete_splice_bound',unit_pairs=len(new.lineage.columns),
                    actual_HZ_sha256=source_digest(actual),construction=state['consumer_construction']))
                extra={**caller_roots(),'applied_fact':fact,'source_proof':raw,'complete_splice_transfer_proof':transfer,
                    'retained_runtime':[runtime.numeric_roots(item) for item in retained],
                    'incoming_numeric_states':[item[0] for item in incoming.values()]}
                roots=collect(tf,extra);candidate=roots.measure()
                record.update(whole_live_state=asdict(candidate),whole_live_numeric_roots=len(roots.numeric),
                    python_shallow_bytes=roots.python_shallow_bytes)
                _,reference=reference_subset(roots,tf._net);lower=reference['reference_lower_bound']
                if (lower['resident_bytes'],lower['resident_entries'])!=(629346312,52428800):
                    raise ValueError('unchanged full two-leaf reference bound changed')
                physical=candidate.resident_bytes<lower['resident_bytes'] and candidate.resident_entries<lower['resident_entries']
                record.update(reference=reference,physical_decrease=physical)
                if not physical or collect(tf,extra).fingerprint!=roots.fingerprint:
                    raise ValueError('complete LIVE state storage/fingerprint gate failed')
                emit(dict(event='c74_complete_live_physical_pass',bytes=candidate.resident_bytes,
                    entries=candidate.resident_entries,reference_lower_bound=lower))
                native=inspect(actual);record['native_ingestion']=native
                if not native['passed'] or native['lowered_n_bin']!=actual.n_bin:
                    raise ValueError('actual changed native coefficient/frame fidelity failed')
                payload=dict(schema='c74_actual_native_splice_checkpoint_v1',formal_gain=0,
                    native_fields=dict(source_fields=new.source.original_fields,source_proof_bytes=new.source.proof_bytes,
                        journal=vars(new.lineage),events=new.events,actual_phase_image=new.actual_phase_image,
                        transfer_proof_bytes=new.transfer_proof_bytes,construction_report=new.construction_report),
                    numeric_roots=extra['retained_runtime'][0],complete_live_extra=extra,
                    net=tf._net,hz_cache=tf._sparse_hz_cache,expr_cache=tf._sparse_affine_expr_cache,
                    frame_widths=tf._sparse_frame_widths,relu_slots=tf._sparse_relu_slots,aux_slots=tf._sparse_aux_slots,
                    materialized_views=state['views'],applied_fact=fact,post_relu_hz=actual,
                    source_sha256=freeze['source_sha256'],provenance=freeze['provenance'],
                    actual_construction=state['consumer_construction'],post_relu_audit=check,
                    whole_live_path_proved=True,terminal_solve_executed=False)
                checkpoint=directory/'relu78.pickle'
                with checkpoint.open('xb') as f:
                    pickle.dump(payload,f,protocol=5);f.flush();os.fsync(f.fileno())
                record.update(passed=True,status='LIVE_NATIVE_FIRST_WRITE_QUALIFIED',whole_live_path_proved=True,
                    actual_cache_publication=True,checkpoint_sha256=_sha256(checkpoint),
                    checkpoint_bytes=checkpoint.stat().st_size,actual_hz_sha256=source_digest(actual))
                emit(dict(event='c74_live_qualified',formal_gain=0))
                raise QualifiedStop()

            with c5_installed(enabled=True,emit=emit):
                with runtime.installed(enabled=True,source_bytes=raw,source_sha=anchored['source_proof_sha256'],
                        transfer_bytes=transfer,transfer_sha=transfer_sha,
                        before=before,ready=ready,consumed=consumed,emit=emit):prefix.main()
        if not record['passed']:raise ValueError('registered target did not execute/qualify')
    except QualifiedStop:pass
    except (runtime.SelectedRejected,Exception) as exc:
        record.update(passed=False,status='LIVE_NATIVE_FIRST_WRITE_REJECTED',failure=dict(type=type(exc).__name__,reason=str(exc)))
    finally:
        source_auth=[];native_auth=[]
        for item in retained:
            obj=item['lifted']
            if type(obj) is runtime.NativeState:
                source_auth.extend(obj.source.authentication);native_auth.extend(obj.authentication)
            elif type(obj) is runtime.SourceState:source_auth.extend(obj.authentication)
        record.update(source_authentication_calls=source_auth,native_authentication_calls=native_auth,
            total_separate_hash_authentication_work=sum(r['work'] for r in source_auth+native_auth),
            all_CPU_work_256M_claim=False)
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(directory/'qualification.json',record)
        print(json.dumps(dict(status=record.get('status'),passed=record['passed'],failure=record.get('failure'))),flush=True)
    if not record['passed']:raise SystemExit(1)


if __name__=='__main__':main()
