# SPDX-License-Identifier: AGPL-3.0-or-later
"""C103 fused exact row construction with unchanged complete terminal checks."""
from dataclasses import asdict
import hashlib
import faulthandler
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from act.back_end.solver.solver_hz import HZSolver
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v1 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed as c5_installed
from experiments.neural_hz_20260831 import c103_live_runtime_v1 as runtime
from experiments.neural_hz_20260831.c32_native_slot_audit_v1 import verify as verify_slots
from experiments.neural_hz_20260831.c100_runtime_final_binding_v1 import verify as verify_final,load as load_final
from experiments.neural_hz_20260831.c100_terminal_observer_v1 import installed as observed
from experiments.neural_hz_20260831.c33_collision_safe_events_v1 import record as event_record
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import plain_entry
from experiments.neural_hz_20260831.c102_runtime_roots_v1 import RuntimeRoots
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import caller_roots
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import reference_subset
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c103_fused_row_20260913_v1'
PROOF_DIRECTORY=EXP/'results/c100_fresh_circuit_terminal_20260913_v4'
from experiments.neural_hz_20260831.c100_native_binding_v1 import journal_image


def main():
    directory=Path(sys.argv[1]).resolve()
    if directory!=DIRECTORY:raise ValueError('unregistered changed terminal directory')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((directory/'preregistered.json').read_text());started=time.monotonic()
    incoming={};retained=[];phase_facts=[]
    native_evaluate=HZSolver.evaluate_spec
    stacks=(directory/'fatal.log').open('x')
    faulthandler.enable(file=stacks,all_threads=True)
    record=dict(schema='c103_changed_terminal_audit_v1',formal_gain=0,terminal_gates_passed=False,
        terminal_solve_executed=False,ordinary_terminal_returned=False,archived_HZ_loaded_or_substituted=False,
        source_sha256=freeze['source_sha256'],provenance=freeze['provenance'],islands=[],observations=[])
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('whole source freeze drift')
        links=json.loads((PROOF_DIRECTORY/'prepared_inputs.json').read_text())
        if any(_sha256(PROOF_DIRECTORY/n)!=sha for n,sha in links.items()):
            raise ValueError('complete prepared input chain differs before fresh execution')
        anchored=json.loads((PROOF_DIRECTORY/'live_inputs.json').read_text())
        if not json.loads((PROOF_DIRECTORY/'prepare_result.json').read_text())['completed']:
            raise ValueError('complete new-source/input/property preparation required')
        PROOF_SHA=anchored['source_proof_sha256'];TRANSFER_SHA=anchored['transfer_proof_sha256']
        raw=(PROOF_DIRECTORY/'source_proof.json').read_bytes();transfer=(PROOF_DIRECTORY/'transfer_proof.json').read_bytes()
        final_raw=(PROOF_DIRECTORY/'final_proof.json').read_bytes()
        final_sha=anchored['final_proof_sha256']
        if _sha256(PROOF_DIRECTORY/'native_bound.json')!=anchored['native_bound_sha256']:
            raise ValueError('complete terminal work bound changed')
        bound=json.loads((PROOF_DIRECTORY/'native_bound.json').read_text())
        if not bound['fits']:raise ValueError('complete terminal bound rejected')
        if hashlib.sha256(raw).hexdigest()!=PROOF_SHA or hashlib.sha256(transfer).hexdigest()!=TRANSFER_SHA:
            raise ValueError('actual source or splice proof changed')
        final_proof=load_final(final_raw,final_sha)
        final_proof.update(native_lowered_n_cont=anchored['native_lowered_n_cont'],
            native_lowered_n_bin=anchored['native_lowered_n_bin'])
        with (directory/'events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(event_record(time.monotonic()-started,event))+'\n');stream.flush()
                if event['event'].startswith(('c100_','c102_')) or event['event'] in ('ordinary_terminal_start','ordinary_milp_return'):
                    print(json.dumps(event),flush=True)

            traversal=RuntimeRoots(emit,enabled=True)

            def before(state):
                tf=state['tf'];plain_entry(tf)
                extra={**caller_roots(),'apply_args':state['apply_args'],'apply_kwargs':state['apply_kwargs'],
                    'current_expression':state['expression'],'source_proof':raw,'splice_proof':transfer,'final_proof':final_raw}
                roots=traversal.collect(tf,extra,boundary='original_entry');incoming[id(state)]=(extra,roots.fingerprint)
                measure=roots.measure()
                emit(dict(event='c100_original_live_entry',layer=state['layer'].id,bytes=measure.resident_bytes,entries=measure.resident_entries))

            def ready(state):
                extra,fingerprint=incoming[id(state)];state['lifted'].validate()
                if traversal.collect(state['tf'],extra,boundary='original_preservation').fingerprint!=fingerprint:raise ValueError('fresh generator changed original live roots')
                retained.append(state)
                record['islands'].append(dict(layer=state['layer'].id,construction=state['construction'],
                    binding=state['source_binding'],report=state['lifted'].report))

            def consumed(state,fact):
                if len(retained)!=1 or state['layer'].id!=78 or state['native_block_calls']!=1:
                    raise ValueError('diagnostic target is not the unique selected original native phase')
                state['lifted'].validate();phase_facts.append(fact)
                bounds=state['apply_args'][0] if state['apply_args'] else state['apply_kwargs']['input_bounds']
                check=verify_slots(state,bounds)
                if (source_digest(state['lifted'].hz)!=final_proof['post_HZ_sha256']
                        or not state['construction']['measured_transient_gate']
                        or not state['consumer_construction']['measured_transient_gate']):
                    raise ValueError('fresh changed native source/resource gate differs from qualification')
                record.update(post_relu=check,actual_native_construction=state['consumer_construction'],
                    actual_first_write=state['lifted'].construction_report)
                emit(dict(event='c100_actual_spliced_native_qualified',unit_pairs=len(state['lifted'].lineage.local.columns),
                    source_HZ_sha256=source_digest(state['lifted'].hz)))

            def evaluated(solver,output,out_spec,**kwargs):
                if len(retained)!=1 or not record.get('post_relu') or record['terminal_solve_executed']:
                    raise runtime.SelectedRejected('terminal without unique qualified actual native source')
                if (solver.tolerance!=1e-7 or not solver.simplify_final_hz or solver.neural_hz_projection
                        or solver.neural_hz_phase_fixing or solver.time_limit!=45. or kwargs.get('timelimit')!=45.):
                    raise runtime.SelectedRejected('ordinary solver parameters or prohibited rescue changed')
                state=retained[0];tf=state['tf'];input_hz=kwargs.get('input_hz')
                state['budget'].charge('terminal_observer_scope_and_exclusive_records',512)
                binding=verify_final(state,output,input_hz,out_spec,kwargs,final_raw,final_sha,pool=state['budget'],enabled=True)
                record['final_source_binding']=binding
                # Explicitly expose every new retained terminal input/property/
                # output/config/proof/fact root; the future solver model is the
                # ordinary native allocation, never a hidden prebuilt oracle.
                extra={**caller_roots(),'retained_runtime':[runtime.numeric_roots(s) for s in retained],
                    'incoming_numeric_states':[v[0] for v in incoming.values()],'actual_phase_facts':phase_facts,
                    'actual_terminal_output':output,'actual_terminal_input':input_hz,
                    'actual_output_spec':out_spec,'terminal_kwargs':kwargs,'solver_fields':dict(vars(solver)),
                    'source_proof':raw,'splice_proof':transfer,'final_proof':final_raw,
                    'prior_traversal_diagnostic_receipts':tuple(traversal.receipts)}
                traversal.pool.charge('c102_terminal_boundary_markers',6*64)
                roots=traversal.collect(tf,extra,boundary='complete_final_live')
                emit(dict(event='c102_final_numeric_measure_started'))
                candidate=roots.measure()
                emit(dict(event='c102_final_numeric_measure_returned',bytes=candidate.resident_bytes,entries=candidate.resident_entries))
                record.update(whole_live_state=asdict(candidate),whole_live_numeric_roots=len(roots.numeric),
                    python_shallow_bytes=roots.python_shallow_bytes,
                    generation_plus_incremental_work=state['budget'].whole_base+state['budget'].used,
                    branch_plus_incremental_work=state['budget'].branch_base+state['budget'].used,
                    actual_incremental_work_parts=dict(state['budget'].parts))
                emit(dict(event='c102_complete_reference_started'))
                witness,reference=reference_subset(roots,tf._net);lower=reference['reference_lower_bound']
                emit(dict(event='c102_complete_reference_returned',bytes=lower['resident_bytes'],entries=lower['resident_entries']))
                if (lower['resident_bytes'],lower['resident_entries'])!=(629346312,52428800):
                    raise runtime.SelectedRejected('same two-leaf reference changed')
                physical=candidate.resident_bytes<lower['resident_bytes'] and candidate.resident_entries<lower['resident_entries']
                record.update(reference=reference,physical_decrease=physical)
                if not physical or traversal.collect(tf,extra,boundary='complete_final_preservation').fingerprint!=roots.fingerprint:
                    raise runtime.SelectedRejected('complete final LIVE source/storage/fingerprint gate failed')
                if (state['budget'].whole_base+state['budget'].used>bound['whole']
                        or state['budget'].branch_base+state['budget'].used>bound['branch']):
                    raise runtime.SelectedRejected('actual complete terminal construction exceeds bound')
                if traversal.calls!=4:raise runtime.SelectedRejected('incomplete four-boundary traversal')
                record.update(complete_traversal_receipts=list(traversal.receipts),
                    diagnostic_shared_token_work=traversal.pool.used,
                    diagnostic_shared_token_parts=dict(traversal.pool.parts),
                    numeric_hash_traffic_in_token_pool=False,all_CPU_work_in_generation_cap=False)
                emit(dict(event='c102_final_native_ingestion_started'))
                native=inspect(output);record['final_native_ingestion']=native
                emit(dict(event='c102_final_native_ingestion_returned',passed=native['passed']))
                if (not native['passed'] or native['lowered_n_cont']!=final_proof['native_lowered_n_cont']
                        or native['lowered_n_bin']!=final_proof['native_lowered_n_bin']):
                    raise runtime.SelectedRejected('actual final model coefficient/frame fidelity failed')
                payload=dict(schema='c100_reconstructable_final_native_checkpoint_v1',formal_gain=0,
                    native_fields=dict(complete_circuit_state=state['lifted'].source.circuit_state,
                        source_proof_bytes=state['lifted'].source.proof_bytes,
                        journal=journal_image(state['lifted'].lineage),events=state['lifted'].events,
                        actual_phase_image=state['lifted'].actual_phase_image,
                        transfer_proof_bytes=state['lifted'].transfer_proof_bytes,
                        construction_report=state['lifted'].construction_report),
                    runtime_numeric_roots=extra['retained_runtime'][0],complete_live_extra=extra,
                    net=tf._net,hz_cache=tf._sparse_hz_cache,expr_cache=tf._sparse_affine_expr_cache,
                    selected_layer=state['layer'],post_relu_hz=state['lifted'].hz,
                    final_hz=output,input_hz=input_hz,output_spec=out_spec,terminal_kwargs=kwargs,
                    frame_widths=tf._sparse_frame_widths,relu_slots=tf._sparse_relu_slots,aux_slots=tf._sparse_aux_slots,
                    source_sha256=freeze['source_sha256'],provenance=freeze['provenance'],
                    final_proof_bytes=final_raw,final_proof_sha256=final_sha,final_source_binding=binding)
                checkpoint=directory/'native_state.pickle'
                with checkpoint.open('xb') as f:pickle.dump(payload,f,protocol=5);f.flush();os.fsync(f.fileno())
                record.update(terminal_gates_passed=True,final_hz_sha256=source_digest(output),
                    native_state_sha256=_sha256(checkpoint),native_state_bytes=checkpoint.stat().st_size,
                    source_authentication=list(state['lifted'].source.authentication),
                    native_authentication=list(state['lifted'].authentication))
                _atomic_exclusive_json(directory/'terminal_gate.json',record)
                emit(dict(event='ordinary_terminal_start',final_HZ_sha256=record['final_hz_sha256'],
                    live_bytes=candidate.resident_bytes,n_cont=native['lowered_n_cont'],n_bin=native['lowered_n_bin'],solver_seconds=45.))
                record['terminal_solve_executed']=True;tick=time.monotonic()
                def observation(event):
                    record['observations'].append(event);emit(event)
                def save_point(model,x):
                    with (directory/'native_witness_1.npz').open('xb') as f:
                        np.savez(f,solver_x=x,cont_source=model.cont_source,bin_source=model.bin_source)
                        f.flush();os.fsync(f.fileno())
                    emit(dict(event='c100_native_candidate_saved_before_reconstruction',sha256=_sha256(directory/'native_witness_1.npz')))
                with observed(state['lifted'],output,input_hz,kwargs['input_shape'],final_proof,
                        enabled=True,emit=observation,on_point=save_point):
                    result=native_evaluate(solver,output,out_spec,**kwargs)
                record.update(ordinary_terminal_returned=True,ordinary_terminal_wall_s=time.monotonic()-tick,
                    ordinary_statuses=[getattr(r.status,'name',str(r.status)) for r in result])
                emit(dict(event='c100_ordinary_terminal_returned',wall_s=record['ordinary_terminal_wall_s'],statuses=record['ordinary_statuses']))
                return result

            HZSolver.evaluate_spec=evaluated
            with c5_installed(enabled=True,emit=emit):
                with runtime.installed(enabled=True,source_bytes=raw,source_sha=PROOF_SHA,
                        transfer_bytes=transfer,transfer_sha=TRANSFER_SHA,
                        before=before,ready=ready,consumed=consumed,emit=emit):prefix.main()
        if not record['ordinary_terminal_returned']:raise ValueError('ordinary changed terminal did not return')
        outcome=json.loads((directory/'result.json').read_text())
        record['concrete_result_verdict']=outcome.get('verdict')
        record['concrete_validations']=outcome.get('concrete_validations',[])
        if outcome.get('error') or any(not item['valid'] for item in record['concrete_validations']):
            raise ValueError('terminal pipeline error or invalid concrete neural witness')
    except (runtime.SelectedRejected,Exception) as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        faulthandler.disable();stacks.close()
        HZSolver.evaluate_spec=native_evaluate
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(directory/'terminal_audit.json',record)
        print(json.dumps(dict(gates=record['terminal_gates_passed'],returned=record['ordinary_terminal_returned'],failure=record.get('failure'))),flush=True)
    if record.get('failure'):raise SystemExit(1)


if __name__=='__main__':main()
