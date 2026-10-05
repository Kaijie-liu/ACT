"""Fresh C31 original-source generation, complete new closure, no native run."""

import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace
import weakref

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.neural_hz_20260831.c31_prepared_emission_v1 import lift
from experiments.neural_hz_20260831.c31_prepared_report_audit_v1 import audit
from experiments.neural_hz_20260831.c24_closed_state_v1 import close,export,restore
from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
from experiments.neural_hz_20260831.c27_owned_journal_worker_v1 import original_expression
from experiments.neural_hz_20260831.c24_dense_closed_worker_v1 import load_oracle
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c31_prepared_generator_20260911_v1'
SNAPSHOT=EXP/'results/c5_first_terminal_20260905_v1/layer75.pickle'
LIVE=EXP/'results/c25_live_relu_20260911_v1'
PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    record=dict(schema='c31_fresh_prepared_owned_generator_v1',completed=False,formal_gain=0,
        source_sha256=freeze['source_sha256'],provenance=freeze['provenance'],
        new_native_relu_executed=False,solver_executed=False,whole_live_path_proved=False,
        complete_C30_native_integration_executed=False)
    started=time.monotonic();initial=None
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
            raise ValueError('frozen source/reference drift before deserialization')
        with SNAPSHOT.open('rb') as f:snapshot=pickle.load(f)
        with (LIVE/'relu78.pickle').open('rb') as f:saved=pickle.load(f)
        if saved['schema']!='c25_live_relu_checkpoint_v1' or not saved['whole_live_path_proved']:
            raise ValueError('old complete native source not authenticated')
        old=restore(saved['closed_fields'],saved['closed_proof_bytes'],expected_proof_sha256=PROOF)
        original_hz,maps=load_oracle()
        expr,widths=original_expression(snapshot)
        inputs=dict(snapshot=snapshot,expression=expr,original_HZ=original_hz,original_maps=maps,complete_C25_archive=saved)
        initial=collect(SimpleNamespace(),inputs).fingerprint
        original_sha=source_digest(original_hz);old_sha=old.fingerprint()
        with (DIRECTORY/'events.jsonl').open('x') as stream:
            def emit(name,values):
                event=dict(event=name,**values,worker_elapsed_s=time.monotonic()-started)
                stream.write(json.dumps(event)+'\n');stream.flush();print(json.dumps(event),flush=True)
            draft,construction=measured_build(lambda:lift(expr,np.ones(expr.n_out,bool),
                enabled=True,frame_widths=widths,observe=emit))
            record.update(fresh_original_expression_generation_executed=True,generation=construction,generation_report=draft.report)
            emit('actual_prepared_owned_generator_completed',dict(construction=construction,report=draft.report))
            diagnostic=WorkPool(256_000_000)
            report_start=time.monotonic()
            report_proof=audit(draft,old,pool=diagnostic)
            record.update(complete_report_proof=report_proof,report_proof_elapsed_s=time.monotonic()-report_start,
                report_proof_diagnostic_work=diagnostic.used,report_proof_diagnostic_parts=dict(diagnostic.parts))
            enc=draft.report['prepared_encoding']
            if (enc['logical_input_coefficients'],enc['logical_input_rows'],enc['actual_preparation_attempts'],
                    enc['physical_rows_before_alias'])!=(11160274,246612,246645,246640):
                raise ValueError('complete actual prepared source population changed')
            if (len(draft.owners),len(draft.def_rows),draft.report['packed_logical_rows'],
                    draft.report['alias_quotient']['selected_aliases'],len(draft.uid_slabs))!=(243162,28,5,100603,19):
                raise ValueError('complete exact MAIN/radix/alias/UID population changed')
            # A changed report cannot inherit the old complete proof receipt.
            try:bind(draft,saved['closed_proof_bytes'],expected_proof_sha256=PROOF,enabled=True)
            except ValueError:record['old_report_proof_rejected']=True
            else:raise ValueError('old proof incorrectly authorized new report fields')
            retired=[weakref.ref(n[k]) for n in draft.nodes for k in ('support','needed','slots','exponents')]
            draft_ref=weakref.ref(draft)
            proof_start=time.monotonic()
            candidate,proof=close(draft,original_hz,maps,enabled=True)
            record['new_complete_source_proof_elapsed_s']=time.monotonic()-proof_start
            if candidate.hz is not draft.hz or candidate.owners is not draft.owners:
                raise ValueError('new full close substituted an archived generated state')
            fields,raw=export(candidate);proof_sha=hashlib.sha256(raw).hexdigest()
            if proof_sha==PROOF or candidate.fingerprint()==old_sha:
                raise ValueError('new report reused old proof identity')
            # Explicitly retire only NEW unpublished graph storage AFTER proof.
            del draft
            gc.collect()
            if draft_ref() is not None or any(ref() is not None for ref in retired):
                raise ValueError('new complete Closed retains Draft graph arrays')
            if len(retired)!=144:raise ValueError('new graph array retirement population changed')
            with (DIRECTORY/'closed_proof.json').open('xb') as f:
                f.write(raw);f.flush();os.fsync(f.fileno())
            emit('new_complete_source_owner_UID_proof',dict(closed_identity=candidate.fingerprint(),
                proof_sha256=proof_sha,proof=proof,all_new_graph_arrays_retired=len(retired),
                elapsed_s=record['new_complete_source_proof_elapsed_s']))
            # The new generator is actual; the appended costs are anchored
            # completed components, NOT a newly executed combined native path.
            c30=json.loads((EXP/'results/c30_first_write_20260911_v1/result.json').read_text())
            c27=json.loads((EXP/'results/c27_owned_journal_20260911_v1/result.json').read_text())
            components=dict(C27_fresh_permit_and_owned_transplant=51637,
                actual_native_sparse_phase_events=37800,C30_full_append_discovery=c30['discovery_work'],
                C30_first_write_extra=c30['writer']['additional_routing_work'])
            # The inherited components are not repriced from source identities.
            if not c30['completed'] or not c27['completed']:raise ValueError('inherited components not completed')
            projected=candidate.report['total_work_upper']+sum(components.values())
            assessment=dict(actual_new_generator_whole_work=candidate.report['total_work_upper'],
                actual_new_generator_branch_work=candidate.report['largest_branch_work_upper'],
                additional_components_not_integrated=components,prospective_combined_incremental_work=projected,
                prospective_256M_headroom=256_000_000-projected,
                common_native_payload_traffic_NOT_free=c30['writer']['baseline_native_payload_transfers'],
                candidate_native_payload_traffic=c30['writer']['candidate_native_payload_transfers'],
                combined_native_admission_or_full_LIVE_work_gate_proved=False,
                paired_wall_speed_claim=False,formal_gain=0)
            if (old.fingerprint()!=old_sha or source_digest(original_hz)!=original_sha
                    or collect(SimpleNamespace(),inputs).fingerprint!=initial):
                raise ValueError('original snapshot/source/oracles were mutated')
            candidate.validate()
            payload=dict(schema='c31_new_checked_prepared_Closed_v1',fields=fields,proof_bytes=raw,
                closed_proof_sha256=proof_sha,identity=proof,complete_report_proof=report_proof,
                provenance=freeze['provenance'],source_sha256=freeze['source_sha256'],
                origin_snapshot_sha256=_sha256(SNAPSHOT),old_checked_proof_sha256=PROOF,
                all_new_graph_fields_physically_retired=True,whole_live_path_proved=False,formal_gain=0)
            path=DIRECTORY/'closed_hz.pickle'
            with path.open('xb') as f:pickle.dump(payload,f,protocol=5);f.flush();os.fsync(f.fileno())
            record.update(completed=True,new_complete_source_proof=proof,new_closed_identity=candidate.fingerprint(),
                new_closed_proof_sha256=proof_sha,new_HZ_sha256=source_digest(candidate.hz),
                all_new_graph_fields_physically_retired=True,new_graph_arrays_retired=len(retired),
                complete_original_snapshot_and_oracles_retained_OFFLINE=True,
                source_snapshot_and_oracles_unchanged=True,cost_assessment=assessment,
                checkpoint_sha256=_sha256(path),checkpoint_bytes=path.stat().st_size)
            emit('actual_generator_and_remaining_native_model',assessment)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        if initial is not None:record['source_snapshot_and_oracles_unchanged']=collect(SimpleNamespace(),inputs).fingerprint==initial
        record.update(wall_s=time.monotonic()-started,max_rss_kib_including_oracles=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'result.json',record)
        print(json.dumps(dict(completed=record['completed'],failure=record.get('failure'),formal_gain=0)),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
