"""One authenticated full-population first-write component qualification."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c30_first_write_v1 import AppendView, splice_append
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import check_append
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import compile_lineage
from experiments.neural_hz_20260831.c26_transplant_audit_v1 import verify
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import storage
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c30_first_write_20260911_v1'
LIVE=EXP/'results/c25_live_relu_20260911_v1'
C26=EXP/'results/c26_transplant_census_20260911_v1'
C28=EXP/'results/c28_consumer_discovery_20260911_v1'
C15=EXP/'results/c15_unit_row_splice_20260910_v1'
PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def bits_equal(a,b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8),b.view(np.uint8))


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    record=dict(schema='c30_actual_first_write_qualification_v1',completed=False,formal_gain=0,
        source_sha256=freeze['source_sha256'],provenance=freeze['provenance'],
        new_generator_executed=False,new_native_relu_executed=False,solver_executed=False,
        whole_live_path_proved=False,live_admission_certificate=False)
    started=time.monotonic()
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
            raise ValueError('complete source/artifact drift before deserialization')
        with (LIVE/'relu78.pickle').open('rb') as f: saved=pickle.load(f)
        if saved['schema']!='c25_live_relu_checkpoint_v1' or not saved['whole_live_path_proved']:
            raise ValueError('actual native reference is not the completed C25 transaction')
        closed=restore(saved['closed_fields'],saved['closed_proof_bytes'],expected_proof_sha256=PROOF)
        post=saved['post_relu_hz']; owned=saved['phase_ownership']
        overlay=Overlay(owned['base'],owned['events'],owned['old_uid_ceiling'])
        post_sha=source_digest(post); pre_sha=source_digest(closed.hz); closed_sha=closed.fingerprint()
        event_sha=hashlib.sha256(overlay.events.tobytes()).hexdigest()
        if (owned['post_hz'] is not post or owned['base'] is not closed.owners
                or owned['post_sha256']!=post_sha or owned['event_sha256']!=event_sha):
            raise ValueError('source/phase/ownership binding differs')
        closed.validate(); overlay.validate(); check_append(closed.hz,post)
        # Only the tiny appended rows are decomposed from the authenticated
        # offline native oracle. This is not a fresh native call or live claim.
        pre=closed.hz
        view=AppendView(pre,post.Ac[pre.n_eq:],post.Ab[pre.n_eq:],post.b[pre.n_eq:],
            post.Auc[pre.n_ineq:],post.Aub[pre.n_ineq:],post.ub[pre.n_ineq:],post.c,post.Gc,post.Gb)
        with (DIRECTORY/'events.jsonl').open('x') as stream:
            def emit(name,values):
                event=dict(event=name,**values,worker_elapsed_s=time.monotonic()-started)
                stream.write(json.dumps(event)+'\n');stream.flush();print(json.dumps(event),flush=True)
            dp=WorkPool(256_000_000);wp=WorkPool(256_000_000)
            def build():
                plans,stats=discover_append(closed,view,overlay,pool=dp,enabled=True)
                result=splice_append(view,plans,pool=wp,enabled=True)
                if result is None:raise ValueError('complete real source produced no strict splice')
                return plans,stats,*result
            (plans,stats,new,writer),construction=measured_build(build)
            record.update(discovery_work=dp.used,discovery_work_parts=dict(dp.parts),
                discovery_statistics=stats,writer=writer,construction=construction,
                new_spliced_HZ_constructed=True)
            emit('actual_first_write_completed',dict(discovery_work=dp.used,writer=writer,construction=construction))
            # Only AFTER complete new selection and writing may any saved plan
            # population/independent spliced reference be opened.
            prior=json.loads((C28/'plans.json').read_text())
            ending=json.loads((C28/'exit.json').read_text())
            prior_result=json.loads((C28/'result.json').read_text())
            if (not prior_result['completed'] or ending.get('worker_exit_code')!=0
                    or ending.get('tests_exit_code')!=0 or ending.get('timeout_s')
                    or ending['source_drift'] or ending['provenance_drift']
                    or prior['source_closed_sha256']!=closed_sha or prior['actual_post_HZ_sha256']!=post_sha):
                raise ValueError('complete independent source/incidence qualification not bound')
            if json.loads(json.dumps([asdict(p) for p in plans]))!=prior['plans']:
                raise ValueError('complete append-view plans differ from C28 factor-checked population')
            if (len(plans),stats['selected_new_consumers'],stats['selected_old_consumers'])!=(268,199,69):
                raise ValueError('complete registered population changed')
            with (C26/'lineage.pickle').open('rb') as f: reference=pickle.load(f)
            with (C15/'spliced_hz.pickle').open('rb') as f: oracle_saved=pickle.load(f)
            oracle=oracle_saved['candidate']; oracle.validate(); reference['draft'].validate()
            c26_result=json.loads((C26/'result.json').read_text());c26_exit=json.loads((C26/'exit.json').read_text())
            if (not c26_result['completed'] or c26_exit.get('worker_exit_code')!=0
                    or c26_exit.get('tests_exit_code')!=0 or c26_exit.get('timeout_s')
                    or c26_exit['source_drift'] or c26_exit['provenance_drift']
                    or reference['closed_identity']!=closed_sha or reference['source_post_HZ_sha256']!=post_sha
                    or reference['proof']!=c26_result['proof'] or reference['oracle_fingerprint']!=oracle.fingerprint()):
                raise ValueError('complete independent C26/C15 source/math/box chain changed')
            diag=WorkPool(256_000_000)
            lineage=compile_lineage(closed.eq_roots,closed.eq_scales,plans,old_n_cont=closed.old_n_cont,
                old_n_eq=closed.old_n_eq,pool=diag,enabled=True)
            if lineage.fingerprint()!=reference['draft'].fingerprint():
                raise ValueError('all semantic lineage maps differ from independent reference')
            checked=0
            for name in ('Ac','Ab','Auc','Aub'):
                a,b=getattr(new,name),getattr(oracle.hz,name)
                diag.charge('independent_all_written_matrix_bits',10*int(a.nnz)+4*len(a.indptr))
                if a.shape!=b.shape or not all(bits_equal(getattr(a,k),getattr(b,k)) for k in ('data','indices','indptr')):
                    raise ValueError('full first-write matrix differs from independent C15: '+name)
                # Recompute flags on a sharing wrapper with no cached flag,
                # rather than trusting the writer's source-derived flags.
                fresh=type(a)((a.data,a.indices,a.indptr),shape=a.shape,copy=False)
                if not fresh.has_canonical_format or np.any(a.data==0.) or not np.isfinite(a.data).all():
                    raise ValueError('independent complete canonical/zero-free guard failed')
                checked+=int(a.nnz)
            diag.charge('independent_complete_RHS_and_output_bits',4*(new.n_eq+new.n_ineq+new.n_out+new.Gc.nnz+new.Gb.nnz))
            if (not bits_equal(new.b,oracle.hz.b) or not bits_equal(new.ub,oracle.hz.ub)
                    or not bits_equal(new.c,post.c) or new.Gc is not post.Gc or new.Gb is not post.Gb
                    or new.frame_id!=post.frame_id or new.n_cont!=post.n_cont or new.n_bin!=post.n_bin or not new.exact):
                raise ValueError('RHS/binary/global/output protection failed')
            diag.charge('independent_closed_UID_metadata',32*(len(closed.owners)+post.n_eq+post.n_ineq))
            eq,le=closed_uid_tables(closed);first=overlay.old_uid_ceiling
            ne,nl=post.n_eq-pre.n_eq,post.n_ineq-pre.n_ineq
            eq=np.r_[eq,np.arange(first,first+ne,dtype=np.int64)]
            le=np.r_[le,np.arange(first+ne,first+ne+nl,dtype=np.int64)]
            proof=verify(closed,post,overlay,lineage.columns,eq,le,lineage,oracle,pool=diag,
                expected_oracle_fingerprint=reference['oracle_fingerprint'])
            if checked!=10_960_724-536 or sum(v.get('parent_terms_written_once',0) for v in writer['matrices'].values())!=1_248_197:
                raise ValueError('complete strict reduction/first-write parent traffic mismatch')
            if (closed.fingerprint()!=closed_sha or source_digest(post)!=post_sha or source_digest(pre)!=pre_sha
                    or hashlib.sha256(overlay.events.tobytes()).hexdigest()!=event_sha):
                raise ValueError('original source/owner fields mutated')
            new_sha=source_digest(new)
            old_storage,new_storage=storage({'hz':post}),storage({'hz':new})
            routed=dp.used-prior_result['discovery_work']
            prospective=252_074_173+routed+writer['additional_routing_work']
            assessment=dict(C29_emitter_C27_plus_C28_prospective_subtotal=252_074_173,
                actual_append_discovery_extra_work=routed,actual_writer_extra_work=writer['additional_routing_work'],
                prospective_generator_and_incremental_metadata_model=prospective,
                model_256M_headroom=256_000_000-prospective,
                common_native_payload_traffic_NOT_free=writer['baseline_native_payload_transfers'],
                actual_native_payload_traffic=writer['candidate_native_payload_transfers'],
                native_traffic_saved=writer['baseline_native_payload_transfers']-writer['candidate_native_payload_transfers'],
                fresh_generator_C29_owned_emitter_and_C27_integration_executed=False,
                full_live_admission_or_full_work_gate_proved=False,paired_wall_speed_claim=False,formal_gain=0)
            artifact=dict(schema='c30_provisional_actual_spliced_HZ_v1',hz=new,lineage=lineage,
                new_HZ_sha256=new_sha,source_post_HZ_sha256=post_sha,closed_identity=closed_sha,
                independent_C26_lineage_sha256=_sha256(C26/'lineage.pickle'),
                independent_C15_archive_sha256=_sha256(C15/'spliced_hz.pickle'),
                full_matrix_and_RHS_bit_identity=True,complete_UID_box_reconstruction_transfer=proof,
                source_sha256=freeze['source_sha256'],provenance=freeze['provenance'],
                formal_gain=0,live_admission_certificate=False,whole_live_path_proved=False)
            with (DIRECTORY/'spliced_hz.pickle').open('xb') as f:pickle.dump(artifact,f,protocol=5)
            record.update(completed=True,all_written_predicate_coefficients_checked=checked,
                complete_proof_transfer=proof,all_matrix_RHS_output_bits_equal=True,
                actual_new_HZ_sha256=new_sha,source_post_HZ_sha256=post_sha,source_pre_HZ_sha256=pre_sha,
                source_closed_identity=closed_sha,lineage_fingerprint=lineage.fingerprint(),
                functional_lineage_copies_are_diagnostic=True,independent_diagnostic_work=diag.used,
                independent_diagnostic_parts=dict(diag.parts),cost_assessment=assessment,
                HZ_body_only_storage=dict(old_bytes=old_storage.resident_bytes,new_bytes=new_storage.resident_bytes,
                    old_entries=old_storage.resident_entries,new_entries=new_storage.resident_entries,
                    NOT_whole_LIVE_or_lineage_storage=True),all_sources_unchanged=True,
                artifact_sha256=_sha256(DIRECTORY/'spliced_hz.pickle'),
                artifact_bytes=(DIRECTORY/'spliced_hz.pickle').stat().st_size)
            emit('complete_independent_first_write_proof',dict(coefficients=checked,proof=proof,diagnostic_work=diag.used))
            emit('integration_model_NOT_a_live_gate',assessment)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'result.json',record)
        print(json.dumps(dict(completed=record['completed'],failure=record.get('failure'),formal_gain=0)),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
