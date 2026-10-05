"""One complete actual-archive transplant census; no fresh HZ or solve."""

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
import numpy as np
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import check_append,incidence_oracle,verify_all_and_discover,BranchPool
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import compile_lineage
from experiments.neural_hz_20260831.c26_transplant_audit_v1 import plans_from_checked_incidence,verify
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c26_transplant_census_20260911_v1'
LIVE=EXP/'results/c25_live_relu_20260911_v1'
C15=EXP/'results/c15_unit_row_splice_20260910_v1'
PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    started=time.monotonic(); record={'completed':False,'formal_gain':0,
        'new_HZ_generation_executed':False,'new_native_relu_executed':False,'solver_executed':False,
        'whole_live_path_proved':False,'source_sha256':freeze['source_sha256'],'provenance':freeze['provenance']}
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
            raise ValueError('frozen source/artifact drift')
        with (LIVE/'relu78.pickle').open('rb') as handle: saved=pickle.load(handle)
        if saved['schema']!='c25_live_relu_checkpoint_v1' or not saved['whole_live_path_proved']:
            raise ValueError('not the complete actual native source archive')
        closed=restore(saved['closed_fields'],saved['closed_proof_bytes'],expected_proof_sha256=PROOF)
        post=saved['post_relu_hz']; original=source_digest(post); before=closed.fingerprint()
        owned=saved['phase_ownership']; overlay=Overlay(owned['base'],owned['events'],owned['old_uid_ceiling'])
        if (owned['post_hz'] is not post or owned['base'] is not closed.owners
                or original!=owned['post_sha256'] or hashlib.sha256(overlay.events.tobytes()).hexdigest()!=owned['event_sha256']):
            raise ValueError('actual phase/source/ownership binding drift')
        overlay.validate(); check_append(closed.hz,post)
        diagnostic=WorkPool(256_000_000)
        with (DIRECTORY/'events.jsonl').open('x') as stream:
            def emit(name,values):
                event={'event':name,**values,'worker_elapsed_s':time.monotonic()-started}
                stream.write(json.dumps(event)+'\n'); stream.flush(); print(json.dumps(event),flush=True)
            diagnostic.charge('complete_closed_UID_metadata',32*(len(closed.owners)+post.n_eq+post.n_ineq))
            eq,le=closed_uid_tables(closed); first=overlay.old_uid_ceiling
            ne,nl=post.n_eq-closed.hz.n_eq,post.n_ineq-closed.hz.n_ineq
            eq=np.r_[eq,np.arange(first,first+ne,dtype=np.int64)]
            le=np.r_[le,np.arange(first+ne,first+ne+nl,dtype=np.int64)]
            actual=incidence_oracle(post,eq,le,closed.old_n_cont,closed.logical_n_cont,pool=diagnostic)
            columns,phase=verify_all_and_discover(closed,post,overlay,actual,eq,le,
                whole=diagnostic,branch=BranchPool(diagnostic))
            emit('complete_actual_source_discovery',{'columns':len(columns),'work':diagnostic.used,**phase})
            plans=plans_from_checked_incidence(closed,post,columns,actual,eq,le,pool=diagnostic)
            # Independent completed reference is opened ONLY AFTER discovery.
            with (C15/'spliced_hz.pickle').open('rb') as handle: oracle_saved=pickle.load(handle)
            if oracle_saved['schema']!='c15_isolated_component_checkpoint_v1': raise ValueError('wrong exact reference')
            oracle=oracle_saved['candidate']; oracle.validate(); oracle_identity=oracle.fingerprint()
            compiler=WorkPool(256_000_000)
            draft,construction=measured_build(lambda:compile_lineage(closed.eq_roots,closed.eq_scales,plans,
                old_n_cont=closed.old_n_cont,old_n_eq=closed.old_n_eq,pool=compiler,enabled=True))
            if draft is None: raise ValueError('no complete transplant draft')
            record.update(compiler_work=compiler.used,compiler_work_parts=dict(compiler.parts),construction=construction)
            emit('functional_metadata_compiled',{'work':compiler.used,'parts':dict(compiler.parts),
                'columns':len(draft.columns),'retired_UIDs':len(draft.retired),'MAIN_tail_changes':len(draft.tails),
                'construction':construction})
            proof=verify(closed,post,overlay,columns,eq,le,draft,oracle,pool=diagnostic,
                expected_oracle_fingerprint=oracle_identity)
            if len(columns)!=268 or proof['new_phase_consumers']!=199 or proof['preexisting_consumers']!=69:
                raise ValueError('whole registered pair population differs')
            if closed.fingerprint()!=before or source_digest(post)!=original or oracle.fingerprint()!=oracle_identity:
                raise ValueError('source or independent reference mutated')
            record.update(proof=proof,independent_diagnostic_work=diagnostic.used,
                independent_diagnostic_work_parts=dict(diagnostic.parts),all_original_sources_unchanged=True)
            base=owned['report']['whole_generation_plus_event_work']
            copies=compiler.parts['functional_lineage_map_copies']
            sparse_entries=sum(len(getattr(draft,k)) for k in ('columns','retired','tails'))
            sparse_bytes=sum(getattr(draft,k).nbytes for k in ('columns','retired','tails'))
            assessment={'C25_generation_plus_event_work':base,'actual_functional_compiler_work':compiler.used,
                'naive_live_append_work':base+compiler.used,'naive_live_append_within_256M':base+compiler.used<=256_000_000,
                'map_copy_work_actually_paid':copies,'other_actual_compiler_work':compiler.used-copies,
                'copy_avoidance_is_implemented':False,'new_sparse_metadata_entries':sparse_entries,
                'new_sparse_metadata_bytes':sparse_bytes,'extra_offset_arrays':0,
                'old_full_owner_vector_copied':False,'parent_incidence_updates_performed':0,
                'known_MAIN_tail_UID_changes':len(draft.tails),'full_fresh_emission_discovery_and_resource_proof':False,
                'whole_live_path_proved':False,'formal_gain':0}
            record['integration_assessment']=assessment
            emit('complete_transplant_proof',proof)
            emit('integration_cost_assessment',assessment)
            path=DIRECTORY/'lineage.pickle'
            with path.open('xb') as handle:
                pickle.dump({'schema':'c26_provisional_transplant_lineage_v1','draft':draft,
                    'source_post_HZ_sha256':original,'closed_identity':before,'oracle_fingerprint':oracle_identity,
                    'source_archive_sha256':_sha256(LIVE/'relu78.pickle'),'proof':proof,
                    'provenance':freeze['provenance'],'formal_gain':0,'live_admission_certificate':False},handle,protocol=5)
                handle.flush(); os.fsync(handle.fileno())
            record.update(completed=True,checkpoint_sha256=_sha256(path),checkpoint_bytes=path.stat().st_size)
    except Exception as exc:
        record['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'result.json',record)
        print(json.dumps({'completed':record['completed'],'failure':record.get('failure'),'formal_gain':0}),flush=True)
    if not record['completed']: raise SystemExit(1)


if __name__=='__main__': main()
