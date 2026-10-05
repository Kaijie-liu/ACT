"""One fresh original-expression HZ, full source proof and physical comparison."""
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
import resource
import sys
import time
import tracemalloc
import weakref
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c66_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c65_full_source_audit_v1 import audit
from experiments.neural_hz_20260831.c65_physical_archive_v1 import metadata,bind_proof,check_restored
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import rss_bytes
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c66_packed_frontier_20260913_v1';DIRECTORY=RUN/'actual'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
OLD=EXP/'results/c31_prepared_generator_20260911_v1/closed_hz.pickle'
OLD_SHA='535b579e853f1ac5092e962e1c307644e89739da4cd63f3732593896649128f5'
OLD_PROOF='cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5'


def complete(pool,branch,emit,record):
    input_binding=json.loads((DIRECTORY/'input_binding.json').read_text())
    if (_sha256(RUN/'preregistered.json')!=input_binding['preregistered_sha256']
            or _sha256(RUN/'preflight/result.json')!=input_binding['complete_source_bound_sha256']):
        raise ValueError('frozen preflight/candidate binding changed')
    prior=json.loads((RUN/'preflight/result.json').read_text())
    if not prior['completed'] or not prior['data']['bound']['work_caps_fit']:raise ValueError('complete fitting source bound required')
    bound=prior['data']['bound']
    if _sha256(OLD)!=OLD_SHA:raise ValueError('original checked C31 archive changed')
    with OLD.open('rb') as stream:old_archive=pickle.load(stream)
    if (old_archive['schema']!='c31_new_checked_prepared_Closed_v1'
            or hashlib.sha256(old_archive['proof_bytes']).hexdigest()!=OLD_PROOF):raise ValueError('complete C31 source proof required')
    legacy=old_archive['fields']
    with SOURCE.open('rb') as stream:saved,decoder=load(stream,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    if (saved['schema']!='c9_integrated_suffix_checkpoint_v1'
            or saved['origin_snapshot_sha256']!='d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
            or not saved['identity_audit']['all_original_coefficients_exact']
            or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']):raise ValueError('complete original affine source/box proof required')
    old_view=dict(fields=legacy,checked_proof_record=(old_archive['identity']['closed_identity'],old_archive['proof_bytes']))
    inputs=dict(original_C9=saved,complete_C31=old_view);input_layout=numeric_layout(inputs,pool)
    if input_layout.resident_entries>64_000_000:raise MemoryError('complete offline input entry cap')
    pool.charge('c65_two_complete_offline_input_fingerprints',2*int(input_layout.resident_entries)+2048)
    before,_=fingerprint(inputs,input_layout,pool,already_paid=True)
    old_layout=numeric_layout(old_view,pool);old_meta=metadata(old_view,pool=pool)
    try:
        # The only generator inputs are the authenticated original expression,
        # keep mask and original shared widths. No planner/reference HZ enters.
        candidate,construction_stats=measured(lambda:lift(saved['expression'],saved['keep'],enabled=True,
            frame_widths=(saved['old_n_cont'],saved['old_n_bin']),observe=lambda n,v:emit(dict(event=n,**v))),
            observe=lambda s:emit(dict(event='actual_generator_measurement',measurement=s)))
        record.update(physical_HZ_constructed=True,new_generator_executed=True,generation=construction_stats)
        fields=candidate['fields'];generation=fields['report'];parts=generation['alias_quotient']['work_parts']
        record['generation_report']=generation
        if (generation['whole_base_work']!=bound['whole_base_work'] or generation['branch_base_work']!=bound['branch_base_work']
                or generation['total_work_upper']>bound['whole_work_upper']
                or generation['largest_branch_work_upper']>bound['branch_work_upper']
                or any(name not in bound['work_parts'] or value>bound['work_parts'][name] for name,value in parts.items())
                or generation['alias_quotient']['identity_sha256']!=bound['expected_identity_sha256']
                or generation['alias_quotient']['optimum']!=bound['selected']):raise ValueError('actual complete generator differs from frozen complete source bound')
        emit(dict(event='fresh_complete_generator_within_each_source_bound',whole_work=generation['total_work_upper'],
            branch_work=generation['largest_branch_work_upper'],generation=construction_stats))
        proof_start=time.monotonic();proof=audit(saved,legacy,candidate,pool=branch,enabled=True)
        if proof['complete_source_boundary_sha256']!=bound['expected_identity_sha256']:raise ValueError('full independent boundary proof differs')
        proof['owner_reference_checkpoint_sha256']=OLD_SHA;proof['owner_reference_proof_sha256']=OLD_PROOF
        record.update(complete_original_source_proof=proof,source_proof_wall_s=time.monotonic()-proof_start)
        emit(dict(event='full_original_source_inverse_owner_UID_proof',proof=proof,diagnostic_work=pool.used))
        refs=[weakref.ref(n[k]) for n in candidate['construction']['nodes'] for k in ('support','needed','slots','exponents')]
        del candidate;gc.collect()
        if any(ref() is not None for ref in refs):raise ValueError('complete new graph arrays remain after physical closure')
        # Complete standalone candidate fingerprint/publication costs are named
        # offline diagnostics, not silently appended to the construction cap.
        preliminary=numeric_layout(fields,pool)
        pool.charge('c65_complete_candidate_proof_fingerprint',int(preliminary.resident_entries)+1024)
        identity,raw=bind_proof(fields,proof,source_sha256=SOURCE_SHA)
        new_view=dict(fields=fields,checked_proof_record=(identity,raw))
        new_layout=numeric_layout(new_view,pool);new_meta=metadata(new_view,pool=pool)
        combined=numeric_layout(dict(offline_inputs=inputs,physical_candidate=new_view),pool)
        if combined.resident_entries>64_000_000:raise MemoryError('complete candidate/oracle union entry cap')
        physical=dict(old_numeric_bytes=old_layout.resident_bytes,new_numeric_bytes=new_layout.resident_bytes,
            old_entries=old_layout.resident_entries,new_entries=new_layout.resident_entries,
            old_known_metadata_bytes=old_meta['nonoverlapping_known_metadata_bytes'],
            new_known_metadata_bytes=new_meta['nonoverlapping_known_metadata_bytes'],
            all_expression_and_operator_metadata_traversed=True,opaque_identity_cancellation_used=False,
            python_allocator_occupancy_not_measured=True,combined_candidate_input_entries=combined.resident_entries)
        physical['numeric_byte_delta']=physical['new_numeric_bytes']-physical['old_numeric_bytes']
        physical['complete_known_byte_delta']=physical['numeric_byte_delta']+physical['new_known_metadata_bytes']-physical['old_known_metadata_bytes']
        record['physical']=physical;emit(dict(event='complete_same_C31_physical_boundary',physical=physical,diagnostic_work=pool.used))
        if (physical['numeric_byte_delta']>=0 or physical['complete_known_byte_delta']>=0
                or new_layout.resident_entries>=old_layout.resident_entries):raise ValueError('complete same-C31 physical state does not strictly decrease')
        pool.charge('c65_exclusive_complete_physical_archive',int(new_layout.resident_entries)+16384)
        payload=dict(schema='c65_graph_free_physical_archive_v1',fields=fields,proof_bytes=raw,
            proof_sha256=hashlib.sha256(raw).hexdigest(),native_or_LIVE_admission=False,formal_gain=0)
        path=DIRECTORY/'physical_hz.pickle'
        with path.open('xb') as stream:pickle.dump(payload,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
        # A later fresh-process restore also checks the externally bound file.
        return dict(proof=proof,physical=physical,physical_identity=identity,proof_sha256=payload['proof_sha256'],
            graph_arrays_physically_retired=len(refs),complete_offline_original_inputs_retained=True,
            archive_sha256=_sha256(path),archive_bytes=path.stat().st_size,decoder=decoder,
            complete_source_and_storage_proved=True,native_or_LIVE_admission=False,formal_gain=0)
    finally:
        unchanged=fingerprint(inputs,input_layout,pool,already_paid=True)[0]==before
        emit(dict(event='complete_original_input_preservation',unchanged=unchanged))
        record['complete_original_inputs_unchanged']=unchanged
        if not unchanged or _sha256(SOURCE)!=SOURCE_SHA or _sha256(OLD)!=OLD_SHA:raise ValueError('complete original source/oracles changed')


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);branch=BranchPool(pool);pool.charge('c65_exclusive_actual_record',16384)
    started=time.monotonic();report=dict(completed=False,formal_gain=0,physical_HZ_constructed=False,
        new_generator_executed=False,native_or_solver_executed=False,default_changed=False,whole_LIVE_path_proved=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):
            if value.get('event') in ('complete_preflight','encoded_node','c65_complete_owned_quotient'):
                value=dict(value,observed_rss_bytes=rss_bytes(),
                    observed_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    observed_trace=tracemalloc.get_traced_memory(),
                    observed_tracer_metadata_bytes=tracemalloc.get_tracemalloc_memory())
            if value.get('event')=='actual_generator_measurement':
                report['actual_builder_returned']=value['measurement']['build_returned']
            log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((RUN/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('complete frozen source drift')
            data=complete(pool,branch,emit,report);report.update(completed=True,data=data)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='actual_generator_or_qualification_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,offline_diagnostic_work=pool.used,offline_branch_work=branch.used,
                diagnostic_parts=dict(pool.parts),offline_branch_parts=dict(branch.parts),
                lifetime_rss_bytes_including_offline_inputs=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            _atomic_exclusive_json(DIRECTORY/'result.json',report);print(json.dumps(report),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
