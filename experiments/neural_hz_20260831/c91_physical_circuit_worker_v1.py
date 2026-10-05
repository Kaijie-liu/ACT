"""Complete C69 physical source comparator and exact current-circuit closure."""
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import make,audit
from experiments.neural_hz_20260831.c91_physical_archive_v1 import bind
from experiments.neural_hz_20260831.c65_physical_archive_v1 import check_restored,metadata
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c91_physical_circuit_20260913_v1'
SOURCE=EXP/'results/c69_prepared_finite_20260913_v1/actual/physical_hz.pickle'
SOURCE_SHA='acb6e5560503d42aa7b62fa4785f179f857471a9873b6d11189da88deea2db79'
C90=EXP/'results/c90_actual_circuit_proof_20260913_v1';C89=EXP/'results/c89_quotient_budget_20260913_v1'
C90_SHA='40d3a498200cc5748733dfa081040a048edae8eae9c3442a8477cb0ed3f67e84'
PACKET_SHA='f1e760a6bab4e722ce8cc8f1c76c823cddafe512b1aa836a126d53af9661c351'


def complete(pool,branch,held,emit):
    if (_sha256(C90/'result.json')!=C90_SHA or _sha256(C90/'all_rebased_selected_native_packets.npz')!=PACKET_SHA
        or _sha256(C89/'result.json')!='f83db6d09f255a1bdc9632bcb7e905d78eec9dae67bc3ae924217cd90aefabc1'):
        raise ValueError('complete circuit/global-plan proof changed')
    raw=(C90/'result.json').read_bytes();c90=json.loads(raw);c89=json.loads((C89/'result.json').read_text())
    if not c90['completed'] or not c89['completed']:raise ValueError('complete independent actual source proofs required')
    with SOURCE.open('rb') as stream:source,decoder=load(stream,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    with np.load(C90/'all_rebased_selected_native_packets.npz',allow_pickle=False) as ar:arrays={k:ar[k] for k in ar.files}
    source_layout=numeric_layout(source,pool);pool.charge('c91_complete_original_source_authentication',int(source_layout.resident_entries)+1024)
    old_proof=check_restored(source);fields=source['fields']
    if (old_proof['physical_identity']!='9e8c3ddcbe09ebf2bdc97a87379beecebb659e3549ede4d10f802b53eba94349'
        or old_proof['proof']['local_inverse_equations']!=100965
        or old_proof['source_checkpoint_sha256']!=c90['data']['decoder']['checkpoint_sha256']):
        raise ValueError('complete current C69/circuit original source binding differs')
    old_view=dict(fields=fields,checked_proof_record=(old_proof['physical_identity'],source['proof_bytes']))
    inputs=dict(complete_C69_source=old_view,complete_C90_native=arrays,c89_global_plan=c89,c90_source_proof=raw)
    held['complete_inputs']=inputs
    layout=numeric_layout(inputs,pool);old_layout=numeric_layout(old_view,pool);old_meta=metadata(old_view,pool=pool)
    if (old_layout.resident_entries!=20973560 or old_layout.resident_bytes!=245318484
        or layout.resident_entries>64_000_000):raise ValueError('same complete C69 physical comparator differs')
    pool.charge('c91_complete_input_before_after_fingerprints',2*int(layout.resident_entries)+2048)
    before,_=fingerprint(inputs,layout,pool,already_paid=True)
    try:
        circuits=[]
        for rec in c90['data']['all_block_proofs']:
            item=c89['data']['all_records'][rec['position']]
            if item['composition']['changed_old_coordinate_occurrences']!=0:
                raise ValueError('existing scalar choices would require new binding')
            prefix=f"node{rec['node']}_tile{rec['y']}_{rec['x']}_"
            p={k:arrays[prefix+k] for k in ('indptr','columns','native','rhs','pivots','gauges','ab_indptr')}
            p['new_factors']=rec['proof']['all_auxiliary_equations_and_redundant_boxes_proved'];circuits.append(p)
            pivots=p['pivots'][p['new_factors']:];roots=fields['eq_roots'][fields['old_n_eq']+pivots-fields['old_n_cont']]
            if np.any(roots<0):raise ValueError('old scalar projection removed a planned original output')
            count=int((fields['hz'].Ac.indptr[roots+1]-fields['hz'].Ac.indptr[roots]).sum())
            if count!=item['actual_source_quotient_nnz']:raise ValueError('actual current source direct row incidence differs')
        if len(circuits)!=len(c89['data']['plan']['selected_positions']):raise ValueError('complete proved plan differs')
        n=sum(p['new_factors'] for p in circuits);h=fields['hz']
        # Complete input plus conservative fresh Ac/RHS/pointers/maps/owners.
        envelope=layout.resident_entries+2*h.Ac.nnz+8*(h.n_eq+n)+16*h.n_cont+16*n
        if envelope>64_000_000:raise MemoryError('complete source/candidate preallocation entry envelope exceeds cap')
        emit(dict(event='complete_current_source_circuit_bound',input_entries=layout.resident_entries,
            input_bytes=layout.resident_bytes,preallocation_entry_upper=int(envelope),whole_work=pool.used))
        candidate=make(fields,circuits,source['proof_bytes'],raw,pool=branch,enabled=True)
        held['complete_candidate']=candidate
        proof=audit(fields,candidate,circuits,pool=branch,enabled=True)
        proof.update(original_source_archive_sha256=SOURCE_SHA,original_circuit_proof_sha256=C90_SHA,
            complete_source_equivalence_and_universal_box_proof_reused=True,
            original_scalar_identity=old_proof['proof']['complete_source_boundary_sha256'])
        emit(dict(event='complete_actual_source_rows_owners_inverse_maps',proof=proof,whole_work=pool.used,branch_work=branch.used))
        preliminary=numeric_layout(candidate,pool);pool.charge('c91_complete_new_state_proof_fingerprint',int(preliminary.resident_entries)+1024)
        identity,proof_raw=bind(candidate,proof)
        new_view=dict(state=candidate,checked_proof_record=(identity,proof_raw))
        new_layout=numeric_layout(new_view,pool);new_meta=metadata(new_view,pool=pool)
        union=numeric_layout(dict(inputs=inputs,complete_new_state=new_view),pool)
        physical=dict(comparator='complete current C69 source, not C9 or component-only',
            before_numeric_bytes=old_layout.resident_bytes,after_numeric_bytes=new_layout.resident_bytes,
            before_entries=old_layout.resident_entries,after_entries=new_layout.resident_entries,
            before_known_metadata_bytes=old_meta['nonoverlapping_known_metadata_bytes'],
            after_known_metadata_bytes=new_meta['nonoverlapping_known_metadata_bytes'],
            complete_union_entries=union.resident_entries,complete_union_bytes=union.resident_bytes,
            numeric_byte_delta=new_layout.resident_bytes-old_layout.resident_bytes,
            entry_delta=new_layout.resident_entries-old_layout.resident_entries,
            full_LIVE_gate_proved=False,python_allocator_occupancy_not_measured=True)
        physical['known_byte_delta']=physical['numeric_byte_delta']+physical['after_known_metadata_bytes']-physical['before_known_metadata_bytes']
        emit(dict(event='complete_current_source_physical_comparison',physical=physical,whole_work=pool.used))
        if (physical['numeric_byte_delta']>=0 or physical['entry_delta']>=0 or physical['known_byte_delta']>=0
            or union.resident_entries>64_000_000):raise ValueError('complete current-source physical gate failed')
        pool.charge('c91_complete_exclusive_physical_archive',int(new_layout.resident_entries)+16384)
        payload=dict(schema='c91_complete_physical_circuit_archive_v1',state=candidate,
            proof_bytes=proof_raw,proof_sha256=hashlib.sha256(proof_raw).hexdigest())
        path=RUN/'complete_physical_circuit.pickle'
        with path.open('xb') as out:pickle.dump(payload,out,protocol=5);out.flush();os.fsync(out.fileno())
        return dict(proof=proof,physical=physical,physical_identity=identity,archive_sha256=_sha256(path),
            archive_bytes=path.stat().st_size,source_decoder=decoder,
            complete_input_entries=layout.resident_entries,complete_input_bytes=layout.resident_bytes,
            source_HZ_constructed=True,full_LIVE_gate_proved=False,fresh_generation_work_proved=False,
            native_solver_ingested=False,solver_executed=False,formal_gain=0)
    finally:
        unchanged=fingerprint(inputs,layout,pool,already_paid=True)[0]==before
        emit(dict(event='complete_original_inputs_preserved',unchanged=unchanged))
        if not unchanged or _sha256(SOURCE)!=SOURCE_SHA:raise ValueError('complete original current-source inputs changed')


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));started=time.monotonic()
    pool=WorkPool(256_000_000);branch=BranchPool(pool);held={};result=dict(completed=False,formal_gain=0)
    with (RUN/'events.jsonl').open('x') as log,(RUN/'fatal.log').open('x') as fatal:
        def emit(v):log.write(json.dumps(dict(v,worker_elapsed_s=time.monotonic()-started),allow_nan=False)+'\n');log.flush()
        try:
            faulthandler.enable(file=fatal,all_threads=True)
            data,stats=measured(lambda:complete(pool,branch,held,emit),observe=lambda s:result.update(measurement=s))
            result.update(completed=True,data=data)
        except Exception as exc:
            result['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='complete_physical_realization_failed',**result['failure']))
        finally:
            faulthandler.disable();result.update(wall_s=time.monotonic()-started,whole_work=pool.used,branch_work=branch.used,
                work_parts=pool.parts,branch_work_parts=branch.parts)
            _atomic_exclusive_json(RUN/'result.json',result);print(json.dumps({k:result[k] for k in ('completed','wall_s','whole_work','branch_work','formal_gain')}),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
