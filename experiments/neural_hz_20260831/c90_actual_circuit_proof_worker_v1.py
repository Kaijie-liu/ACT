"""Independent real source equations with complete original diagnostic custody."""
import faulthandler
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c90_actual_circuit_proof_v1 import rebase,prove
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c90_actual_circuit_proof_20260913_v1'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
C89=EXP/'results/c89_quotient_budget_20260913_v1'
ARCHIVES={
    'C87':('results/c87_masked_tile_20260913_v1/all_actual_source_tiles_and_transforms.npz','a634b1becc25ae8b33e22246334200c29f6e3a491728a73861ed4a4087c0c3e1'),
    'C88':('results/c88_inline_tile_20260913_v1/all_native_tile_packets.npz','abedd1d085e04e9140960923d6d13f46e34ebb11a2163a2329d27d6c3ae04e75'),
    'C89':('results/c89_quotient_budget_20260913_v1/all_mapped_native_packets.npz','3e60eeb4e16c24e1d8fbd1c7bd26376612f166c64f4b5efb612ccb1f8668eaa6')}


def complete(pool,branch,held,emit):
    if _sha256(C89/'result.json')!='f83db6d09f255a1bdc9632bcb7e905d78eec9dae67bc3ae924217cd90aefabc1':
        raise ValueError('complete C89 actual quotient and plan changed')
    previous=json.loads((C89/'result.json').read_text())
    if not previous['completed']:raise ValueError('complete C89 prerequisite required')
    inputs={}
    for name,(relative,sha) in ARCHIVES.items():
        path=EXP/relative
        if _sha256(path)!=sha:raise ValueError('complete prior numerical archive changed')
        with np.load(path,allow_pickle=False) as archive:inputs[name]={k:archive[k] for k in archive.files}
    with SOURCE.open('rb') as stream:saved,decoder=load(stream,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    inputs['C9']=saved;held['complete_inputs']=inputs
    if (not saved['identity_audit']['all_original_coefficients_exact']
        or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']):
        raise ValueError('complete original source/box theorem required')
    layout=numeric_layout(inputs,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete original input entry cap')
    pool.charge('c90_complete_input_before_after_fingerprints',2*int(layout.resident_entries)+2048)
    before,shallow=fingerprint(inputs,layout,pool,already_paid=True)
    try:
        data=previous['data'];plan=data['plan']
        if (data['old_quotient_report']['identity_sha256']!='2174cc144bcfe499ab00e97f67eae2b72d8a3c3d6f994780f63e349d00f36770'
            or data['old_quotient_report']['optimum']!=100965 or plan['whole_auxiliary_reserve_used']>16384
            or plan['whole_declared_emission_reserve_used']>16_000_000):
            raise ValueError('complete unchanged scalar/global plan proof differs')
        emit(dict(event='complete_all_archive_custody',entries=layout.resident_entries,bytes=layout.resident_bytes,whole_work=pool.used))
        records=[];rebased={};held['new_rebased_packets']=rebased
        base=saved['hz'].n_cont;offset=0;seen_outputs=set()
        for position in plan['selected_positions']:
            item=data['all_records'][position]
            if not item['mapped_native_pass'] or item['composition']['changed_old_coordinate_occurrences']:
                raise ValueError('this actual theorem requires the C89-proved unchanged original-coordinate image')
            prefix=f"node{item['node']}_tile{item['y']}_{item['x']}_"
            packet={k:inputs['C89'][prefix+k] for k in ('indptr','columns','native','rhs','pivots','gauges','ab_indptr')}
            branch.charge('c90_complete_existing_circuit_image_binding',2*sum(int(v.size) for v in packet.values())+128)
            for key in ('indptr','columns','native','rhs','pivots','gauges'):
                if not np.array_equal(packet[key],inputs['C88'][prefix+key]):
                    raise ValueError('C89 unchanged actual coefficient image differs from C88')
            n=item['composition']['new_factors'];pivots=packet['pivots'][n:]
            if (len(set(map(int,pivots)))!=len(pivots) or set(map(int,pivots))&seen_outputs):
                raise ValueError('one plan cannot replace an output twice')
            seen_outputs.update(map(int,pivots))
            logical=saved['old_n_eq']+pivots-saved['old_n_cont']
            physical=saved['eq_roots'][logical];gauges=saved['eq_scales'][logical]
            if not np.array_equal(physical,inputs['C89'][prefix+'original_physical_rows']):
                raise ValueError('original full physical row routing changed')
            original=[]
            for r,pivot in zip(physical,pivots,strict=True):
                r=int(r);hz=saved['hz'];a,b=map(int,hz.Ac.indptr[r:r+2])
                branch.charge('c90_original_source_literal_snapshot',32+8*(b-a))
                if hz.Ab.indptr[r+1]!=hz.Ab.indptr[r]:raise ValueError('original binary predicate cannot be replaced')
                original.append(dict(pivot=int(pivot),rhs=float(hz.b[r]),
                    coefficients=tuple(zip(map(int,hz.Ac.indices[a:b]),map(float,hz.Ac.data[a:b])))))
            native=rebase(packet,base,offset,pool=branch)
            proof=prove(native,original,gauges,old_n_cont=base,first_aux=base+offset,new_factors=n,pool=branch,enabled=True)
            record=dict(position=position,node=item['node'],y=item['y'],x=item['x'],first_global_aux=base+offset,proof=proof)
            records.append(record);offset+=n
            for key,value in native.items():rebased[prefix+key]=value
            emit(dict(event='complete_actual_block_source_theorem',**record,whole_work=pool.used,branch_work=branch.used))
        if offset+28!=plan['whole_auxiliary_reserve_used']:raise ValueError('complete global new-factor namespace differs')
        combined=numeric_layout(dict(inputs=inputs,rebased=rebased),pool)
        if combined.resident_entries>64_000_000:raise MemoryError('complete rebased plan entry cap')
        pool.charge('c90_exclusive_complete_rebased_packet_export',sum(int(v.size) for v in rebased.values())+4096)
        with (RUN/'all_rebased_selected_native_packets.npz').open('xb') as out:np.savez(out,**rebased)
        return dict(all_block_proofs=records,complete_original_output_rows=len(seen_outputs),
            complete_new_factors=offset,complete_global_n_cont=base+offset,
            original_scalar_quotient_reused_under_exact_hash=True,original_scalar_factors_restored=0,
            source_equation_equivalence_proved=True,universal_redundant_box_inverse_proved=True,
            complete_input_entries=layout.resident_entries,complete_input_bytes=layout.resident_bytes,
            complete_input_and_rebased_entries=combined.resident_entries,complete_input_and_rebased_bytes=combined.resident_bytes,
            input_shallow_bytes=shallow,decoder=decoder,full_physical_HZ_constructed=False,
            full_LIVE_physical_gate_proved=False,fresh_runtime_payment_proved=False,
            native_solver_ingested=False,solver_executed=False,concrete_network_witness=False,formal_gain=0)
    finally:
        unchanged=fingerprint(inputs,layout,pool,already_paid=True)[0]==before
        emit(dict(event='complete_original_inputs_preserved',unchanged=unchanged))
        if not unchanged or _sha256(SOURCE)!=SOURCE_SHA:raise ValueError('complete original source inputs changed')


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();pool=WorkPool(256_000_000);branch=BranchPool(pool);held={};result=dict(completed=False,formal_gain=0)
    with (RUN/'events.jsonl').open('x') as log,(RUN/'fatal.log').open('x') as fatal:
        def emit(v):log.write(json.dumps(dict(v,worker_elapsed_s=time.monotonic()-started),allow_nan=False)+'\n');log.flush()
        try:
            faulthandler.enable(file=fatal,all_threads=True)
            data,stats=measured(lambda:complete(pool,branch,held,emit),observe=lambda s:result.update(measurement=s))
            result.update(completed=True,data=data)
        except Exception as exc:
            result['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='actual_source_proof_failed',**result['failure']))
        finally:
            faulthandler.disable()
            result.update(wall_s=time.monotonic()-started,whole_work=pool.used,branch_work=branch.used,work_parts=pool.parts,branch_work_parts=branch.parts)
            _atomic_exclusive_json(RUN/'result.json',result);print(json.dumps({k:result[k] for k in ('completed','wall_s','whole_work','branch_work','formal_gain')}),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
