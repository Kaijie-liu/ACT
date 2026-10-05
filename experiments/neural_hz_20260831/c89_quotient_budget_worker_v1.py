"""Complete immutable source/packet inputs, exact old quotient and global plan."""
import faulthandler
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c89_quotient_budget_v1 import lower,direct_count,bill,select
from experiments.neural_hz_20260831.c88_inline_tile_v1 import NativeUnproved
from experiments.neural_hz_20260831.c63_precision_plan_v1 import plan
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c89_quotient_budget_20260913_v1'
C87=EXP/'results/c87_masked_tile_20260913_v1/all_actual_source_tiles_and_transforms.npz'
C88=EXP/'results/c88_inline_tile_20260913_v1'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
PACKET_SHA='abedd1d085e04e9140960923d6d13f46e34ebb11a2163a2329d27d6c3ae04e75'
OLD_ID='2174cc144bcfe499ab00e97f67eae2b72d8a3c3d6f994780f63e349d00f36770'


def complete(pool,branch,held,emit):
    if (_sha256(C87)!='a634b1becc25ae8b33e22246334200c29f6e3a491728a73861ed4a4087c0c3e1'
        or _sha256(C88/'all_native_tile_packets.npz')!=PACKET_SHA
        or _sha256(C88/'result.json')!='a8d8d46ee6d022c5a1b04d76ec022b011ec46ad04b9b9ece44882e0de7027be0'):
        raise ValueError('complete actual C87/C88 proof input changed')
    previous=json.loads((C88/'result.json').read_text())
    if not previous['completed']:raise ValueError('complete source circuit diagnostic required')
    with np.load(C87,allow_pickle=False) as ar:transforms={n:ar[n] for n in ar.files}
    with np.load(C88/'all_native_tile_packets.npz',allow_pickle=False) as ar:packets={n:ar[n] for n in ar.files}
    with SOURCE.open('rb') as stream:saved,decoder=load(stream,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    inputs=dict(complete_C9=saved,complete_C87=transforms,complete_C88=packets)
    held['complete_inputs']=inputs
    if (saved['schema']!='c9_integrated_suffix_checkpoint_v1'
        or not saved['identity_audit']['all_original_coefficients_exact']
        or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']):
        raise ValueError('complete original source/box theorem required')
    layout=numeric_layout(inputs,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete original/circuit input entry cap')
    pool.charge('c89_complete_input_before_after_fingerprints',2*int(layout.resident_entries)+2048)
    before,shallow=fingerprint(inputs,layout,pool,already_paid=True)
    try:
        quotient=plan(saved,pool=branch,enabled=True)
        if quotient['report']['identity_sha256']!=OLD_ID or quotient['report']['optimum']!=100965:
            raise ValueError('complete existing C69 scalar quotient differs')
        emit(dict(event='complete_original_quotient_bound',old_eliminations=100965,input_entries=layout.resident_entries,
            input_bytes=layout.resident_bytes,whole_work=pool.used,branch_work=branch.used))
        records=[];lowered={};held['new_native_packets']=lowered
        for node in previous['data']['all_operator_records']:
            node_id=node['node']
            for item in node['all_tiles']:
                rec=dict(node=node_id,y=item['y'],x=item['x'],raw_nnz_win=item['strict_nnz_win'],mapped_native_pass=False)
                records.append(rec)
                if not item['strict_nnz_win']:
                    rec['rejection']='uniform raw-nnz prefilter'
                    continue
                prefix=f"node{node_id}_tile{item['y']}_{item['x']}_"
                packet={key:packets[prefix+key] for key in ('indptr','columns','words','powers','native','pivots','pivot_powers','gauges','rhs')}
                try:
                    native,info=lower(item,packet,quotient['roots'],quotient['weights'],pool=branch,enabled=True)
                    pivots=packet['pivots'][item['new_factors']:]
                    if np.any(pivots<saved['old_n_cont']) or np.any(pivots>=saved['logical_n_cont']):
                        raise ValueError('original MAIN output coordinate binding differs')
                    physical=saved['eq_roots'][saved['old_n_eq']+pivots-saved['old_n_cont']]
                    original_count=int((saved['hz'].Ac.indptr[physical+1]-saved['hz'].Ac.indptr[physical]).sum())
                    if original_count!=item['direct_nnz']:
                        raise NativeUnproved('complete original direct rows have different radix incidence')
                    direct=direct_count(saved['hz'],physical,pivots,quotient['roots'],quotient['weights'],pool=branch)
                    rec.update(mapped_native_pass=True,composition=info,bill=bill(native,new_factors=item['new_factors'],direct_nnz=direct),
                        actual_source_quotient_nnz=direct,old_scalar_factors_restored=0)
                    for key,value in native.items():lowered[prefix+key]=value
                    lowered[prefix+'original_physical_rows']=physical.copy()
                except NativeUnproved as exc:
                    rec['rejection']=str(exc)
            emit(dict(event='complete_node_quotient_composition',node=node_id,
                mapped=sum(r['mapped_native_pass'] for r in records if r['node']==node_id),whole_work=pool.used,branch_work=branch.used))
        if len(records)!=previous['data']['tiles']:raise ValueError('complete tile population differs')
        chosen=select(records,existing_aux=28,existing_work=593353,existing_entries=252,pool=branch,enabled=True)
        combined=numeric_layout(dict(inputs=inputs,new_native=lowered),pool)
        if combined.resident_entries>64_000_000:raise MemoryError('complete input/new native entry cap')
        pool.charge('c89_complete_native_plan_export',sum(int(v.size) for v in lowered.values())+64*len(records))
        with (RUN/'all_mapped_native_packets.npz').open('xb') as stream:np.savez(stream,**lowered)
        return dict(all_records=records,plan=chosen,old_quotient_report=quotient['report'],
            complete_input_entries=layout.resident_entries,complete_input_bytes=layout.resident_bytes,
            complete_input_shallow_bytes=shallow,complete_input_and_new_entries=combined.resident_entries,
            complete_input_and_new_bytes=combined.resident_bytes,decoder=decoder,
            actual_mapped_native_tiles=sum(r['mapped_native_pass'] for r in records),
            old_eliminations=100965,old_scalar_factors_restored=0,
            fresh_complete_actual_circuit_source_theorem=False,full_HZ_constructed=False,
            native_solver_ingested=False,solver_executed=False,formal_gain=0)
    finally:
        unchanged=fingerprint(inputs,layout,pool,already_paid=True)[0]==before
        emit(dict(event='complete_original_inputs_unchanged',unchanged=unchanged))
        if not unchanged or _sha256(SOURCE)!=SOURCE_SHA or _sha256(C88/'all_native_tile_packets.npz')!=PACKET_SHA:
            raise ValueError('complete original inputs changed')


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();pool=WorkPool(256_000_000);branch=BranchPool(pool);held={}
    result=dict(completed=False,formal_gain=0)
    with (RUN/'events.jsonl').open('x') as log,(RUN/'fatal.log').open('x') as fatal:
        def emit(v):
            log.write(json.dumps(dict(v,worker_elapsed_s=time.monotonic()-started),allow_nan=False)+'\n');log.flush()
        try:
            faulthandler.enable(file=fatal,all_threads=True)
            data,stats=measured(lambda:complete(pool,branch,held,emit),observe=lambda s:result.update(measurement=s))
            result.update(completed=True,data=data)
        except Exception as exc:
            result['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='diagnostic_failed',**result['failure']))
        finally:
            faulthandler.disable()
            result.update(wall_s=time.monotonic()-started,whole_work=pool.used,branch_work=branch.used,
                work_parts=pool.parts,branch_work_parts=branch.parts)
            _atomic_exclusive_json(RUN/'result.json',result)
            print(json.dumps({k:result[k] for k in ('completed','wall_s','whole_work','branch_work','formal_gain')}),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
