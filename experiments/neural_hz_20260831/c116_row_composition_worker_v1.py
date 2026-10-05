# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete row-composition proof; all emitted artifacts survive a late failure."""
import faulthandler
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,metadata
from experiments.neural_hz_20260831.c88_inline_tile_v1 import actual_rows
from experiments.neural_hz_20260831.c116_row_composition_v1 import prove
from experiments.neural_hz_20260831.c115_atomic_word_plan_v1 import _plan_row,reduce_rows,PreparedAtomic,emit_tile
from experiments.neural_hz_20260831.run_c111_bound_accounting_v1 import screen
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;PREV=EXP/'results/c115_atomic_word_plan_20260913_v1'
RUN=EXP/'results/c116_row_composition_20260920_v1'
VIEW=EXP/'results/c114_joint_sink_census_20260913_v1/actual_diagnostic_views.pickle'
VIEW_SHA='22447cc49c93747a00b111eeedd41981891227b7866a1c96d4984168f466ab20'


def build(pool,held):
    auth=time.monotonic();binding=screen(pool,held)
    if _sha256(VIEW)!=VIEW_SHA:raise ValueError('complete frozen native view changed')
    with VIEW.open('rb') as stream:views=pickle.load(stream)
    held['complete_C114_views']=views
    originals=[*held['all_aux'],*held['all_outputs']]
    if views['original']['effective_native_rows']!=originals:raise ValueError('full C90/C114 original rows differ')
    aliased=views['sealed_root_aliases'];records=aliased['effective_native_rows']
    if aliased['report']['independently_checked_prior_root_aliases']!=len(held['aliases']):
        raise ValueError('complete prior root quotient population differs')
    auth_s=time.monotonic()-auth;base=held['all_aux'][0]['slot']
    aux_count=aliased['report']['effective_auxiliary_rows'];symbolic=[]
    for row in records:
        pool.charge('c115_complete_native_to_word_decode',64*len(row['coefficients'])+64)
        if row['rhs']!=0:raise ValueError('word producer requires zero auxiliary/output RHS in this scope')
        values={c:F(v) for c,v in row['coefficients']};pivot=values.pop(row['slot'])/F(2)**row['gauge']
        if (pivot<=0 or pivot.numerator&(pivot.numerator-1) or pivot.denominator&(pivot.denominator-1)):
            raise ValueError('ungauged exact dyadic pivot is not a positive power of two')
        power=pivot.numerator.bit_length()-pivot.denominator.bit_length()
        terms=[]
        for col,value in values.items():
            numerator,denominator=value.numerator,value.denominator
            if denominator&(denominator-1):raise ValueError('non-dyadic native value')
            terms.append((col,numerator,-(denominator.bit_length()-1)-row['gauge']))
        symbolic.append(_plan_row(terms,row['slot'],power,pool=pool))
    m_slots={r['pivot'] for r in symbolic[:aux_count]
             if any(c>=base and c!=r['pivot'] for c in r['columns'])}
    start=pool.used
    state=reduce_rows(symbolic,aux_count,m_slots,base,len(held['aliases']),pool=pool)
    labels=[r['pivot'] for r in state['rows'][:state['new_factors']]]
    state['source_report']=dict(reused_v=len(held['aliases']),kept_m=len(m_slots)-len(state['removed_m']))
    ready=PreparedAtomic(state,pool,base,dict(new_factors=state['new_factors']))
    report,packet=emit_tile(ready,base,pool=pool,enabled=True)
    reducer_work=pool.used-start
    after_aux,after_out=actual_rows(report,packet)
    held.update(symbolic_native_input=symbolic,candidate_packet=packet,
        candidate_rows=dict(aux=after_aux,outputs=after_out),complete_plan_decisions=report)

    # Persist the exact candidate BEFORE independent proof. This receipt is
    # deliberately never overwritten by the later mathematical proof result.
    coordinate_map=dict(old_kept_slots=np.array(labels,np.int64),
        new_slots=base+np.arange(len(labels),dtype=np.int64))
    held['complete_coordinate_map']=coordinate_map
    with (RUN/'candidate_complete_packet.npz').open('xb') as stream:
        np.savez(stream,**packet);stream.flush();os.fsync(stream.fileno())
    with (RUN/'complete_coordinate_map.npz').open('xb') as stream:
        np.savez(stream,**coordinate_map);stream.flush();os.fsync(stream.fileno())
    _atomic_exclusive_json(RUN/'candidate_UNPROVED.json',dict(status='UNPROVED',
        packet_sha256=_sha256(RUN/'candidate_complete_packet.npz'),
        coordinate_map_sha256=_sha256(RUN/'complete_coordinate_map.npz'),
        plan=report,source_or_LIVE_admitted=False,formal_gain=0))
    proof_start=time.monotonic();proof_work_start=pool.used
    proof,inverse=prove(held['all_aux'],held['all_outputs'],after_aux,after_out,
        base,labels,held['aliases'],pool=pool,enabled=True)
    proof_s=time.monotonic()-proof_start;proof_work=pool.used-proof_work_start
    if not proof['strict_total_nnz_decrease']:
        raise ValueError('complete original/candidate total nnz is not strictly reduced')
    held.update(complete_row_proof=proof,complete_original_factor_inverse=inverse)
    with (RUN/'complete_exact_inverse.pickle').open('xb') as stream:
        pickle.dump(inverse,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    _atomic_exclusive_json(RUN/'complete_native_row_proof.json',dict(proof=proof,
        inverse_sha256=_sha256(RUN/'complete_exact_inverse.pickle'),
        original_source_identity=binding['source_identity'],
        candidate_packet_sha256=_sha256(RUN/'candidate_complete_packet.npz'),
        coordinate_map_sha256=_sha256(RUN/'complete_coordinate_map.npz'),
        scope='native_equation_equivalence_only_pending_physical_and_source_admission'))
    layout=numeric_layout(held,pool)
    whole_metadata=metadata(held,pool);inverse_metadata=metadata(inverse,pool)
    if layout.resident_entries>64_000_000:
        raise MemoryError('complete native proof numeric entries exceed64M')
    original_nnz=proof['original_nnz']
    removed=len(held['all_aux'])-report['new_factors'];delta=report['nnz']-original_nnz
    return dict(report=report,proof=proof,original_auxiliary_rows=len(held['all_aux']),
        original_output_rows=len(held['all_outputs']),original_nnz=original_nnz,
        new_auxiliary_rows=report['new_factors'],new_nnz=report['nnz'],
        removed_V=len(held['aliases']),removed_M=report['removed_m'],
        total_removed_factors=removed,nnz_delta=delta,
        conditional_packet_byte_delta_formula=12*delta-88*removed,
        conditional_packet_entry_delta_formula=2*delta-13*removed,
        original_root_binding=binding['source_identity'],authenticated_decode_s=auth_s,
        reducer_and_final_emission_work=reducer_work,complete_proof_s=proof_s,
        complete_proof_work=proof_work,complete_numeric_bytes=layout.resident_bytes,
        complete_numeric_entries=layout.resident_entries,complete_known_metadata=whole_metadata,
        complete_inverse_metadata=inverse_metadata,
        original_kernel_coordinate_source_producer_executed=False,
        original_real_HZ_numeric_roots_present=False,original_HZ_loaded_into_live_target=False,
        complete_original_binary_predicate_numeric_roots_present=False,
        no_old_source_identity_attached_to_changed_state=True,new_source_or_LIVE_admission=False,
        complete_owned_source_physical_gate_proved=False,generation_budget_proved=False,formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((RUN/'preregistered.json').read_text());pool=WorkPool(256_000_000)
    record=dict(completed=False,formal_gain=0);held={};start=time.monotonic()
    fatal=(RUN/'fatal.log').open('x');faulthandler.enable(file=fatal,all_threads=True)
    try:
        if any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()):
            raise ValueError('full source drift')
        data,measurement=measured(lambda:build(pool,held),observe=lambda m:record.update(measurement=m))
        record.update(completed=True,data=data,measurement=measurement)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        faulthandler.disable();fatal.close()
        record.update(work=pool.used,work_parts=pool.parts,wall_s=time.monotonic()-start,
            source_drift=any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='data'}),flush=True)
    if not record['completed'] or record['source_drift']:raise SystemExit(1)


if __name__=='__main__':main()

