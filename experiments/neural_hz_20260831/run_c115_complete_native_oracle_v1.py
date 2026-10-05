# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete archived native mathematical oracle, explicitly not source loading."""
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
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c88_inline_tile_v1 import actual_rows
from experiments.neural_hz_20260831.c114_joint_sink_census_v1 import bounded
from experiments.neural_hz_20260831.c115_atomic_word_plan_v1 import _plan_row,reduce_rows,PreparedAtomic,emit_tile
from experiments.neural_hz_20260831.run_c111_bound_accounting_v1 import screen
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;PREV=EXP/'results/c115_atomic_word_plan_20260913_v1'
RUN=EXP/'results/c115_complete_native_oracle_20260913_v1'
VIEW=EXP/'results/c114_joint_sink_census_20260913_v1/actual_diagnostic_views.pickle'
VIEW_SHA='22447cc49c93747a00b111eeedd41981891227b7866a1c96d4984168f466ab20'


def projected(aux,outputs,base,*,pool):
    """Independent sparse original-coordinate oracle; no word-plan helpers."""
    memo={};out=[]
    for row in [*aux,*outputs]:
        slot=row['slot'];entries={c:bounded(v) for c,v in row['coefficients']}
        if len(entries)!=len(row['coefficients']):raise ValueError('oracle row is not coalesced')
        rhs=bounded(row['rhs']);is_aux=slot>=base
        pivot=entries.pop(slot) if is_aux else F(1)
        if is_aux and (pivot<=0 or any(c>=slot for c in entries)
                or abs(rhs)+sum(map(abs,entries.values()),F(0))>pivot):
            raise ValueError('complete positive triangular redundant-box theorem failed')
        count=sum(len(memo[c]) if c>=base else 1 for c in entries)
        pool.charge('c115_independent_complete_polynomial_projection',64*count+64)
        poly={-1:rhs/pivot if is_aux else -rhs}
        for col,value in entries.items():
            coefficient=-value/pivot if is_aux else value
            for key,term in (memo[col].items() if col>=base else ((col,F(1)),)):
                poly[key]=bounded(poly.get(key,F(0))+coefficient*term)
        poly={c:v for c,v in poly.items() if v}
        if is_aux:memo[slot]=poly
        else:
            gauge=F(2)**row['gauge']
            out.append({c:bounded(v/gauge) for c,v in poly.items()})
    return memo,out


def encode_polynomials(value,*,pool):
    if type(value) is dict:
        result={}
        for key,poly in value.items():
            pool.charge('c115_complete_rational_proof_serialization',48*len(poly))
            result[key]=[(c,v.numerator,v.denominator) for c,v in sorted(poly.items())]
        return result
    result=[]
    for poly in value:
        pool.charge('c115_complete_rational_proof_serialization',48*len(poly))
        result.append([(c,v.numerator,v.denominator) for c,v in sorted(poly.items())])
    return result


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
    before_memo,before_out=projected(held['all_aux'],held['all_outputs'],base,pool=pool)
    after_memo,after_poly=projected(after_aux,after_out,base,pool=pool)
    pool.charge('c115_every_original_output_and_kept_factor_comparison',
        32*(sum(len(p) for p in before_out)+sum(len(after_memo[base+i]) for i in range(len(labels)))))
    if before_out!=after_poly:raise ValueError('full original-coordinate output polynomials differ')
    for i,old in enumerate(labels):
        if before_memo[old]!=after_memo[base+i]:raise ValueError('kept original factor/inverse polynomial changed')
    encoded=dict(before_aux=encode_polynomials(before_memo,pool=pool),
        after_aux=encode_polynomials(after_memo,pool=pool),
        before_outputs=encode_polynomials(before_out,pool=pool),
        after_outputs=encode_polynomials(after_poly,pool=pool))
    held['complete_encoded_polynomial_proof']=encoded
    del before_memo,after_memo,before_out,after_poly
    layout=numeric_layout(held,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete native proof numeric entries exceed64M')
    with (RUN/'candidate_complete_packet.npz').open('xb') as stream:np.savez(stream,**packet)
    with (RUN/'complete_exact_polynomials.pickle').open('xb') as stream:
        pickle.dump(encoded,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    with (RUN/'complete_coordinate_map.npz').open('xb') as stream:
        np.savez(stream,old_kept_slots=np.array(labels,np.int64),new_slots=base+np.arange(len(labels),dtype=np.int64))
    original_nnz=sum(len(r['coefficients']) for r in originals)
    r=len(held['all_aux'])-report['new_factors'];d=report['nnz']-original_nnz
    return dict(report=report,original_auxiliary_rows=len(held['all_aux']),original_output_rows=len(held['all_outputs']),
        original_nnz=original_nnz,new_auxiliary_rows=report['new_factors'],new_nnz=report['nnz'],
        removed_V=len(held['aliases']),removed_M=report['removed_m'],total_removed_factors=r,nnz_delta=d,
        conditional_packet_byte_delta_formula=12*d-88*r,conditional_packet_entry_delta_formula=2*d-13*r,
        all_original_output_polynomials_equal=True,all_kept_factor_inverse_polynomials_equal=True,
        all_old_factor_polynomials_reconstructed=True,all_old_and_new_auxiliary_boxes_redundant=True,
        original_root_binding=binding['source_identity'],authenticated_decode_s=auth_s,
        reducer_and_final_emission_work=reducer_work,complete_numeric_bytes=layout.resident_bytes,
        complete_numeric_entries=layout.resident_entries,
        original_kernel_coordinate_source_producer_executed=False,original_real_HZ_numeric_roots_present=False,
        original_HZ_loaded_into_live_target=False,complete_original_binary_predicate_numeric_roots_present=False,
        no_old_source_identity_attached_to_changed_state=True,new_source_or_LIVE_admission=False,
        complete_owned_source_physical_gate_proved=False,generation_budget_proved=False,formal_gain=0)


def worker():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((RUN/'preregistered.json').read_text());pool=WorkPool(256_000_000)
    pool.charge('c115_complete_fixture_batch_already_spent',12337760)
    record=dict(completed=False,formal_gain=0);held={};start=time.monotonic()
    fatal=(RUN/'fatal.log').open('x');faulthandler.enable(file=fatal,all_threads=True)
    try:
        if any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()):raise ValueError('full source drift')
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


def main():
    if RUN.exists():raise FileExistsError(RUN)
    if _sha256(PREV/'exit.json')!='b0a1d35176abf452051445e2a64397d51d94fa56e8a7076ce50bbc1cc6c1b3f1':
        raise ValueError('complete primitive qualification binding differs')
    done=json.loads((PREV/'exit.json').read_text());prior=json.loads((PREV/'preregistered.json').read_text())
    if not done['all_stages_passed'] or done['tests_count']!=3298 or done['work']!=12337760:
        raise ValueError('full qualification/previous work differs')
    hashes=dict(prior['source_sha256'])
    hashes.update({str((PREV/n).relative_to(EXP)):h for n,h in done['artifacts'].items()})
    hashes[str(VIEW.relative_to(EXP))]=VIEW_SHA
    for n in ('C115_COMPLETE_NATIVE_ORACLE_PREREG_20260913.md','run_c115_complete_native_oracle_v1.py',
              str((PREV/'exit.json').relative_to(EXP))):hashes[n]=_sha256(EXP/n)
    if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('prior complete source/artifact drift')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('production provenance drift')
    RUN.mkdir();_atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        completed_unchanged_tests_reused=3298,previous_work=12337760,aggregate_work_cap=256_000_000,
        worker_wall_cap_s=240,CPU_threads=1,GPU=False,address_space_bytes=16*1024**3,
        transient_bytes=1024**3,unique_numeric_entries_cap=64_000_000,
        scope='complete_native_mathematical_oracle_not_original_source_or_live_admission',formal_gain=0))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    start=time.monotonic();record=dict(completed=False,formal_gain=0)
    try:
        with (RUN/'worker.log').open('x') as stream:
            job=subprocess.run([sys.executable,__file__,'--worker'],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        result=json.loads((RUN/'result.json').read_text());record.update(worker_exit=job.returncode,
            completed=job.returncode==0 and result['completed'])
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-start,source_drift=any(_sha256(EXP/n)!=h for n,h in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={p.name:_sha256(p) for p in RUN.iterdir() if p.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['completed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':
    worker() if '--worker' in sys.argv else main()
