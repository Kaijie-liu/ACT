# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete commuting-footprint arithmetic; not a transformed HZ candidate."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;PREV=EXP/'results/c114_joint_sink_census_20260913_v1'
RUN=EXP/'results/c114_neutral_footprint_20260913_v1'
C90=EXP/'results/c90_actual_circuit_proof_20260913_v1/result.json'


def observe(pool,held):
    for label,path in (('actual',PREV/'actual.json'),('c90',C90)):
        pool.charge('c114_footprint_complete_input',4*path.stat().st_size)
        held[label+'_raw']=path.read_bytes();held[label]=json.loads(held[label+'_raw'])
    actual=held['actual'];old=actual['reports']['original'];view=actual['reports']['sealed_root_aliases']
    groups=view['group_certificates'];base=held['c90']['data']['all_block_proofs'][0]['first_global_aux']
    pool.charge('c114_complete_group_footprints',64*sum(len(g['slots'])+len(g['consumers'])+1 for g in groups))
    neutral=[g for g in groups if g.get('nnz_delta')==0 and g['reason']=='exact_nnz_not_smaller']
    slots=[s for g in neutral for s in g['slots']];consumers=[s for g in neutral for s in g['consumers']]
    prior_aliases=actual['scope']['complete_source_chain']['aliases'];alias_slots={a['old'] for a in prior_aliases}
    flags=dict(all_defining_slots_disjoint=len(slots)==len(set(slots)),
        all_consumer_rows_disjoint=len(consumers)==len(set(consumers)),
        all_definition_slots_are_auxiliary=all(base<=s<base+old['original_auxiliary_rows'] for s in slots),
        all_consumers_are_original_output_slots=all(0<=s<base for s in consumers),
        no_removed_root_alias_slot_is_a_sink=not alias_slots.intersection(slots),
        all_groups_still_rejected_by_C114=all(not g['strict_nnz_reduction_proved'] for g in neutral),
        all_exact_native_results_have_zero_nnz_delta=all(g['new_local_nnz']==g['old_local_nnz'] for g in neutral))
    arrays=dict(neutral_slots=np.array(slots,np.int64),consumer_slots=np.array(consumers,np.int64),
        group_factor_counts=np.array([len(g['slots']) for g in neutral],np.int64),
        group_consumer_counts=np.array([len(g['consumers']) for g in neutral],np.int64))
    held['complete_footprints']=arrays
    layout=numeric_layout(held,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete footprint ledger exceeds64M')
    with (RUN/'complete_footprints.npz').open('xb') as stream:np.savez(stream,**arrays)
    result=dict(flags=flags,neutral_groups=len(neutral),neutral_factors=len(slots),
        neutral_consumer_rows=len(consumers),complete_neutral_groups=neutral,
        all_prior_root_aliases=len(prior_aliases),conditional_atomic_count_applicable=all(flags.values()),
        complete_numeric_bytes=layout.resident_bytes,complete_numeric_entries=layout.resident_entries,
        original_real_HZ_roots_present=False,new_packet_emitted=False,new_source_or_LIVE_admission=False,
        C114_acceptance_gate_unchanged=True,actual_eliminations=0,formal_gain=0)
    if result['conditional_atomic_count_applicable']:
        delta_nnz=view['effective_nnz']-old['original_nnz']
        r=len(prior_aliases)+len(slots)
        result['conditional_component_arithmetic']=dict(original_nnz=old['original_nnz'],
            after_atomic_nnz=view['effective_nnz'],nnz_delta=delta_nnz,
            original_factors=old['original_auxiliary_rows'],after_atomic_factors=old['original_auxiliary_rows']-r,
            removed_root_factors=len(prior_aliases),removed_sink_factors=len(slots),removed_total_factors=r,
            packet_byte_delta_formula=12*delta_nnz-88*r,packet_entry_delta_formula=2*delta_nnz-13*r,
            incremental_sink_packet_byte_delta_formula=-88*len(slots),
            incremental_sink_packet_entry_delta_formula=-13*len(slots),
            source_first_no_extra_postpass_inverse_assumption=True,complete_owned_source_metric_proved=False,
            generation_budget_proved=False,nnz_gain_over_root_aliases_only=0)
    return result


def worker():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    frozen=json.loads((RUN/'preregistered.json').read_text());pool=WorkPool(256_000_000)
    pool.charge('c114_full_census_already_spent',185666884)
    record=dict(completed=False,formal_gain=0);started=time.monotonic();held={}
    try:
        if any(_sha256(EXP/n)!=h for n,h in frozen['source_sha256'].items()):raise ValueError('source drift')
        data,measurement=measured(lambda:observe(pool,held),observe=lambda m:record.update(measurement=m))
        record.update(completed=True,data=data,measurement=measurement)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(work=pool.used,work_parts=pool.parts,wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in frozen['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='data'}),flush=True)
    if not record['completed'] or record['source_drift']:raise SystemExit(1)


def main():
    if RUN.exists():raise FileExistsError(RUN)
    if _sha256(PREV/'exit.json')!='c58af25877751a12f2fc75424bfdaaef47dad865ec4595d9fefa3e7cc378d43c':
        raise ValueError('full C114 terminal binding differs')
    done=json.loads((PREV/'exit.json').read_text());prior=json.loads((PREV/'preregistered.json').read_text())
    if not done['all_stages_passed'] or done['work']!=185666884:raise ValueError('prior complete gate differs')
    hashes=dict(prior['source_sha256'])
    hashes.update({str((PREV/n).relative_to(EXP)):h for n,h in done['artifacts'].items()})
    for n in ('C114_NEUTRAL_FOOTPRINT_PREREG_20260913.md','run_c114_neutral_footprint_v1.py',
              str((PREV/'exit.json').relative_to(EXP))):hashes[n]=_sha256(EXP/n)
    if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('prior source/artifact drift')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('production provenance drift')
    RUN.mkdir();_atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,
        provenance=provenance,completed_unchanged_tests_reused=3280,prior_spent_work=185666884,
        complete_aggregate_cap=256_000_000,CPU_threads=1,GPU=False,transient_bytes=1024**3,
        address_space_bytes=16*1024**3,worker_wall_cap_s=60,new_candidate_qualification=False,formal_gain=0))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    start=time.monotonic();record=dict(completed=False,formal_gain=0)
    try:
        with (RUN/'worker.log').open('x') as stream:
            job=subprocess.run([sys.executable,__file__,'--worker'],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        result=json.loads((RUN/'result.json').read_text())
        record.update(worker_exit=job.returncode,completed=job.returncode==0 and result['completed'])
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
