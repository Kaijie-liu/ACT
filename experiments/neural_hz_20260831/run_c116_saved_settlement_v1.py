# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded arithmetic on complete sealed evidence; no source/proof/solver rerun."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;PRIOR=EXP/'results/c116_row_composition_20260920_v1'
RUN=EXP/'results/c116_saved_settlement_20260920_v1'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    if RUN.exists():raise FileExistsError(RUN)
    freeze=json.loads((PRIOR/'preregistered.json').read_text())
    done=json.loads((PRIOR/'exit.json').read_text())
    if (done['all_stages_passed'] or done['tests_count']!=3319 or done['tests_exit']
            or done['worker_exit']!=1 or done['source_drift'] or done['provenance_drift']):
        raise ValueError('complete failed C116 prerequisite differs')
    hashes=dict(freeze['source_sha256'])
    hashes.update({str((PRIOR/n).relative_to(EXP)):h for n,h in done['artifacts'].items()})
    for stage in ('c114_joint_sink_census_20260913_v1','c113_failure_diagnostic_20260913_v1'):
        path=EXP/'results'/stage;receipt=json.loads((path/'exit.json').read_text())
        # These exit and result hashes are already in the inherited source chain.
        key=str((path/'exit.json').relative_to(EXP))
        if key not in hashes or _sha256(path/'exit.json')!=hashes[key]:
            raise ValueError('complete fixed guard source chain missing')
        hashes.update({str((path/n).relative_to(EXP)):h for n,h in receipt['artifacts'].items()})
    for name in ('C116_SAVED_EVIDENCE_SETTLEMENT_20260920.md',Path(__file__).name,
                 str((PRIOR/'exit.json').relative_to(EXP))):hashes[name]=_sha256(EXP/name)
    if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('sealed evidence drift')
    if _provenance(ROOT)!=freeze['provenance']:raise ValueError('production drift')
    RUN.mkdir();pool=WorkPool(256_000_000)
    pool.charge('c116_previous_complete_diagnostic_actual_work',174986556)
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,
        provenance=freeze['provenance'],tests_reused=3319,work_cap=256_000_000,
        already_spent_work=174986556,formal_gain=0,scope='saved_evidence_arithmetic_only'))
    record=dict(completed=False,formal_gain=0);start=time.monotonic()
    def read(path,kind='json'):
        pool.charge('c116_complete_saved_evidence_bytes',path.stat().st_size)
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=hashes[str(path.relative_to(EXP))]:
            raise ValueError('saved input changed during read')
        return pickle.loads(raw) if kind=='pickle' else json.loads(raw)
    def settle():
        original=read(PRIOR/'result.json')
        if original['work']!=174986556 or original['measurement']['measured_transient_gate']:
            raise ValueError('full failed resource measurement differs')
        proof=read(PRIOR/'complete_native_row_proof.json')['proof']
        inverse=read(PRIOR/'complete_exact_inverse.pickle','pickle')
        recipes=inverse['inverse_recipes'];scalars=[]
        for r in recipes:
            pool.charge('c116_all_saved_inverse_terms',64*(len(r['new_coordinate_terms'])+1))
            scalars.extend((r['old_slot'],*r['constant']))
            for term in r['new_coordinate_terms']:scalars.extend(term)
        terms=sum(len(r['new_coordinate_terms']) for r in recipes)
        groups=[]
        for mode in ('dense','masked'):
            data=read(EXP/f'results/c114_joint_sink_census_20260913_v1/{mode}.json')
            for name,r in data['reports'].items():
                certs=r['group_certificates'];pool.charge('c116_complete_saved_guard_groups',64*len(certs))
                excess=[c['new_nnz_lower']-c['old_local_nnz'] for c in certs]
                if (len(certs)!=r['groups'] or not certs or min(excess)<=0
                        or any(c['reason']!='necessary_nnz_lower_bound' for c in certs)):
                    raise ValueError('complete strict lower-bound obstruction not established')
                groups.append(dict(mode=mode,packet=name,groups=len(certs),
                    sinks=r['eligible_sinks'],minimum_lower_excess=min(excess),
                    no_first_C115_union_upper_accepted=True))
        masked=read(EXP/'results/c113_failure_diagnostic_20260913_v1/masked.json')
        failures=[]
        for r in masked['reports']:
            row=r['implementations']['C113']
            failures.append(dict(byte_excess=row['actual_packet_byte_formula']-row['original_byte_formula'],
                entry_excess=row['actual_packet_entry_delta_formula']))
        removed=proof['all_removed_factors_reconstructible'];delta=proof['new_nnz']-proof['original_nnz']
        return dict(complete_native_proof=proof,inverse_recipe_count=len(recipes),inverse_terms=terms,
            inverse_scalar_count=len(scalars),all_inverse_scalars_fit_signed_int64=all(-(1<<63)<=v<(1<<63) for v in scalars),
            illustrative_uniform_int64_recipe_bytes=8*len(scalars),
            illustrative_coordinate_map_array_bytes=inverse['kept_old_slots'].nbytes+inverse['new_slots'].nbytes,
            archive_file_bytes={n:(PRIOR/n).stat().st_size for n in
                ('candidate_complete_packet.npz','complete_exact_inverse.pickle','complete_coordinate_map.npz')},
            conditional_source_packet_only_byte_delta=12*delta-88*removed,
            conditional_source_packet_only_entry_delta=2*delta-13*removed,
            illustrative_encoding_is_not_runtime_allocation_or_universal_lower_bound=True,
            all_guard_packets=groups,all_guard_groups=sum(g['groups'] for g in groups),
            preserved_masked_physical_failures=failures,
            same_consumer_C115_M_rule_cannot_repair_fixed_C113_guard=True,
            native_RSS_failure_cause_not_localized=True,original_source_or_LIVE_admitted=False,
            formal_gain=0)
    try:
        data,measurement=measured(settle,observe=lambda m:record.update(measurement=m))
        record.update(completed=True,data=data,measurement=measurement)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(work=pool.used,work_parts=pool.parts,wall_s=time.monotonic()-start,
            source_drift=any(_sha256(EXP/n)!=h for n,h in hashes.items()),
            provenance_drift=_provenance(ROOT)!=freeze['provenance'])
        _atomic_exclusive_json(RUN/'result.json',record);print(json.dumps(record),flush=True)
    if not record['completed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
