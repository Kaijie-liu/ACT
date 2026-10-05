# SPDX-License-Identifier: AGPL-3.0-or-later
"""One all-packet read-only occurrence census; not new source/LIVE admission."""
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
from experiments.neural_hz_20260831.c88_inline_tile_v1 import actual_rows
from experiments.neural_hz_20260831.c110_existing_factor_v1 import discover
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent
PREVIOUS=EXP/'results/c110_existing_factor_20260913_v1'
RUN=EXP/'results/c110_source_recipes_20260913_v1'
IDENTITY='a6e5ac15c643e73b02d6ab5523c242b0a894f1c2986836d47c061c5db1f310bf'
INPUTS={
    'packets':('results/c90_actual_circuit_proof_20260913_v1/all_rebased_selected_native_packets.npz',
        'f1e760a6bab4e722ce8cc8f1c76c823cddafe512b1aa836a126d53af9661c351'),
    'c90':('results/c90_actual_circuit_proof_20260913_v1/result.json',
        '40d3a498200cc5748733dfa081040a048edae8eae9c3442a8477cb0ed3f67e84'),
    'c98':('results/c98_stream_circuit_20260913_v2/actual/result.json',
        'c08ecfb4f089e4ac4394668f98a9301f6abb0bfca60f6d5d455b993b836166af'),
    'c100':('results/c100_fresh_circuit_terminal_20260913_v4/source_proof.json',
        'dbb6feca675d0c82264b7d78a149731528666ad67c5cf5fe0727a38beec6f7b9'),
    'c107':('results/c107_canonical_encoding_20260913_v1/events.jsonl',
        '2668b170ed25a5e25836a611e99f13a3d3f2744b4fcee6cc538420fad53f9352')}


def screen(pool, held):
    # Entire proof documents/events and every field of all four packets remain
    # held. Full original HZ numeric roots are NOT restored or claimed present.
    auth_started=time.monotonic()
    for name,(path,sha) in INPUTS.items():
        if _sha256(EXP/path)!=sha:raise ValueError('exact archived input binding differs')
        if name!='packets':held[name+'_raw']=(EXP/path).read_bytes()
    c90=json.loads(held['c90_raw']);c98=json.loads(held['c98_raw'])
    c100=json.loads(held['c100_raw'])
    events=[json.loads(line) for line in held['c107_raw'].splitlines()]
    held.update(c90=c90,c98=c98,c100=c100,events=events)
    source_events=[e for e in events if e.get('event')=='c100_fresh_source_bound']
    if (not c90['completed'] or not c98['completed'] or c100['schema']!='c91_complete_physical_circuit_proof_v1'
        or c98['data']['physical_identity']!=IDENTITY or c100['identity']!=IDENTITY
        or c98['data']['proof']!=c100['proof'] or len(source_events)!=1
        or source_events[0]['identity']!=IDENTITY or not source_events[0]['fresh_original_generator']
        or source_events[0]['archived_HZ_loaded']):
        raise ValueError('complete recorded source/theorem identity chain differs')
    proof=c100['proof']
    for flag in ('all_original_maps_and_other_predicates_preserved',
                 'every_fresh_native_literal_matches_original_theorem',
                 'independent_complete_owner_delta_proved','fresh_original_expression_generator_executed'):
        if not proof[flag]:raise ValueError('required recorded original-source proof absent')
    if (proof['original_circuit_proof_sha256']!=INPUTS['c90'][1]
        or proof['original_expression_archive_sha256']!=c90['data']['decoder']['checkpoint_sha256']
        or proof['complete_proof_root_ledger']['complete_packets']!=4):
        raise ValueError('whole recorded proof roots do not bind all four packets')
    with np.load(EXP/INPUTS['packets'][0],allow_pickle=False) as archive:
        packets={key:archive[key] for key in archive.files}
    held['complete_packets']=packets
    entries=sum(a.size for a in packets.values())
    if entries>64_000_000:raise MemoryError('complete packet numeric-entry cap')
    pool.charge('c110_all_packet_arrays_and_headers',4*entries+4096)
    authentication_s=time.monotonic()-auth_started
    blocks=c90['data']['all_block_proofs'];base=blocks[0]['first_global_aux']
    if len(blocks)!=4:raise ValueError('all selected blocks required')
    aux,outputs,used,block_ranges=[],[],set(),[]
    for block in blocks:
        prefix=f"node{block['node']}_tile{block['y']}_{block['x']}_"
        fields=('indptr','columns','native','rhs','pivots','gauges','ab_indptr')
        packet={field:packets[prefix+field] for field in fields}
        used.update(prefix+field for field in fields)
        n=block['proof']['all_auxiliary_equations_and_redundant_boxes_proved']
        if (block['first_global_aux']!=base+len(aux)
            or not block['proof']['original_source_equivalence']
            or not block['proof']['universal_unique_box_extension']
            or len(packet['rhs'])!=n+block['proof']['all_original_output_equations_proved']
            or np.any(packet['ab_indptr'])):
            raise ValueError('complete original block topology and nonbinary definitions required')
        a,o=actual_rows(dict(rows=len(packet['rhs']),new_factors=n),packet)
        block_ranges.append((block,base+len(aux),base+len(aux)+n))
        aux.extend(a);outputs.extend(o)
    held.update(all_aux=aux,all_outputs=outputs)
    if (used!=set(packets) or len(aux)!=c90['data']['complete_new_factors']
        or base+len(aux)!=c90['data']['complete_global_n_cont']
        or len(outputs)!=c90['data']['complete_original_output_rows']
        or len(aux)!=proof['actual_auxiliary_rows_checked']):
        raise ValueError('complete all-packet numerical scope differs')
    aliases,roots,report=discover(aux,base,pool=pool)
    report.update(complete_auxiliary_rows=len(aux),complete_output_rows=len(outputs),
        base_original_continuous=base,complete_numeric_entries=entries,
        complete_numeric_bytes=sum(a.nbytes for a in packets.values()),
        complete_raw_proof_bytes=sum(len(v) for k,v in held.items() if k.endswith('_raw')),
        full_packet_and_recorded_proof_chain_authenticated=True,
        complete_original_HZ_numeric_roots_present=False,new_source_or_LIVE_admission=False,
        actual_changed_source_constructed=False,actual_eliminations=0,formal_gain=0,
        source_identity=IDENTITY,authentication_and_loading_s=authentication_s,
        hash_and_exact_Fraction_CPU_not_generation_tokens=True,
        aliases=[dict(old=s,representative=rep,scale=value) for s,(rep,value) in sorted(aliases.items())],
        blocks=[dict(node=b['node'],y=b['y'],x=b['x'],auxiliary_rows=hi-lo,
            input_roots=sum(lo<=s<hi for s in roots),candidate_aliases=sum(lo<=s<hi for s in aliases))
            for b,lo,hi in block_ranges])
    return report


def worker():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((RUN/'preregistered.json').read_text());started=time.monotonic()
    pool=WorkPool(256_000_000)
    pool.charge('c110_completed_fixture_batch_already_spent',freeze['previous_work'])
    record=dict(completed=False,formal_gain=0);held={}
    try:
        if any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()):
            raise ValueError('full qualified-source freeze differs')
        data,measurement=measured(lambda:screen(pool,held),observe=lambda m:record.update(measurement=m))
        record.update(completed=True,data=data,measurement=measurement)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(work=pool.used,work_parts=pool.parts,wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps(record),flush=True)
    if not record['completed'] or record['source_drift']:raise SystemExit(1)


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((PREVIOUS/'preregistered.json').read_text())
    done=json.loads((PREVIOUS/'exit.json').read_text())
    if (_sha256(PREVIOUS/'exit.json')!='f52d736c95820c7076d631c68c24dc53fbbc9289d70efc8e38357716cd16ba19'
        or not done['all_stages_passed'] or done['numeric_winning_cases']!=2
        or done['tests_count']!=3224 or done['test_wall_s']>60):
        raise ValueError('complete prior mathematical/numeric qualification missing')
    hashes=dict(prior['source_sha256'])
    hashes.update({str(Path(__file__).relative_to(EXP)):_sha256(Path(__file__))})
    hashes.update({path:sha for path,sha in INPUTS.values()})
    hashes.update({str((PREVIOUS/n).relative_to(EXP)):h for n,h in done['artifacts'].items()})
    hashes[str((PREVIOUS/'exit.json').relative_to(EXP))]=_sha256(PREVIOUS/'exit.json')
    if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('complete frozen custody drift')
    if _provenance(ROOT)!=prior['provenance']:raise ValueError('production provenance differs')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,
        provenance=prior['provenance'],previous_work=done['work'],worker_wall_cap_s=240,
        all_qualified_tests_reused_for_unchanged_math_source=True,tests_count=3224,
        complete_original_HZ_numeric_roots_required_for_admission_but_not_claimed_present=True,
        scope='read_only_complete_bound_archived_packet_occurrences_only',formal_gain=0))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    started=time.monotonic();record=dict(completed=False,formal_gain=0)
    try:
        with (RUN/'worker.log').open('x') as log:
            job=subprocess.run([sys.executable,__file__,'--worker'],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,timeout=240)
        result=json.loads((RUN/'result.json').read_text())
        record.update(worker_exit=job.returncode,completed=job.returncode==0 and result['completed'])
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in hashes.items()),
            provenance_drift=_provenance(ROOT)!=prior['provenance'],
            artifacts={f.name:_sha256(f) for f in RUN.iterdir() if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['completed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':
    worker() if '--worker' in sys.argv else main()
