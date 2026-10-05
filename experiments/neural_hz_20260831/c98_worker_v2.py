"""Frozen C98 source-bound and complete fresh source qualification workers."""
import faulthandler
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
import weakref
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift as prefix_lift
from experiments.neural_hz_20260831.c98_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c98_circuit_stream_v1 import additional_bound
from experiments.neural_hz_20260831.c94_raw_mask_plan_v1 import raw_plan
from experiments.neural_hz_20260831.c98_source_audit_v1 import audit,expression_key
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import row
from experiments.neural_hz_20260831.c91_physical_archive_v1 import bind
from experiments.neural_hz_20260831.c65_physical_archive_v1 import check_restored,metadata
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c7_factored_hz_v1 import scaled_exact
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c98_stream_circuit_20260913_v2'
PREFLIGHT=EXP/'results/c98_stream_circuit_20260913_v1/preflight/result.json'
FILES={
 'C9':('results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle','616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'),
 'C97':('results/c97_once_power_20260913_v1/actual/physical_hz.pickle','f43d4798bd5dfcbab19fc9929896b175facd7dfaa8357a8339d5109241f41abc'),
 'bound':('results/c97_once_power_20260913_v1/preflight/result.json','68039db23422bf8f6878e18a40a5f79f5a35b50694f550c94e54ad01f189713a'),
 'C90':('results/c90_actual_circuit_proof_20260913_v1/result.json','40d3a498200cc5748733dfa081040a048edae8eae9c3442a8477cb0ed3f67e84'),
 'native':('results/c90_actual_circuit_proof_20260913_v1/all_rebased_selected_native_packets.npz','f1e760a6bab4e722ce8cc8f1c76c823cddafe512b1aa836a126d53af9661c351')}


def inputs(pool):
    result={}
    for key,(name,sha) in FILES.items():
        path=EXP/name
        if _sha256(path)!=sha:raise ValueError('complete prior binding changed: '+key)
        if key in ('C9','C97'):
            with path.open('rb') as stream:result[key],_=load(stream,expected_sha256=sha,pool=pool,enabled=True)
        elif key=='native':
            with np.load(path,allow_pickle=False) as ar:result[key]={k:ar[k] for k in ar.files}
        elif key=='C90':result[key]=path.read_bytes()
        else:result[key]=json.loads(path.read_text())
    layout=numeric_layout(result['C97'],pool)
    pool.charge('c98_complete_C97_source_authentication',int(layout.resident_entries)+1024)
    record=check_restored(result['C97'])
    saved=result['C9']
    if (record['source_checkpoint_sha256']!=FILES['C9'][1]
        or record['physical_identity']!='35e77ad7c8f1aa2fed291b663bbe78e57c98c59fceb883e66e02b5a0c1111d1e'
        or not saved['identity_audit']['all_original_coefficients_exact']
        or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']
        or not result['bound']['completed']):
        raise ValueError('complete qualified original source and C97 required')
    old=inspect.getsource(prefix_lift).split('    ac, ab, b = matrices')[0]
    new=inspect.getsource(lift).split('    stream=install')[0]
    if old!=new:raise ValueError('fresh source prefix differs from complete C97 source theorem/bound')
    result['source_proof']=record
    return result


def source_bound(data,pool,branch):
    fields=data['C97']['fields'];saved=data['C9'];start=branch.used
    records,plan,masks=raw_plan(saved['definition_graph'],existing_aux=len(fields['def_rows']),
        existing_work=fields['report']['actual_radix_work'],pool=branch,enabled=True)
    planner=branch.used-start
    extra=additional_bound(saved['definition_graph'],records,plan,main_count=len(fields['owners']))
    previous=data['bound']['data']['bound'];new=planner+extra['after_plan_upper']
    whole=previous['whole_work_upper']+new;br=previous['branch_work_upper']+new
    return dict(schema='c98_complete_fresh_circuit_source_bound_v1',source_bound=previous,
        full_prefix_code_identical=True,original_source_archive_sha256=FILES['C9'][1],
        planner_work=planner,extra=extra,selected=plan['selected_positions'],plan=plan,
        whole_work_upper=whole,branch_work_upper=br,whole_headroom=256_000_000-whole,
        branch_headroom=200_000_000-br,work_caps_fit=whole<=256_000_000 and br<=200_000_000,
        source_and_native_glue_included=True,final_native_solver_glue_included=False)


def rebind_actual_theorem(data,draft,pool):
    saved=data['C9'];old=data['C97']['fields'];reference=json.loads(data['C90'])
    if (not reference['completed'] or reference['data']['decoder']['checkpoint_sha256']!=FILES['C9'][1]):
        raise ValueError('authenticated full C90 source theorem required')
    packets=draft['construction']['circuits'];selected=draft['construction']['selected']
    proven=reference['data']['all_block_proofs'];names=set();checked_rows=checked_coefficients=0
    if len(packets)!=len(proven):raise ValueError('complete independent circuit theorem population differs')
    for item,p,proof in zip(selected,packets,proven,strict=True):
        if any(item[k]!=proof[k] for k in ('node','y','x','position')):
            raise ValueError('fresh selected structure does not match this complete theorem')
        theorem=proof['proof']
        if (not theorem['original_source_equivalence'] or not theorem['universal_unique_box_extension']
            or theorem['all_auxiliary_equations_and_redundant_boxes_proved']!=p['new_factors']):
            raise ValueError('incomplete independent source/box theorem')
        prefix=f"node{item['node']}_tile{item['y']}_{item['x']}_"
        for name,value in p.items():
            if name=='new_factors':continue
            pool.charge('c98_every_original_theorem_native_literal',4*int(value.size)+128)
            expected=data['native'][prefix+name];names.add(prefix+name)
            if value.dtype!=expected.dtype or value.shape!=expected.shape or value.tobytes()!=expected.tobytes():
                raise ValueError('fresh actual native literal differs: '+name)
        for pivot in p['pivots'][p['new_factors']:]:
            rank=old['old_n_eq']+int(pivot)-old['old_n_cont']
            original_rank=saved['old_n_eq']+int(pivot)-saved['old_n_cont']
            physical=int(old['eq_roots'][rank]);original_physical=int(saved['eq_roots'][original_rank])
            a,av=row(old['hz'].Ac,physical);b,bv=row(saved['hz'].Ac,original_physical)
            pool.charge('c98_all_selected_original_source_equations',64+24*(len(a)+len(b)))
            if (not np.array_equal(a,b)
                or not np.array_equal(scaled_exact(av,-int(old['eq_scales'][rank])),
                    scaled_exact(bv,-int(saved['eq_scales'][original_rank])))
                or old['hz'].b[physical]!=0 or saved['hz'].b[original_physical]!=0
                or old['hz'].Ab.indptr[physical+1]!=old['hz'].Ab.indptr[physical]
                or saved['hz'].Ab.indptr[original_physical+1]!=saved['hz'].Ab.indptr[original_physical]):
                raise ValueError('selected actual old source equation differs from C90 ORIGINAL source')
            checked_rows+=1;checked_coefficients+=len(a)
    if names!=set(data['native']):raise ValueError('a full original native field was omitted')
    return dict(complete_selected_original_rows=checked_rows,complete_selected_original_coefficients=checked_coefficients,
        every_fresh_native_literal_matches_original_theorem=True,complete_C90_theorem_reused_not_recomputed=True)


def proof_root_layout(data,draft,pool):
    """Full retained input + fresh graph, UID, packet and physical source roots."""
    whole=dict(complete_inputs=data,complete_fresh_draft=draft)
    layout=numeric_layout(whole,pool)
    if layout.resident_entries>64_000_000:
        raise MemoryError('complete retained proof root union exceeds entry cap')
    graph=draft['construction']['nodes'];packets=draft['construction']['circuits']
    return dict(complete_proof_root_entries=layout.resident_entries,
        complete_proof_root_numeric_bytes=layout.resident_bytes,
        new_graph_arrays=sum(k in n for n in graph for k in ('support','needed','slots','exponents')),
        complete_packets=len(packets),all_original_inputs_and_fresh_proof_roots_retained=True)


def actual(data,bound,pool,branch,emit,record):
    if not bound['work_caps_fit']:raise MemoryError('complete aggregate bound rejected before source construction')
    saved=data['C9'];old=data['C97']['fields']
    draft,stats=measured(lambda:lift(saved['expression'],saved['keep'],enabled=True,
        frame_widths=(saved['old_n_cont'],saved['old_n_bin']),observe=lambda n,v:emit(dict(event=n,**v))),
        observe=lambda s:record.update(generation_measurement=s))
    record.update(physical_source_constructed=True,generation=stats)
    state=draft['state'];fields=state['fields'];report=fields['report']
    record['generation_report']=report
    if (report['total_work_upper']>bound['whole_work_upper']
        or report['largest_branch_work_upper']>bound['branch_work_upper']
        or report['circuit_generation']['plan']!=bound['plan']
        or report['alias_quotient']['identity_sha256']!=bound['source_bound']['expected_identity_sha256']
        or report['alias_quotient']['optimum']!=bound['source_bound']['selected']):
        raise ValueError('actual complete fresh source violates the source-derived bound')
    prefix_parts=report['alias_quotient']['work_parts']
    if any(k not in bound['source_bound']['work_parts'] or v>bound['source_bound']['work_parts'][k]
        for k,v in prefix_parts.items()):raise ValueError('a C97 prefix operation exceeds authenticated bound')
    def prove_all():
        root_ledger=proof_root_layout(data,draft,pool)
        record['complete_proof_root_ledger']=root_ledger
        emit(dict(event='complete_retained_proof_root_union',**root_ledger))
        theorem=rebind_actual_theorem(data,draft,branch)
        proof=audit(old,state,draft['construction']['circuits'],pool=branch,enabled=True)
        proof.update(theorem,complete_proof_root_ledger=root_ledger,original_source_archive_sha256=FILES['C97'][1],
            original_expression_archive_sha256=FILES['C9'][1],original_circuit_proof_sha256=FILES['C90'][1],
            original_scalar_identity=data['source_proof']['proof']['complete_source_boundary_sha256'],
            complete_C97_prefix_source_theorem_reused=True,fresh_original_expression_generator_executed=True)
        return proof
    proof,proof_stats=measured(prove_all,observe=lambda s:record.update(source_proof_measurement=s))
    record.update(complete_source_proof=proof,source_proof=proof_stats)
    state['original_source_proof']=data['C97']['proof_bytes'];state['original_circuit_proof']=data['C90']
    refs=[weakref.ref(n[k]) for n in draft['construction']['nodes'] for k in ('support','needed','slots','exponents')]
    del draft;gc.collect()
    if any(r() is not None for r in refs):raise ValueError('new construction graph not physically retired')
    old_view=dict(fields=old,checked_proof_record=(data['source_proof']['physical_identity'],data['C97']['proof_bytes']))
    old_layout=numeric_layout(old_view,pool);old_meta=metadata(old_view,pool=pool)
    preliminary=numeric_layout(state,pool)
    pool.charge('c98_complete_fresh_source_proof_fingerprint',int(preliminary.resident_entries)+1024)
    identity,raw=bind(state,proof);new_view=dict(state=state,checked_proof_record=(identity,raw))
    layout=numeric_layout(new_view,pool);meta=metadata(new_view,pool=pool)
    union=numeric_layout(dict(complete_inputs=data,complete_candidate=new_view),pool)
    physical=dict(before_numeric_bytes=old_layout.resident_bytes,after_numeric_bytes=layout.resident_bytes,
        before_entries=old_layout.resident_entries,after_entries=layout.resident_entries,
        before_metadata_bytes=old_meta['nonoverlapping_known_metadata_bytes'],after_metadata_bytes=meta['nonoverlapping_known_metadata_bytes'],
        numeric_byte_delta=layout.resident_bytes-old_layout.resident_bytes,entry_delta=layout.resident_entries-old_layout.resident_entries,
        complete_union_entries=union.resident_entries,complete_union_bytes=union.resident_bytes)
    physical['known_byte_delta']=physical['numeric_byte_delta']+physical['after_metadata_bytes']-physical['before_metadata_bytes']
    record['physical']=physical;emit(dict(event='complete_fresh_circuit_physical_comparison',physical=physical))
    if (physical['numeric_byte_delta']>=0 or physical['entry_delta']>=0
        or physical['known_byte_delta']>=0 or union.resident_entries>64_000_000):
        raise MemoryError('complete fresh source physical/entry gate rejected')
    pool.charge('c98_exclusive_complete_circuit_source_archive',int(layout.resident_entries)+16384)
    payload=dict(schema='c91_complete_physical_circuit_archive_v1',state=state,proof_bytes=raw,
        proof_sha256=hashlib.sha256(raw).hexdigest())
    path=RUN/'actual/physical_hz.pickle'
    with path.open('xb') as stream:pickle.dump(payload,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    return dict(physical=physical,proof=proof,physical_identity=identity,archive_sha256=_sha256(path),
        archive_bytes=path.stat().st_size,graph_arrays_physically_retired=len(refs),
        complete_source_and_storage_proved=True,original_inputs_retained=True,
        native_or_LIVE_admission=False,solver_executed=False,formal_gain=0)


def main():
    stage=sys.argv[1]
    if stage not in ('preflight','actual'):raise ValueError('unregistered stage')
    directory=RUN/stage
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);branch=BranchPool(pool);started=time.monotonic()
    result=dict(completed=False,formal_gain=0,solver_executed=False)
    with (directory/'events.jsonl').open('x') as log,(directory/'fatal.log').open('x') as fatal:
        def emit(value):
            log.write(json.dumps(dict(value,worker_elapsed_s=time.monotonic()-started),allow_nan=False)+'\n');log.flush()
        try:
            faulthandler.enable(file=fatal,all_threads=True)
            freeze=json.loads((RUN/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
                raise ValueError('complete frozen source drift')
            data=inputs(pool);layout=numeric_layout(data,pool)
            if layout.resident_entries>64_000_000:raise MemoryError('complete input cap')
            pool.charge('c98_two_complete_input_fingerprints',2*int(layout.resident_entries)+2048)
            before,_=fingerprint(data,layout,pool,already_paid=True)
            try:
                if stage=='preflight':
                    bound,stats=measured(lambda:source_bound(data,pool,branch),observe=lambda s:result.update(measurement=s))
                    result['data']=dict(bound=bound)
                    if not bound['work_caps_fit']:raise MemoryError('full fresh source plus glue does not fit')
                else:
                    binding=json.loads((RUN/'actual/input_binding.json').read_text())
                    if (_sha256(RUN/'preregistered.json')!=binding['preregistered_sha256']
                        or _sha256(PREFLIGHT)!=binding['complete_source_bound_sha256']):
                        raise ValueError('frozen aggregate preflight binding changed')
                    prior=json.loads((PREFLIGHT).read_text())
                    if not prior['completed']:raise ValueError('completed preflight required')
                    result['data']=actual(data,prior['data']['bound'],pool,branch,emit,result)
                result['completed']=True
            finally:
                unchanged=fingerprint(data,layout,pool,already_paid=True)[0]==before
                result['complete_original_inputs_unchanged']=unchanged
                if not unchanged or any(_sha256(EXP/n)!=sha for n,sha in FILES.values()):
                    raise ValueError('complete original inputs changed')
        except Exception as exc:
            result['completed']=False;result['failure']=dict(type=type(exc).__name__,reason=str(exc))
            emit(dict(event='fresh_circuit_source_rejected',**result['failure']))
        finally:
            faulthandler.disable();result.update(wall_s=time.monotonic()-started,
                offline_diagnostic_work=pool.used,offline_branch_work=branch.used,
                diagnostic_parts=dict(pool.parts),offline_branch_parts=dict(branch.parts))
            _atomic_exclusive_json(directory/'result.json',result);print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
