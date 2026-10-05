"""One measured original preparation/C31 physical migration, automatic retention."""
import gc
import json
import pickle
from pathlib import Path
import resource
import sys
import time
import weakref
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c62_precision_plan_v1 import plan
from experiments.neural_hz_20260831.c62_physical_quotient_v1 import prepare,emit
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint,metadata
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c62_physical_boundary_20260913_v1'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
OLD=EXP/'results/c31_prepared_generator_20260911_v1/closed_hz.pickle'
OLD_SHA='535b579e853f1ac5092e962e1c307644e89739da4cd63f3732593896649128f5'
OLD_PROOF='cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5'
PRIOR=EXP/'results/c61_retained_boundary_20260912_v1/result.json'
PRIOR_SHA='8f8bbb772dbfff3c52dcfcc6b8df079919af75a57fa124f8bdfd50d39a224644'


def complete(pool,branch,observe):
    if _sha256(OLD)!=OLD_SHA or _sha256(PRIOR)!=PRIOR_SHA:raise ValueError('completed C31/C61 authority changed')
    prior=json.loads(PRIOR.read_text())
    if not prior['completed'] or not prior['data']['profile']['native_symbolic_optimality_proved']:
        raise ValueError('complete native-symbolic optimum required')
    with OLD.open('rb') as stream:archive=pickle.load(stream)
    if archive['schema']!='c31_new_checked_prepared_Closed_v1' or archive['closed_proof_sha256']!=OLD_PROOF:
        raise ValueError('qualified original C31 state required')
    legacy=archive['fields'];raw=archive['proof_bytes']
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=OLD_PROOF:raise ValueError('complete C31 source proof differs')
    legacy_view=dict(fields=legacy,checked_proof_record=(archive['identity']['closed_identity'],raw))
    del archive
    old_layout=numeric_layout(legacy_view,pool)
    pool.charge('c62_two_complete_C31_input_fingerprints',2*int(old_layout.resident_entries)+2048)
    old_before,old_shallow=fingerprint(legacy_view,old_layout,pool,already_paid=True)
    old_meta=metadata(legacy_view,pool)
    with SOURCE.open('rb') as stream:saved,decoder=load(stream,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    if (saved['schema']!='c9_integrated_suffix_checkpoint_v1' or not saved['identity_audit']['all_original_coefficients_exact']
            or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']):
        raise ValueError('qualified complete original source required')
    source_layout=numeric_layout(saved,pool)
    union=numeric_layout(dict(original=saved,legacy=legacy_view),pool)
    if union.resident_entries>64_000_000:raise MemoryError('complete input entry cap')
    pool.charge('c62_two_complete_C9_input_fingerprints',2*int(source_layout.resident_entries)+2048)
    before,source_shallow=fingerprint(saved,source_layout,pool,already_paid=True)
    selected=plan(saved,pool=branch,enabled=True);p=selected['report'];expected=prior['data']['profile']
    if (p['raw_nodes'],p['optimum'],p['ancestor_states'],p['selected_classes'])!=(expected['raw_nodes'],expected['precision_optimal_removed'],expected['ancestor_states'],expected['selected_classes']):
        raise ValueError('new sufficient-statistic planner differs from complete C61 optimum')
    observe(dict(event='complete_new_precision_plan',plan=p,whole_work=pool.used,branch_work=branch.used))
    prepared=prepare(saved,legacy,selected,pool=branch,enabled=True)
    if prepared['report']['exchange_classes']!=expected['boundary_exchange_vs_old']:raise ValueError('complete boundary exchange differs from C61')
    if fingerprint(saved,source_layout,pool,already_paid=True)[0]!=before:raise ValueError('complete original C9 source changed')
    source_ref=weakref.ref(saved['hz']);expr_ref=weakref.ref(saved['expression'])
    source_measure=dict(bytes=source_layout.resident_bytes,entries=source_layout.resident_entries,python_shallow_bytes=source_shallow)
    del saved,selected;gc.collect()
    if source_ref() is not None or expr_ref() is not None:raise ValueError('transient migration packet retains original C9 source')
    observe(dict(event='complete_bound_migration_prepared',proof=prepared['report'],original_C9_retired=True,whole_work=pool.used))
    state,proof=emit(legacy,prepared,pool=branch,enabled=True)
    state['checked_producer_proof_record']=legacy_view['checked_proof_record']
    del prepared
    observe(dict(event='physical_HZ_and_inverse_constructed',proof=proof,whole_work=pool.used))
    new_layout=numeric_layout(state,pool);combined=numeric_layout(dict(candidate=state,legacy=legacy_view),pool)
    if combined.resident_entries>64_000_000:raise MemoryError('complete physical candidate/input entry cap')
    new_fingerprint,new_shallow=fingerprint(state,new_layout,pool);new_meta=metadata(state,pool)
    if old_meta['opaque_inherited_ids']!=new_meta['opaque_inherited_ids']:raise ValueError('unmeasured opaque metadata is not identical inherited storage')
    if fingerprint(legacy_view,old_layout,pool,already_paid=True)[0]!=old_before:raise ValueError('complete old C31 input changed')
    physical=dict(old_numeric_bytes=old_layout.resident_bytes,new_numeric_bytes=new_layout.resident_bytes,
        old_entries=old_layout.resident_entries,new_entries=new_layout.resident_entries,
        old_python_shallow_bytes=old_shallow,new_python_shallow_bytes=new_shallow,
        old_known_metadata_bytes=old_meta['nonoverlapping_known_metadata_bytes'],new_known_metadata_bytes=new_meta['nonoverlapping_known_metadata_bytes'],
        opaque_inherited_metadata_identical=True,allocator_occupancy_omission=True,
        original_C9_source_checkpoint=source_measure,original_graph_boundary_not_a_C31_comparator=True)
    physical['numeric_byte_delta']=new_layout.resident_bytes-old_layout.resident_bytes
    physical['complete_known_byte_delta']=physical['numeric_byte_delta']+physical['new_known_metadata_bytes']-physical['old_known_metadata_bytes']
    observe(dict(event='complete_physical_boundary_comparison',physical=physical,whole_work=pool.used))
    if physical['numeric_byte_delta']>=0 or physical['complete_known_byte_delta']>=0:raise ValueError('complete C31-boundary physical storage does not strictly decrease')
    pool.charge('c62_exclusive_physical_archive_publication',int(new_layout.resident_entries)+16384)
    path=DIRECTORY/'physical_state.pickle'
    with path.open('xb') as stream:pickle.dump(state,stream,protocol=5)
    if _sha256(SOURCE)!=SOURCE_SHA or _sha256(OLD)!=OLD_SHA:raise ValueError('original source files changed')
    return dict(proof=proof,physical=physical,decoder=decoder,complete_inputs_unchanged=True,
        original_C9_physically_retired_after_complete_binding=True,physical_HZ_constructed=True,
        state_fingerprint=new_fingerprint,checkpoint_sha256=_sha256(path),checkpoint_bytes=path.stat().st_size,
        inherited_C61_diagnostic_work_not_runtime_payment=248095109,source_first_or_native_admission=False,formal_gain=0)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);branch=BranchPool(pool);pool.charge('c62_exclusive_result_record',16384)
    started=time.monotonic();report=dict(completed=False,physical_HZ_constructed=False,formal_gain=0,
        native_or_solver_executed=False,new_generator_executed=False,full_qualification_suite_passed=False,default_changed=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def observe(value):
            if value['event']=='physical_HZ_and_inverse_constructed':report['physical_HZ_constructed']=True
            log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
            data,stats=measured(lambda:complete(pool,branch,observe),observe=lambda s:observe(dict(event='complete_physical_measurement',measurement=s)))
            report.update(completed=True,data=data,measurement=stats)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));observe(dict(event='physical_candidate_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,branch_diagnostic_work=branch.used,
                work_parts=dict(pool.parts),branch_work_parts=dict(branch.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report);print(json.dumps({k:report[k] for k in ('completed','physical_HZ_constructed','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
