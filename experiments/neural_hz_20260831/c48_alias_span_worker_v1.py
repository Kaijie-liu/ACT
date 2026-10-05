"""Read-only full original source census, including decoder in transient gate."""
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c48_alias_span_census_v1 import assess
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c48_alias_span_20260911_v1'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'


def complete(pool,branch,emit):
    # Never move decoding outside the measured window or discard other keys
    # of this independent whole checkpoint to make its load/owners fit.
    with SOURCE.open('rb') as f:saved,decoder=load(f,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    if (saved['schema']!='c9_integrated_suffix_checkpoint_v1'
            or saved['origin_snapshot_sha256']!='d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
            or saved['identity_audit']['all_original_coefficients_exact'] is not True
            or saved['identity_audit']['all_redundant_main_and_radix_boxes_proved'] is not True):
        raise ValueError('wrong independently frozen complete original source/box checkpoint')
    emit(dict(event='complete_original_checkpoint_decoded',decoder=decoder))
    pool.charge('c48_complete_checkpoint_owner_schema_and_public_evidence_allowance',1_048_576)
    roots=collect(SimpleNamespace(),dict(complete_original_checkpoint=saved));owner=roots.measure()
    emit(dict(event='all_original_checkpoint_numeric_owners_checked',resident_bytes=owner.resident_bytes,
        resident_entries=owner.resident_entries,numeric_roots=len(roots.numeric),python_shallow_bytes=roots.python_shallow_bytes))
    hz=saved['hz'];root=saved['definition_graph'][saved['root']]
    result=assess(hz,old_nc=saved['old_n_cont'],logical_nc=saved['logical_n_cont'],old_eq=saved['old_n_eq'],
        eq_roots=saved['eq_roots'],def_rows=saved['def_rows'],output_slots=root['slots'][root['needed']],
        pool=branch,enabled=True)
    prior=json.loads((EXP/'results/c31_prepared_generator_20260911_v1/result.json').read_text())
    q=prior['generation_report']['alias_quotient'];parts=q['work_parts']
    if (result['local_aliases']!=q['local_aliases'] or result['all_original_hits']!=q['alias_products_checked']
            or result['all_original_hit_rows']!=q['product_certification']['rows']
            or result['original_C31_incidence_scan_work']!=parts['continuous_incidence_scan']):
        raise ValueError('complete original source population differs from the independently proved C31 counters')
    if _sha256(SOURCE)!=SOURCE_SHA:raise ValueError('original source archive changed')
    saving=result['component_work_saving']
    projection=dict(basis='complete_original_prequotient_source_census_not_a_new_generator_run',
        inherited_C31_generation_work=prior['generation_report']['total_work_upper'],
        hypothetical_C31_generation_work=prior['generation_report']['total_work_upper']-saving,
        removed_old_incidence_work=result['original_C31_incidence_scan_work'],
        replacement_routing_and_retained_scan_work=result['new_routing_plus_retained_scan_work'],
        includes_no_unrelated_C40_or_C45_credit=True,complete_new_generation_or_source_proof=False,
        native_binding_or_full_LIVE_payment_proved=False,formal_gain=0)
    return dict(census=result,decoder=decoder,projection=projection,
        all_original_checkpoint_fields_retained_through_census=True,
        source_archive_unchanged=True,complete_checkpoint_numeric_bytes=owner.resident_bytes,
        complete_checkpoint_numeric_entries=owner.resident_entries,
        complete_checkpoint_numeric_roots=len(roots.numeric),python_shallow_bytes=roots.python_shallow_bytes,
        read_only_actual_source_census=True,new_HZ_or_source_receipt_constructed=False)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered source span census')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);branch=BranchPool(pool);pool.charge('c48_fixed_exclusive_evidence',16384)
    started=time.monotonic();report=dict(completed=False,formal_gain=0,generator_executed=False,
        native_executed=False,solver_executed=False,default_changed=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):
            log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('complete source freeze drift')
            result,stats=measured(lambda:complete(pool,branch,emit),
                observe=lambda stats:emit(dict(event='entire_decode_source_owner_and_census_measurement',measurement=stats)))
            report.update(result=result,measurement=stats,completed=True,
                proposal_paid_on_complete_census=result['census']['strict_component_work_payment'],
                whole_fresh_C31_C32_C40_runtime_proved=False,formal_gain=0)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='source_span_census_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,
                branch_diagnostic_work=branch.used,work_parts=dict(pool.parts),branch_work_parts=dict(branch.parts),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
