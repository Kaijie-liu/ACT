"""Actual source-bound routed/generic planner equality; no HZ or solver writer."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c62_precision_plan_v1 import plan as generic
from experiments.neural_hz_20260831.c63_precision_plan_v1 import plan
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c63_birth_routing_20260913_v1'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
EXPECTED='2174cc144bcfe499ab00e97f67eae2b72d8a3c3d6f994780f63e349d00f36770'


def complete(pool,branch,emit):
    with SOURCE.open('rb') as stream:saved,decoder=load(stream,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    if (saved['schema']!='c9_integrated_suffix_checkpoint_v1'
            or saved['origin_snapshot_sha256']!='d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
            or not saved['identity_audit']['all_original_coefficients_exact']
            or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']):
        raise ValueError('complete original source/box binding required')
    layout=numeric_layout(saved,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete original input entry cap')
    pool.charge('c63_two_complete_original_input_fingerprints',2*int(layout.resident_entries)+2048)
    before,shallow=fingerprint(saved,layout,pool,already_paid=True)
    start=branch.used;t=time.monotonic();reference=generic(saved,pool=branch,enabled=True)
    reference_work=branch.used-start;reference_s=time.monotonic()-t
    emit(dict(event='complete_generic_reference',profile=reference['report'],work=reference_work,elapsed_s=reference_s))
    prior_parts=dict(branch.parts);start=branch.used;t=time.monotonic();candidate=plan(saved,pool=branch,enabled=True)
    new_work=branch.used-start;new_s=time.monotonic()-t
    parts={n:w-prior_parts.get(n,0) for n,w in branch.parts.items() if w!=prior_parts.get(n,0)}
    emit(dict(event='complete_routed_plan',profile=candidate['report'],work=new_work,elapsed_s=new_s,parts=parts))
    pool.charge('c63_independent_complete_factor_and_row_comparison',
        8*(3*saved['hz'].n_cont+saved['hz'].n_eq+sum(v.size for v in reference['raw_hits'].values()))+64*len(reference['parents']))
    for key in ('selected','roots','erased'):
        if not np.array_equal(reference[key],candidate[key]):raise ValueError('complete factor map differs: '+key)
    for key in ('weights','parents','local','tags','numerators'):
        if reference[key]!=candidate[key]:raise ValueError('complete exact factor field differs: '+key)
    for key in ('Ac','Auc','Gc'):
        if not np.array_equal(reference['raw_hits'][key],candidate['raw_hits'][key]):raise ValueError('complete row incidence differs: '+key)
    profile=dict(candidate['report']);routing=profile.pop('birth_routing')
    if profile!=reference['report'] or profile['identity_sha256']!=EXPECTED:
        raise ValueError('full sufficient-statistic/selection binding differs from source-bound reference')
    old_scan=3*saved['hz'].Ac.nnz+8*saved['hz'].n_eq+1024
    new_scan=parts.get('c63_complete_birth_block_paths_and_maps',0)+parts.get('c63_routed_incidence_row',0)
    cost=dict(generic_complete_Ac_scan=int(old_scan),new_complete_route_binding_and_scan=new_scan,
        route_component_work_delta=new_scan-int(old_scan),generic_complete_planner_work=reference_work,
        new_complete_planner_work=new_work,complete_planner_work_delta=new_work-reference_work,
        generic_elapsed_s=reference_s,routed_elapsed_s=new_s,
        timing_not_speed_gate=True,complete_fused_source_generator_upper_bound_proved=False)
    emit(dict(event='complete_route_component_cost',cost=cost))
    if fingerprint(saved,layout,pool,already_paid=True)[0]!=before or _sha256(SOURCE)!=SOURCE_SHA:
        raise ValueError('complete original source changed')
    if new_scan>=old_scan or new_work>=reference_work:raise ValueError('complete routed component including overhead does not strictly improve')
    return dict(profile=profile,routing=routing,cost=cost,new_plan_work_parts=parts,
        complete_factor_and_row_maps_equal=True,complete_original_inputs_unchanged=True,
        original_input_numeric_bytes=layout.resident_bytes,original_input_numeric_entries=layout.resident_entries,
        original_input_python_shallow_bytes=shallow,decoder=decoder,
        source_first_or_HZ_writer_executed=False,full_source_qualification_inherited_not_recomputed=True,formal_gain=0)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);branch=BranchPool(pool);pool.charge('c63_exclusive_result_record',16384)
    started=time.monotonic();report=dict(completed=False,formal_gain=0,physical_HZ_constructed=False,
        new_generator_executed=False,native_or_solver_executed=False,default_changed=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):
            log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
            data,stats=measured(lambda:complete(pool,branch,emit),observe=lambda s:emit(dict(event='complete_route_measurement',measurement=s)))
            report.update(completed=True,data=data,measurement=stats)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='birth_route_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,branch_diagnostic_work=branch.used,
                work_parts=dict(pool.parts),branch_work_parts=dict(branch.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()

