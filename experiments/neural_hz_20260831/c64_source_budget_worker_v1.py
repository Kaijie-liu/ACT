"""One full source-bound cost preflight; no fresh target generator yet."""
import hashlib
import json
import pickle
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c63_precision_plan_v1 import plan
from experiments.neural_hz_20260831.c64_source_budget_v1 import bound
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c64_source_budget_20260913_v1'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
OLD=EXP/'results/c31_prepared_generator_20260911_v1/closed_hz.pickle'
OLD_SHA='535b579e853f1ac5092e962e1c307644e89739da4cd63f3732593896649128f5'
OLD_PROOF='cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5'
EXPECTED='2174cc144bcfe499ab00e97f67eae2b72d8a3c3d6f994780f63e349d00f36770'


def complete(pool,branch,emit):
    if _sha256(OLD)!=OLD_SHA:raise ValueError('complete qualified C31 source changed')
    with OLD.open('rb') as stream:archive=pickle.load(stream)
    if archive['schema']!='c31_new_checked_prepared_Closed_v1' or hashlib.sha256(archive['proof_bytes']).hexdigest()!=OLD_PROOF:
        raise ValueError('complete source/box/owner proof not bound')
    legacy=archive['fields']
    with SOURCE.open('rb') as stream:saved,decoder=load(stream,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    if (saved['schema']!='c9_integrated_suffix_checkpoint_v1'
            or saved['origin_snapshot_sha256']!='d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
            or not saved['identity_audit']['all_original_coefficients_exact']
            or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']):
        raise ValueError('original complete source/box proof required')
    original_view=dict(fields=legacy,checked_proof_record=(archive['identity']['closed_identity'],archive['proof_bytes']))
    roots=dict(original_C9=saved,complete_C31=original_view);layout=numeric_layout(roots,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete source/oracle entry cap')
    pool.charge('c64_two_complete_original_and_C31_fingerprints',2*int(layout.resident_entries)+2048)
    before,shallow=fingerprint(roots,layout,pool,already_paid=True)
    try:
        reference=plan(saved,pool=branch,enabled=True)
        if reference['report']['identity_sha256']!=EXPECTED:raise ValueError('full original optimum identity changed')
        emit(dict(event='complete_reference_boundary',profile=reference['report'],diagnostic_work=pool.used))
        result=bound(saved,legacy,reference,pool=branch,enabled=True)
        emit(dict(event='complete_source_construction_bound',bound=result,diagnostic_work=pool.used))
        if not result['work_caps_fit']:raise MemoryError('complete source-derived generator bound does not fit unchanged caps')
        return dict(bound=result,decoder=decoder,complete_input_numeric_bytes=layout.resident_bytes,
            complete_input_entries=layout.resident_entries,complete_input_python_shallow_bytes=shallow,
            complete_inputs_unchanged=True,new_target_generator_executed=False,new_physical_HZ=False,formal_gain=0)
    finally:
        unchanged=fingerprint(roots,layout,pool,already_paid=True)[0]==before
        emit(dict(event='complete_input_preservation',unchanged=unchanged))
        if not unchanged or _sha256(SOURCE)!=SOURCE_SHA or _sha256(OLD)!=OLD_SHA:raise ValueError('complete original inputs changed')


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);branch=BranchPool(pool);pool.charge('c64_exclusive_cost_record',16384)
    started=time.monotonic();report=dict(completed=False,formal_gain=0,new_target_generator_executed=False,
        physical_HZ_constructed=False,native_or_solver_executed=False,default_changed=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):
            log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
            data,stats=measured(lambda:complete(pool,branch,emit),observe=lambda s:emit(dict(event='complete_cost_measurement',measurement=s)))
            report.update(completed=True,data=data,measurement=stats)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='source_budget_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,branch_diagnostic_work=branch.used,
                work_parts=dict(pool.parts),branch_work_parts=dict(branch.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report);print(json.dumps(report),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
