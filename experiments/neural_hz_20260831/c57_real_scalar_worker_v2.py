"""One full original checkpoint read and logical-source census, auto-preserved."""
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c57_real_scalar_census_v2 import assess
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c57_shared_scalar_census_20260912_v2'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'


def complete(pool,branch,emit):
    with SOURCE.open('rb') as f:saved,decoder=load(f,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    if (saved['schema']!='c9_integrated_suffix_checkpoint_v1'
            or saved['origin_snapshot_sha256']!='d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
            or saved['identity_audit']['all_original_coefficients_exact'] is not True
            or saved['identity_audit']['all_redundant_main_and_radix_boxes_proved'] is not True):
        raise ValueError('wrong independently qualified original source/box checkpoint')
    emit(dict(event='complete_original_checkpoint_decoded',decoder=decoder))
    before=collect(SimpleNamespace(),dict(complete_original_checkpoint=saved));owner=before.measure()
    pool.charge('c57_two_complete_checkpoint_owner_hash_traversals',2*owner.resident_entries+128*len(before.numeric)+1024)
    if owner.resident_entries>64_000_000:raise MemoryError('complete checkpoint entry cap')
    emit(dict(event='complete_original_checkpoint_owners_checked',resident_bytes=owner.resident_bytes,
        resident_entries=owner.resident_entries,numeric_roots=len(before.numeric),python_shallow_bytes=before.python_shallow_bytes))
    result=assess(saved,pool=branch,enabled=True,observe=emit)
    after=collect(SimpleNamespace(),dict(complete_original_checkpoint=saved));after_owner=after.measure()
    if before.fingerprint!=after.fingerprint or owner!=after_owner:raise ValueError('complete original checkpoint changed')
    if _sha256(SOURCE)!=SOURCE_SHA:raise ValueError('original source archive changed')
    return dict(census=result,decoder=decoder,source_archive_unchanged=True,
        inherited_original_HZ_sha256='16337ddcef267f17eff8313db81e92a8049589613a9031db799453b759089ba2',
        original_HZ_digest_freshly_recomputed=False,complete_checkpoint_unchanged=True,
        all_original_checkpoint_fields_retained_through_census=True,
        checkpoint_numeric_bytes=owner.resident_bytes,checkpoint_numeric_entries=owner.resident_entries,
        checkpoint_numeric_roots=len(before.numeric),checkpoint_python_shallow_bytes=before.python_shallow_bytes,
        inherited_original_matrix_and_box_identity=saved['identity_audit'],
        old_matrix_proof_freshly_recomputed=False,read_only_actual_source_census=True,
        new_HZ_native_solver_or_witness_executed=False)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered census output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);branch=BranchPool(pool);pool.charge('c57_exclusive_evidence_record',16384)
    started=time.monotonic();report=dict(completed=False,formal_gain=0,new_generator_executed=False,
        native_or_solver_executed=False,full_qualification_suite_passed=False,default_changed=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):
            log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
            data,stats=measured(lambda:complete(pool,branch,emit),
                observe=lambda stats:emit(dict(event='entire_source_census_measurement',measurement=stats)))
            report.update(completed=True,data=data,measurement=stats)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='census_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,
                branch_diagnostic_work=branch.used,work_parts=dict(pool.parts),branch_work_parts=dict(branch.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
