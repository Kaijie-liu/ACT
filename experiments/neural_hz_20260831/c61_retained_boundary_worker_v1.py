"""One source/inverse authenticated consumer profile; automatically retained."""
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c61_retained_boundary_v1 import assess
from experiments.neural_hz_20260831.c58_reconstruction_equations_v1 import decode
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c61_retained_boundary_20260912_v1'
SOURCE=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
INVERSE=EXP/'results/c58_owned_inverse_equations_20260912_v2/inverse_equations.npz'
INVERSE_SHA='1ea67792278d07179c0400d7c25dbcdbcfdf605b308709b53a72306dc88be26f'
PROOF=EXP/'results/c58_owned_inverse_equations_20260912_v2/result.json'
PROOF_SHA='4889e37adf988d75c6b4e640011746ee0f3f31d60b05af42839bfd83a16057c5'


PRIOR=EXP/'results/c60_native_sharing_census_20260912_v1/result.json'
PRIOR_SHA='04eb6fb80513e16ca40f27460530a008241564566f471ee3b9f5bf3563fad930'
LEGACY=EXP/'results/c31_prepared_generator_20260911_v1/result.json'
LEGACY_SHA='c0d9a310c5d7a814f903ed48686d8db43bd7724a9c4e270d084a4b60a6c7a20a'


def complete(pool,branch,emit):
    if _sha256(LEGACY)!=LEGACY_SHA:raise ValueError('frozen C31 comparator changed')
    legacy=json.loads(LEGACY.read_text())['generation_report']['alias_quotient']
    if _sha256(PRIOR)!=PRIOR_SHA:raise ValueError('frozen complete consumer profile changed')
    prior=json.loads(PRIOR.read_text())
    if not prior['completed'] or not prior['data']['complete_inputs_unchanged']:
        raise ValueError('completed original consumer proof required')
    if _sha256(INVERSE)!=INVERSE_SHA or _sha256(PROOF)!=PROOF_SHA:raise ValueError('external inverse/proof binding differs')
    inherited=json.loads(PROOF.read_text())
    if (not inherited['completed'] or not inherited['data']['complete_checkpoint_unchanged']
            or inherited['data']['decoder']['checkpoint_sha256']!=SOURCE_SHA):raise ValueError('successful source-bound inverse qualification required')
    original_inverse=inherited['data']['census']['general_scalar_diagnostic']
    if original_inverse['inverse_archive_sha256']!=INVERSE_SHA:raise ValueError('inverse archive has no source-bound proof')
    pool.charge('c59_inverse_file_decode_owners_preflight',4*INVERSE.stat().st_size+4096)
    payload=np.frombuffer(INVERSE.read_bytes(),np.uint8).copy();packet=decode(payload)
    if packet['seal']!=original_inverse['inverse_packet_sha256'] or packet['source_binding']!=original_inverse['chain_identity_sha256']:
        raise ValueError('original inverse identity differs')
    with SOURCE.open('rb') as f:saved,decoder=load(f,expected_sha256=SOURCE_SHA,pool=pool,enabled=True)
    if (saved['schema']!='c9_integrated_suffix_checkpoint_v1'
            or saved['origin_snapshot_sha256']!='d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
            or saved['identity_audit']['all_original_coefficients_exact'] is not True
            or saved['identity_audit']['all_redundant_main_and_radix_boxes_proved'] is not True):
        raise ValueError('qualified complete original source/box checkpoint required')
    before=collect(SimpleNamespace(),dict(complete_original_checkpoint=saved,inverse_packet=packet,inverse_archive=payload));owner=before.measure()
    pool.charge('c59_two_complete_input_owner_hash_traversals',2*owner.resident_entries+128*len(before.numeric)+1024)
    if owner.resident_entries>64_000_000:raise MemoryError('complete inputs entry cap')
    emit(dict(event='complete_authenticated_source_inverse_inputs',resident_bytes=owner.resident_bytes,
        resident_entries=owner.resident_entries,numeric_roots=len(before.numeric),python_shallow_bytes=before.python_shallow_bytes,
        inherited_C58_source_qualification=True,inherited_C58_whole_work=251916898))
    result=assess(saved,packet,pool=branch,enabled=True,observe=emit)
    if (result['legacy_local'],result['legacy_eligible'],result['legacy_selected'],result['legacy_incidence_hits'])!=tuple(
            legacy[n] for n in ('local_aliases','eligible_aliases','selected_aliases','alias_products_checked')):
        raise ValueError('complete old C31 comparator differs')
    if result['raw_nodes']!=prior['data']['profile']['fresh_original_defining_rows_proved']:
        raise ValueError('complete C60 source cohort differs')
    if result['raw_nodes']!=original_inverse['counts']['singletons']:
        raise ValueError('complete original raw singleton cohort differs')
    after=collect(SimpleNamespace(),dict(complete_original_checkpoint=saved,inverse_packet=packet,inverse_archive=payload))
    if before.fingerprint!=after.fingerprint or owner!=after.measure():raise ValueError('complete original inputs changed')
    if _sha256(SOURCE)!=SOURCE_SHA or _sha256(INVERSE)!=INVERSE_SHA:raise ValueError('source artifacts changed')
    return dict(profile=result,decoder=decoder,complete_inputs_unchanged=True,
        complete_input_numeric_bytes=owner.resident_bytes,complete_input_numeric_entries=owner.resident_entries,
        complete_input_python_shallow_bytes=before.python_shallow_bytes,
        original_source_and_C58_full_qualification_inherited_not_recomputed=True,
        fresh_original_defining_rows_and_all_forward_consumers_checked=True,
        inherited_C58_source_producer_whole_work=251916898,source_first_or_native_writer_executed=False)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);branch=BranchPool(pool);pool.charge('c59_exclusive_evidence_record',16384)
    started=time.monotonic();report=dict(completed=False,formal_gain=0,new_generator_executed=False,
        native_or_solver_executed=False,full_qualification_suite_passed=False,default_changed=False,native_candidate_admitted=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):
            log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
            data,stats=measured(lambda:complete(pool,branch,emit),observe=lambda stats:emit(dict(event='complete_consumer_measurement',measurement=stats)))
            report.update(completed=True,data=data,measurement=stats)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='consumer_profile_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,
                branch_diagnostic_work=branch.used,work_parts=dict(pool.parts),branch_work_parts=dict(branch.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
