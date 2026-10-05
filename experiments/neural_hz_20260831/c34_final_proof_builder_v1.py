"""Offline bind actual C33 post -> original Dense/ASSERT -> independent C15 final."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import sparse_hz_linear
from experiments.neural_hz_20260831.c32_splice_binding_v1 import admit
from experiments.neural_hz_20260831.c34_terminal_binding_v1 import suffix,array
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c34_changed_terminal_20260911_v1'
ANCHORS={
    'results/c33_live_splice_20260911_v1/relu78.pickle':'96871d1bae328919eab0cd36bdad15dc6f40323325848bdb5095e0f8239ffa6b',
    'results/c33_live_splice_20260911_v1/qualification.json':'de1bbbea9a27fb126bbf748701464d2123108ce820e20c70b3b01a1730f7ad3e',
    'results/c33_live_splice_20260911_v1/restore_guard.json':'3b8c31b75d968a998f1b9efdb70dd75ca0864156a0f8e60707f2402ceb7a4de6',
    'results/c33_live_splice_20260911_v1/exit.json':'970dd7d5733a9abd5e11cea8e33cdecdce62e4d29602f51b8e8b1b3c6c7e7736',
    'results/c15_unit_row_splice_20260910_v1/spliced_hz.pickle':'6b93e5f929be286d762f9c0779c0b8d505440325e9b786f943df73ff0022dc30',
}


def build(*,enabled=False):
    if not enabled:return None
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('complete independent input/proof archive drift')
    q=json.loads((EXP/'results/c33_live_splice_20260911_v1/qualification.json').read_text())
    e=json.loads((EXP/'results/c33_live_splice_20260911_v1/exit.json').read_text())
    g=json.loads((EXP/'results/c33_live_splice_20260911_v1/restore_guard.json').read_text())
    if (not q['passed'] or not q['whole_live_path_proved'] or not g['completed'] or e['worker_exit_code']!=0
            or e['restore_exit_code']!=0 or e['source_drift'] or e['provenance_drift'] or q['terminal_solve_executed']):
        raise ValueError('C33 exact/native/physical prerequisite failed')
    with (EXP/'results/c33_live_splice_20260911_v1/relu78.pickle').open('rb') as f:saved=pickle.load(f)
    with (EXP/'results/c15_unit_row_splice_20260910_v1/spliced_hz.pickle').open('rb') as f:old=pickle.load(f)
    new,_=admit(enabled=True,**saved['spliced_state_fields']);candidate=old['candidate'];candidate.validate()
    pool=WorkPool(256_000_000)
    producers=[lid for lid,hz in saved['hz_cache'].items() if hz is new.hz]
    if len(producers)!=1:raise ValueError('archived actual post cache is not unique')
    dense,assertion,signature=suffix(saved['net'],producers[0],pool=pool)
    weight=array(dense.params['weight']);bias=array(dense.params['bias']).reshape(-1)
    pool.charge('offline_original_final_affine_application',16*weight.size+16*new.hz.n_out)
    final=sparse_hz_linear(new.hz,sp.csr_matrix(weight),bias)
    if source_digest(final)!=source_digest(candidate.hz):
        raise ValueError('ALL final affine/predicate fields differ from independent C15 result')
    if (old['input_hz'].frame_id!=final.frame_id or old['input_hz'].n_cont>new.lineage.old_n_cont
            or q['actual_hz_sha256']!=source_digest(new.hz)):
        raise ValueError('independent original input/source/global identities differ')
    # C15 bound the very same predicate matrix and input before C33. The
    # complete C33 transfer already binds its all-row box/Fraction proof; this
    # adds only the unchanged original final affine image/property signature.
    result=dict(schema='c34_independent_final_affine_transfer_v1',completed=True,
        whole_actual_spliced_source_bound=True,all_final_output_and_predicate_bits_equal=True,
        unchanged_original_input_and_property_bound=True,post_HZ_sha256=source_digest(new.hz),
        final_HZ_sha256=source_digest(final),input_HZ_sha256=source_digest(old['input_hz']),
        input_shape=list(old['input_shape']),suffix_signature=signature,
        underlying_splice_transfer_sha256=new.transfer_proof_sha256,
        native_lowered_n_cont=q['native_ingestion']['lowered_n_cont'],
        native_lowered_n_bin=q['native_ingestion']['lowered_n_bin'],
        independent_component_sha256=ANCHORS['results/c15_unit_row_splice_20260910_v1/spliced_hz.pickle'],
        input_sha256=ANCHORS,diagnostic_work=pool.used,diagnostic_work_parts=pool.parts,
        new_solver_call_executed=False,root_domain_base_feasibility_proved=False,formal_gain=0)
    return json.dumps(result,sort_keys=True,allow_nan=False).encode()


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered proof output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();record=dict(completed=False,formal_gain=0)
    try:
        freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('builder freeze drift')
        raw=build(enabled=True)
        with (DIRECTORY/'final_proof.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        record.update(completed=True,final_proof_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'final_proof_result.json',record);print(json.dumps(record),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
