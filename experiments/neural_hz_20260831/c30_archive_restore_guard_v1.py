"""Read-only fresh-process restoration of the actual C30 HZ and all lineage."""

import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
PRIOR=EXP/'results/c30_first_write_20260911_v1'
DIRECTORY=EXP/'results/c30_archive_restore_guard_20260911_v1'
INPUTS={'spliced_hz.pickle':'ed1099a692ef4dad2f0f6b1a04591dbc258cc2edc8083919ce86b465f208e145',
    'result.json':'7b0af27cfac7ae0826a740229d746287d626b97886d522a2bf47d4f289c5f5da',
    'exit.json':'33a4bd117a2279a7beea59036cd26661d330cbb1f006242db332c1a43ad6b560'}


def equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint8),b.view(np.uint8))


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    DIRECTORY.mkdir()
    freeze=dict(input_sha256=INPUTS,guard_source_sha256=_sha256(Path(__file__)),provenance=_provenance(ROOT),
        writer_generator_native_solver_execution_authorized=False,formal_gain=0)
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',freeze)
    started=time.monotonic();record=dict(completed=False,formal_gain=0)
    try:
        if any(_sha256(PRIOR/n)!=sha for n,sha in INPUTS.items()):raise ValueError('C30 completed archive drift')
        result=json.loads((PRIOR/'result.json').read_text());ending=json.loads((PRIOR/'exit.json').read_text())
        if (not result['completed'] or ending.get('worker_exit_code')!=0 or ending.get('tests_exit_code')!=0
                or ending.get('timeout_s') or ending['source_drift'] or ending['provenance_drift']
                or result['provenance']!=freeze['provenance']):
            raise ValueError('not a completed matching-source qualification')
        if any(_sha256(EXP/n)!=sha for n,sha in result['source_sha256'].items()):
            raise ValueError('complete original source/reference manifest drift')
        with (PRIOR/'spliced_hz.pickle').open('rb') as f:saved=pickle.load(f)
        with (EXP/'results/c15_unit_row_splice_20260910_v1/spliced_hz.pickle').open('rb') as f:ref=pickle.load(f)
        with (EXP/'results/c26_transplant_census_20260911_v1/lineage.pickle').open('rb') as f:tags=pickle.load(f)
        with (EXP/'results/c25_live_relu_20260911_v1/relu78.pickle').open('rb') as f:live=pickle.load(f)
        hz=saved['hz'];lineage=saved['lineage'];oracle=ref['candidate'];post=live['post_relu_hz']
        oracle.validate();lineage.validate();tags['draft'].validate()
        if (saved['schema']!='c30_provisional_actual_spliced_HZ_v1'
                or saved['source_sha256']!=result['source_sha256'] or saved['provenance']!=freeze['provenance']
                or source_digest(hz)!=saved['new_HZ_sha256'] or saved['new_HZ_sha256']!=result['actual_new_HZ_sha256']
                or source_digest(post)!=saved['source_post_HZ_sha256']
                or saved['closed_identity']!=tags['closed_identity']
                or saved['complete_UID_box_reconstruction_transfer']!=result['complete_proof_transfer']
                or lineage.fingerprint()!=result['lineage_fingerprint']
                or lineage.fingerprint()!=tags['draft'].fingerprint()
                or saved['independent_C15_archive_sha256']!=_sha256(EXP/'results/c15_unit_row_splice_20260910_v1/spliced_hz.pickle')
                or saved['independent_C26_lineage_sha256']!=_sha256(EXP/'results/c26_transplant_census_20260911_v1/lineage.pickle')):
            raise ValueError('actual HZ/full lineage/proof restoration binding changed')
        count=0;pool=WorkPool(256_000_000)
        for name in ('Ac','Ab','Auc','Aub'):
            a,b=getattr(hz,name),getattr(oracle.hz,name)
            pool.charge('all_restored_predicate_buffer_bits',4*int(a.nnz)+4*len(a.indptr))
            if a.shape!=b.shape or not all(equal(getattr(a,k),getattr(b,k)) for k in ('data','indices','indptr')):
                raise ValueError('restored complete independent matrix mismatch: '+name)
            count+=int(a.nnz)
        pool.charge('all_restored_RHS_outputs_lineage',4*(hz.n_eq+hz.n_ineq+hz.n_out+hz.Gc.nnz+hz.Gb.nnz+2*len(lineage.eq_roots)))
        if not equal(hz.b,oracle.hz.b) or not equal(hz.ub,oracle.hz.ub) or not equal(hz.c,post.c):
            raise ValueError('restored RHS or actual native output changed')
        for name in ('Gc','Gb'):
            a,b=getattr(hz,name),getattr(post,name)
            if a.shape!=b.shape or not all(equal(getattr(a,k),getattr(b,k)) for k in ('data','indices','indptr')):
                raise ValueError('restored actual output map changed')
        if (count!=result['all_written_predicate_coefficients_checked'] or hz.n_bin!=1350
                or hz.n_cont!=post.n_cont or hz.frame_id!=post.frame_id or not hz.exact
                or len(lineage.columns)!=268):raise ValueError('restored frame/population incomplete')
        record.update(completed=True,all_restored_predicate_coefficients=count,
            all_restored_lineage_slots=len(lineage.eq_roots),all_restored_unit_pairs=len(lineage.columns),
            original_binary_factors=hz.n_bin,full_independent_matrix_RHS_output_lineage_identity=True,
            exact_UID_box_reconstruction_proof_bound=True,diagnostic_work=pool.used,
            new_writer_executed=False,new_generator_executed=False,new_native_relu_executed=False,
            solver_executed=False,live_admission_certificate=False,whole_live_path_proved=False)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_sha256=INPUTS,guard_source_sha256=freeze['guard_source_sha256'],provenance=freeze['provenance'])
        _atomic_exclusive_json(DIRECTORY/'result.json',record);print(json.dumps(record),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
