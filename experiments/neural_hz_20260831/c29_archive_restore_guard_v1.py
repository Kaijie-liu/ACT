"""Fresh-process read-only validation of every archived C29 head code."""

import json
import math
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c29_prepared_row_v1 import decode_head
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
PRIOR=EXP/'results/c29_prepared_rows_20260911_v1'
DIRECTORY=EXP/'results/c29_archive_restore_guard_20260911_v1'
OLD=EXP/'results/c9_live_relu_20260906_v1/relu78.pickle'
LIVE=EXP/'results/c25_live_relu_20260911_v1/relu78.pickle'
INPUTS={'head_codes.npz':'d6d45f76b25f980c828cd054e0107f7b0cb80f0de60d9b8504283f557e739127',
    'result.json':'57f8eb6b3e6e5b5f1ed0ed90209483c080e9fbd92b3612a437de7210d7f54dbe',
    'exit.json':'c5193e8ddcf7b68d8ecdc605132a56214ac0cb0a297d1d95a28cfe38a6af097d'}


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    DIRECTORY.mkdir()
    freeze={'input_sha256':INPUTS,'guard_source_sha256':_sha256(Path(__file__)),'provenance':_provenance(ROOT),
        'new_encoding_or_generator_native_solver_authorized':False,'formal_gain':0}
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',freeze)
    started=time.monotonic();record={'completed':False,'formal_gain':0}
    try:
        if any(_sha256(PRIOR/n)!=sha for n,sha in INPUTS.items()):raise ValueError('completed archive drift')
        prior=json.loads((PRIOR/'result.json').read_text());ending=json.loads((PRIOR/'exit.json').read_text())
        if (not prior['completed'] or ending.get('worker_exit_code')!=0 or ending.get('tests_exit_code')!=0
                or ending.get('timeout_s') or ending['source_drift'] or ending['provenance_drift']
                or prior['provenance']!=freeze['provenance']):raise ValueError('not a completed matching-source qualification')
        if any(_sha256(EXP/n)!=sha for n,sha in prior['source_sha256'].items()):raise ValueError('complete source manifest changed')
        with OLD.open('rb') as handle:old=pickle.load(handle)
        with LIVE.open('rb') as handle:live=pickle.load(handle)
        sources=[('before_alias',old['preactivation_hz']),('after_alias_and_actual_phase',live['post_relu_hz'])]
        if {label:source_digest(hz) for label,hz in sources}!=prior['source_HZ_sha256']:
            raise ValueError('complete original source identities differ')
        checked=0;pool=WorkPool(256_000_000)
        with np.load(PRIOR/'head_codes.npz',allow_pickle=False) as saved:
            keys={label+suffix for label,hz in sources for suffix in ('_EQ','_INEQ')}
            if set(saved.files)!=keys:raise ValueError('complete source head-array set changed')
            for label,hz in sources:
                for suffix,matrix in (('_EQ',hz.Ac),('_INEQ',hz.Auc)):
                    codes=saved[label+suffix]
                    if codes.dtype!=np.dtype(np.uint8) or codes.shape!=(matrix.shape[0],):
                        raise ValueError('restored head-array geometry/dtype changed')
                    pool.charge('all_restored_source_heads',16*len(codes))
                    for row,code in enumerate(codes):
                        a,b=int(matrix.indptr[row]),int(matrix.indptr[row+1])
                        expected=None if a==b or math.frexp(abs(float(matrix.data[a])))[0]!=.5 else float(matrix.data[a])
                        if decode_head(int(code))!=expected:raise ValueError('restored head differs from actual complete source')
                        checked+=1
        if checked!=prior['all_physical_rows_checked']:raise ValueError('restoration did not cover whole row population')
        record.update(completed=True,all_source_rows_restored=checked,all_four_arrays_equal_actual_source=True,
            complete_source_manifest_checked=True,diagnostic_work=pool.used,head_tables_are_diagnostic_only=True,
            row_revisions_or_pre_post_alignment_not_certified=True,new_encoding_executed=False,
            new_HZ_generator_executed=False,new_native_relu_executed=False,solver_executed=False,
            live_admission_certificate=False)
    except Exception as exc:record['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_sha256=INPUTS,guard_source_sha256=freeze['guard_source_sha256'],provenance=freeze['provenance'])
        _atomic_exclusive_json(DIRECTORY/'result.json',record);print(json.dumps(record),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
