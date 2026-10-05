"""Bounded export to existing supported protocol5; no new numeric owner adapter."""
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import signal
import sys
import time
import weakref
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy.sparse as sp
import torch
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c52_signed_first_write_audit_v1 import audit,comparison
from experiments.neural_hz_20260831.c52_neural_source_fixtures_v1 import check_points
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c52_math_archive_transport_20260911_v1'
PRIOR=EXP/'results/c52_signed_first_write_math_20260911_v2'
SOURCE_SHA='45bd7df5cfeaa9deb9607e7af954a07124a4e45b3f2be5258581b6a9db0107ef'


def array_refs(value):
    seen=set();refs=[]
    def visit(x):
        if id(x) in seen:return
        seen.add(id(x))
        if type(x) is np.ndarray:refs.append(weakref.ref(x))
        elif type(x) is SparseHZono:
            for item in vars(x).values():visit(item)
        elif sp.isspmatrix_csr(x):
            for name in ('data','indices','indptr'):visit(getattr(x,name))
        elif type(x) is dict:
            for item in x.values():visit(item)
        elif type(x) in (list,tuple):
            for item in x:visit(item)
    visit(value);return refs


def transaction():
    whole=WorkPool(256_000_000);pool=BranchPool(whole)
    source=PRIOR/'complete_synthetic_sources_and_states.pickle'
    if _sha256(source)!=SOURCE_SHA:raise ValueError('old mathematical artifact changed')
    registered=json.loads((PRIOR/'result.json').read_text())
    pool.charge('c52_complete_archive_read_export_and_metadata',8*source.stat().st_size+65536)
    with source.open('rb') as f:raw=pickle.load(f)
    refs=array_refs(raw);target=DIRECTORY/'complete_owned_protocol5_sources_and_states.pickle'
    with target.open('xb') as f:pickle.dump(raw,f,protocol=5);f.flush();os.fsync(f.fileno())
    del raw;gc.collect()
    if any(r() is not None for r in refs):raise ValueError('old protocol4 numeric arrays are still retained')
    new_sha=_sha256(target)
    with target.open('rb') as f:saved,decoder=load(f,expected_sha256=new_sha,pool=pool,enabled=True)
    rows=[]
    for kind,case,wanted in zip(('chain','shared_add','conv_relu'),saved['cases'],registered['data']['cases'],strict=True):
        for name,proof_name in (('expanded','reference_proof'),('compact','proof')):
            start=whole.used;proof=audit(case['source'],case[name],pool=pool);proof['proof_work']-=start
            if proof!=case[proof_name]:raise ValueError('complete restored source proof changed')
        if case['proof']!=wanted['proof']:raise ValueError('restored proof differs from frozen v2 result')
        points=check_points(kind,case['source'],case['expanded'],case['compact'])
        if points!=wanted['points']:raise ValueError('complete toy-network inverse/output proof changed')
        physical=comparison(case['source'],case['expanded'],case['compact'])
        if not all(physical[n] for n in ('strict_predicate_nnz_decrease','strict_numeric_bytes_decrease','strict_numeric_entries_decrease','strict_combined_reported_accounting_decrease')):
            raise ValueError('restored complete representation reduction failed')
        rows.append(dict(kind=kind,all_source_proofs_recomputed_equal=True,points=points,physical=physical))
    pool.charge('c52_complete_restored_owner_and_final_archive_checks',8*target.stat().st_size+65536)
    roots=collect(SimpleNamespace(),{'complete_restored_artifact':saved});owner=roots.measure()
    if owner.resident_entries>64_000_000 or _sha256(source)!=SOURCE_SHA:raise ValueError('entry cap or old archive changed')
    return dict(cases=rows,old_numeric_arrays_retired=len(refs),all_old_numeric_arrays_retired=True,
        decoder=decoder,whole_work=whole.used,branch_work=pool.used,new_artifact_sha256=new_sha,
        new_artifact_bytes=target.stat().st_size,complete_numeric_bytes=owner.resident_bytes,
        complete_numeric_entries=owner.resident_entries,complete_numeric_roots=len(roots.numeric),
        complete_python_shallow_bytes=roots.python_shallow_bytes,original_request_LIVE_or_native_proved=False)


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));torch.set_num_threads(1);torch.set_num_interop_threads(1)
    def timeout(signum,frame):raise TimeoutError('unchanged240s archive transport cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(240)
    hashes=dict(json.loads((PRIOR/'preregistered.json').read_text())['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('frozen source drift')
    for name in (Path(__file__).name,'C52_ARCHIVE_TRANSPORT_PREREG_20260911.md'):
        hashes[name]=_sha256(EXP/name)
    provenance=_provenance(ROOT);DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,
        provenance=provenance,input_sha256=SOURCE_SHA,transport_only=True,wall_cap_s=240,transient_cap_bytes=1024**3,
        address_space_cap_bytes=16*1024**3,whole_work_cap=256_000_000,branch_cap=200_000_000,formal_gain=0))
    result=dict(completed=False,formal_gain=0,actual_target_or_native_executed=False);started=time.monotonic()
    try:
        data,stats=measured(transaction,observe=lambda s:result.update(measurement=s))
        result.update(completed=True,data=data)
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance)
        _atomic_exclusive_json(DIRECTORY/'result.json',result);signal.alarm(0)
    print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
