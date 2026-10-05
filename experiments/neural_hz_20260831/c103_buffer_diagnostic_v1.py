"""Bounded exporter-metadata lifetime diagnosis; cannot admit the closed source."""
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c103_buffer_diagnostic_20260913_v1'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    if RUN.exists():raise FileExistsError(RUN)
    old=EXP/'results/c103_fused_row_20260913_v1'
    exit_record=json.loads((old/'exit.json').read_text())
    if exit_record.get('fresh_terminal_exit')!=1 or exit_record['source_drift'] or exit_record['provenance_drift']:
        raise ValueError('honestly closed original source required')
    freeze=json.loads((old/'preregistered.json').read_text());hashes=dict(freeze['source_sha256'])
    names=['c103_buffer_diagnostic_v1.py','C103_BUFFER_DIAGNOSTIC_PREREG_20260913.md']
    hashes.update({n:_sha256(EXP/n) for n in names})
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('frozen source drift')
    provenance=_provenance(ROOT)
    if provenance!=freeze['provenance']:raise ValueError('production drift')
    RUN.mkdir();_atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,
        provenance=provenance,array_count=20000,shape=[3],passes=2,formal_gain=0,
        source_target_closed=True,whole_work_cap=256_000_000,both_transient_bytes=1024**3))
    started=time.monotonic();record=dict(completed=False,formal_gain=0,numpy=np.__version__)
    pool=WorkPool(256_000_000);observations=[]
    try:
        def build():
            pool.charge('complete_zero_array_allocation',64*20000)
            arrays=[np.zeros(3,np.float64) for _ in range(20000)]
            before=tracemalloc.get_traced_memory()[0];trace=[before]
            for _ in range(2):
                pool.charge('complete_buffer_export_release',128*len(arrays))
                for array in arrays:
                    view=memoryview(array);view.release();del view
                trace.append(tracemalloc.get_traced_memory()[0])
            pool.charge('complete_original_numeric_check',32*len(arrays))
            if any(a.dtype!=np.float64 or a.shape!=(3,) or np.any(a!=0) for a in arrays):raise ValueError('original arrays differ')
            return arrays,trace
        (arrays,trace),stats=measured(build,observe=observations.append)
        record.update(completed=True,array_count=len(arrays),all_original_arrays_retained=True,
            all_numeric_values_unchanged=True,trace_current=trace,first_export_retained_bytes=trace[1]-trace[0],
            second_export_extra_bytes=trace[2]-trace[1],measurement=stats,
            metadata_persistence_observed=trace[1]-trace[0]>64*len(arrays))
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,work=pool.used,work_parts=pool.parts,observations=observations,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance)
        _atomic_exclusive_json(RUN/'result.json',record)
        _atomic_exclusive_json(RUN/'exit.json',dict(completed=record['completed'],formal_gain=0,
            source_drift=record['source_drift'],provenance_drift=record['provenance_drift'],
            result_sha256=_sha256(RUN/'result.json')))
        print(json.dumps(record),flush=True)
    if not record['completed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
