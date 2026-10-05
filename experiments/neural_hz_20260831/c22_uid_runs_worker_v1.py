"""Complete MAIN run-index proof on a successful sealed C10 source; no generator."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.neural_hz_20260831.c22_uid_runs_v1 import build,validate,row_for_uid,uid_for_row
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c10_fused_emission_v1 import FusedIntegrated
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c10_portable_binding_v1 import identity
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import row_uid_tables
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c22_uid_runs_20260911_v1'
FUSED=EXP/'results/c10_fused_emission_20260908_v1/fused_hz.pickle'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    record={'completed':False,'formal_gain':0,'generator_executed':False,
        'graph_fields_retired':False,'solver_executed':False,'native_ingestion_executed':False,
        'source_sha256':freeze['source_sha256'],'provenance':freeze['provenance']}
    started=time.monotonic()
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
            raise ValueError('frozen index/source drift')
        if _sha256(FUSED)!='24b22d8a06c0c4d51525dd9d5ee0902941c40f20cc6a711e6231673e3efa9d08':
            raise ValueError('sealed successful source drift')
        with FUSED.open('rb') as f: saved=pickle.load(f)
        if saved['schema']!='c10_fused_emission_checkpoint_v1':
            raise ValueError('wrong source schema')
        fields=saved['fields']
        candidate=FusedIntegrated(**fields,origin_binding=expression_binding(fields['expression']))
        candidate.seal=candidate.fingerprint()
        before=identity(candidate)
        auth_elapsed=time.monotonic()-started
        main_first=candidate.old_n_eq+len(candidate.ineq_roots)
        width=sum(n['width'] for n in candidate.nodes)
        radix_base=main_first+width
        # Only the diagnostic holder receives a derived scalar; never modify
        # the sealed original report or its full graph arrays.
        holder=SimpleNamespace(**vars(candidate))
        holder.report={**candidate.report,'radix_uid_base':radix_base}
        pool=WorkPool(0,0)
        def execute():
            pool.charge('complete_uid_metadata',32*(width+candidate.logical_n_cont-candidate.old_n_cont
                +candidate.hz.n_eq+candidate.hz.n_ineq))
            eq,le=row_uid_tables(holder)
            words,summary=build(eq,main_first,radix_base,pool=pool)
            if summary['tracked_MAIN_rows']!=142559 or len(eq)!=143737 or len(le)!=2300:
                raise ValueError('complete source row population differs')
            word_sha=hashlib.sha256(words.tobytes()).hexdigest()
            inverse=np.full(width,-1,np.int64)
            active=(eq>=main_first)&(eq<radix_base)
            inverse[eq[active]-main_first]=np.flatnonzero(active)
            probes=len(inverse)+len(eq)
            probe_work=probes*(8*max(1,len(words).bit_length())+8)
            if pool.used+probe_work>pool.capacity:
                raise MemoryError('complete all-UID/all-row proof cannot fit diagnostic cap')
            # ALL reserved MAIN UIDs, including unused coordinates and aliases,
            # are checked; not just positive dictionary members.
            for offset,expected in enumerate(inverse):
                actual=row_for_uid(words,main_first+offset,pool=pool)
                if actual!=(None if expected<0 else int(expected)):
                    raise ValueError('compact index changes a MAIN UID or hole')
            for row,uid in enumerate(eq):
                actual=uid_for_row(words,row,pool=pool)
                expected=int(uid) if main_first<=uid<radix_base else None
                if actual!=expected:
                    raise ValueError('compact index changes a physical row or non-MAIN gap')
            validate(words)
            if hashlib.sha256(words.tobytes()).hexdigest()!=word_sha:
                raise ValueError('index mutated during complete proof')
            summary.update(all_reserved_MAIN_UIDs_checked=width,all_EQ_rows_checked=len(eq),
                all_old_INEQ_UIDs_derived=len(le),unused_or_erased_UIDs_checked=int(np.count_nonzero(inverse<0)),
                main_first=main_first,radix_uid_base=radix_base,word_sha256=word_sha,
                complete_bidirectional_index_proved=True,complete_probe_work=probe_work,
                diagnostic_work=pool.used,diagnostic_work_parts=dict(pool.parts))
            return words,summary
        (words,report),construction=measured_build(execute)
        if identity(candidate)!=before:
            raise ValueError('diagnostic mutated original source/graph/lineage')
        with (DIRECTORY/'uid_runs.npz').open('xb') as f:
            np.savez(f,words=words); f.flush(); os.fsync(f.fileno())
        report.update(full_original_dictionary_retained=True,source_unchanged=True,
            source_portable_identity=before,original_authentication_and_load_s=auth_elapsed)
        _atomic_exclusive_json(DIRECTORY/'index_audit.json',report)
        record.update(completed=True,report=report,construction=construction)
    except Exception as exc:
        record['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'result.json',record)
        print(json.dumps({'completed':record['completed'],'failure':record.get('failure'),'formal_gain':0}),flush=True)
    if not record['completed']: raise SystemExit(1)


if __name__=='__main__': main()
