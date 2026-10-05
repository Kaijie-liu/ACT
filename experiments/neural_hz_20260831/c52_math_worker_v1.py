"""One bounded synthetic Neural-HZ source transaction; no actual target/native."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import build,reference,source_hash
from experiments.neural_hz_20260831.c52_signed_first_write_audit_v1 import audit,comparison
from experiments.neural_hz_20260831.c52_neural_source_fixtures_v1 import fixture,check_points
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c52_signed_first_write_math_20260911_v1'


def transaction(emit):
    payload=[];summaries=[]
    for kind in ('chain','shared_add','conv_relu'):
        source=fixture(kind,128);before=source_hash(source)
        expanded=reference(source);compact=build(source,enabled=True)
        reference_proof=audit(source,expanded);proof=audit(source,compact);physical=comparison(source,expanded,compact)
        points=check_points(kind,source,expanded,compact)
        gates=('strict_predicate_nnz_decrease','strict_numeric_bytes_decrease','strict_numeric_entries_decrease','strict_combined_reported_accounting_decrease')
        if not all(physical[n] for n in gates):raise ValueError('complete prototype representation gate failed')
        if source_hash(source)!=before:raise ValueError('original prototype source changed')
        summary=dict(kind=kind,original_n_cont=source['nc'],compact_n_cont=compact['hz'].n_cont,
            n_binary=source['nb'],source_sha256=before,proof=proof,physical=physical,points=points,
            construction=compact['report'],reference_construction=expanded['report'])
        summaries.append(summary);payload.append(dict(source=source,expanded=expanded,compact=compact,
            reference_proof=reference_proof,proof=proof,physical=physical,points=points))
        emit('complete_synthetic_source_proved',summary)
    # Every common source and both sides remain in this measured transaction.
    roots=collect(SimpleNamespace(),{'all_complete_source_and_reference_and_compact_states':payload});owners=roots.measure()
    if owners.resident_entries>64_000_000:raise MemoryError('complete prototype retained entry cap exceeded')
    before=roots.fingerprint
    artifact=DIRECTORY/'complete_synthetic_sources_and_states.pickle'
    with artifact.open('xb') as f:pickle.dump(dict(schema='c52_synthetic_source_states_v1',cases=payload,formal_gain=0),f,protocol=4);f.flush();os.fsync(f.fileno())
    if collect(SimpleNamespace(),{'all_complete_source_and_reference_and_compact_states':payload}).fingerprint!=before:
        raise ValueError('complete source/state roots changed during proof or save')
    return dict(cases=summaries,complete_retained_numeric_bytes=owners.resident_bytes,
        complete_retained_numeric_entries=owners.resident_entries,complete_python_shallow_bytes=roots.python_shallow_bytes,
        complete_numeric_roots=len(roots.numeric),artifact_sha256=_sha256(artifact),artifact_bytes=artifact.stat().st_size,
        all_common_source_reference_compact_proof_and_report_roots_retained=True)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered math-only output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));torch.set_num_threads(1);torch.set_num_interop_threads(1)
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('complete frozen source drift')
    started=time.monotonic();result=dict(completed=False,actual_target_executed=False,native_or_solver_executed=False,
        full_qualification_suite_passed=False,formal_gain=0)
    with (DIRECTORY/'events.jsonl').open('x') as f:
        def emit(name,values):f.write(json.dumps(dict(event=name,elapsed_s=time.monotonic()-started,**values),sort_keys=True)+'\n');f.flush()
        try:
            data,stats=measured(lambda:transaction(emit),observe=lambda s:emit('complete_prototype_transaction_measurement',s))
            result.update(completed=True,data=data,measurement=stats,provenance=freeze['provenance'])
        except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc));emit('math_transaction_rejected',result['failure'])
        finally:
            result['wall_s']=time.monotonic()-started;_atomic_exclusive_json(DIRECTORY/'result.json',result)
    print(json.dumps({'completed':result['completed'],'failure':result.get('failure'),'formal_gain':0}),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
