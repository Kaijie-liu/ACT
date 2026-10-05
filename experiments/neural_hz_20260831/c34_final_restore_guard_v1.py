"""Restore complete final source/input/property/reconstruction after return or kill."""
import json
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c32_splice_binding_v1 import admit
from experiments.neural_hz_20260831.c34_terminal_binding_v1 import verify
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c34_changed_terminal_20260911_v1'


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered final restore target')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();report=dict(completed=False,solver_executed=False,formal_gain=0)
    try:
        freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('guard source freeze drift')
        gate=json.loads((DIRECTORY/'terminal_gate.json').read_text())
        if not gate['terminal_gates_passed'] or _sha256(DIRECTORY/'native_state.pickle')!=gate['native_state_sha256']:
            raise ValueError('completed pre-terminal source checkpoint changed/missing')
        with (DIRECTORY/'native_state.pickle').open('rb') as f:saved=pickle.load(f)
        new,source=admit(enabled=True,**saved['spliced_state_fields'])
        if saved['final_proof_sha256']!=_sha256(DIRECTORY/'final_proof.json'):
            raise ValueError('independent final proof bytes changed')
        state=dict(lifted=new,native_block_calls=1,layer=saved['selected_layer'],
            tf=SimpleNamespace(_net=saved['net'],_sparse_hz_cache=saved['hz_cache'],_sparse_affine_expr_cache=saved['expr_cache']))
        pool=WorkPool(256_000_000)
        final=verify(state,saved['final_hz'],saved['input_hz'],saved['output_spec'],saved['terminal_kwargs'],
            saved['final_proof_bytes'],saved['final_proof_sha256'],pool=pool,enabled=True)
        if saved['terminal_kwargs']['input_hz'] is not saved['input_hz']:
            raise ValueError('archive lost original terminal input identity')
        report.update(completed=True,whole_new_splice_source=source,whole_final_input_property=final,
            all_new_shared_maps_and_predicates_restored=True,native_state_sha256=gate['native_state_sha256'],
            diagnostic_work=pool.used)
    except Exception as exc:report['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        report.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'restore_guard.json',report);print(json.dumps(report),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
