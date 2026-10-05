"""One bound read-only census with complete failure/exit artifact retention."""
import json
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c32_splice_binding_v1 import admit
from experiments.neural_hz_20260831.c34_terminal_binding_v1 import verify
from experiments.neural_hz_20260831.c34_portable_rhs_restore_v1 import restore
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c35_nonunit_exact_census_v1 import census
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json

EXP=Path(__file__).resolve().parent
SOURCE=EXP/'results/c34_changed_terminal_20260911_v1'
DIRECTORY=EXP/'results/c35_current_nonunit_census_20260911_v1'


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered census target')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();whole=WorkPool(256_000_000);branch=BranchPool(whole)
    report=dict(completed=False,formal_gain=0,new_HZ_constructed=False,solver_executed=False)
    with (DIRECTORY/'events.jsonl').open('x') as events:
        def emit(event):
            record=dict(worker_elapsed_s=time.monotonic()-started,**event)
            events.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');events.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
                raise ValueError('complete census source/proof freeze drift')
            gate=json.loads((SOURCE/'terminal_gate.json').read_text())
            if not gate['terminal_gates_passed'] or _sha256(SOURCE/'native_state.pickle')!=gate['native_state_sha256']:
                raise ValueError('actual complete C34 state drift')
            with (SOURCE/'native_state.pickle').open('rb') as f:saved=pickle.load(f)
            state,source=admit(enabled=True,**saved['spliced_state_fields'])
            report['source_binding']=source
            report['portable_RHS_restore']=restore(saved['final_hz'],state.hz,
                expected_final_sha256=gate['final_hz_sha256'],expected_source_sha256=source['complete_new_HZ_sha256'],
                pool=whole,enabled=True)
            runtime=dict(lifted=state,native_block_calls=1,layer=saved['selected_layer'],
                tf=SimpleNamespace(_net=saved['net'],_sparse_hz_cache=saved['hz_cache'],_sparse_affine_expr_cache=saved['expr_cache']))
            report['final_input_property_binding']=verify(runtime,saved['final_hz'],saved['input_hz'],saved['output_spec'],
                saved['terminal_kwargs'],saved['final_proof_bytes'],saved['final_proof_sha256'],pool=whole,enabled=True)
            emit(dict(event='whole_source_final_input_property_restored',charged_work=whole.used))
            (result,table),measurement=measured_build(lambda:census(state,saved['final_hz'],pool=whole,branch=branch,observe=emit,enabled=True))
            # Only a completely covered, measured, unchanged-source result can be saved.
            with (DIRECTORY/'complete_MAIN_table.npz').open('xb') as f:np.savez(f,table=table)
            _atomic_exclusive_json(DIRECTORY/'census.json',result)
            report.update(completed=True,census=result,measurement=measurement,
                table_sha256=_sha256(DIRECTORY/'complete_MAIN_table.npz'),
                census_sha256=_sha256(DIRECTORY/'census.json'),native_state_sha256=gate['native_state_sha256'])
            emit(dict(event='complete_census_saved',all_MAIN_classified=result['all_MAIN_classified'],
                independent_pairs=result['simultaneous_independent_pairs'],formal_gain=0))
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc))
            emit(dict(event='census_failed_no_subset_published',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                whole_diagnostic_work=whole.used,whole_work_parts=dict(whole.parts),
                branch_work=branch.used,branch_work_parts=dict(branch.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','branch_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
