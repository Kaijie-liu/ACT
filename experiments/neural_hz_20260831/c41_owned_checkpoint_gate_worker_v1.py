"""New owned decode + actual functional HZ + observed complete checkpoint gates."""
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
from experiments.neural_hz_20260831.c34_portable_rhs_restore_v1 import restore
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import functional,checkpoint_payload
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load as owned_load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured,checkpoint_closure
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json

EXP=Path(__file__).resolve().parent
SOURCE=EXP/'results/c34_changed_terminal_20260911_v1'
DIRECTORY=EXP/'results/c41_owned_checkpoint_gate_20260911_v1'


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered functional target')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();whole=WorkPool(256_000_000);branch=BranchPool(whole)
    whole.charge('c41_prepaid_evidence_events',4096);event_count=0;provisional={}
    report=dict(provisional_in_memory_diagnostic=provisional,completed=False,formal_gain=0,new_HZ_constructed=False,solver_executed=False,
        native_admission_certificate=False,whole_C34_LIVE_gate_proved=False)
    with (DIRECTORY/'events.jsonl').open('x') as events:
        def emit(event):
            nonlocal event_count
            event_count+=1
            if event_count>64:raise ValueError('unpaid additional evidence event')
            record=dict(worker_elapsed_s=time.monotonic()-started,**event)
            events.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');events.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
                raise ValueError('complete census source/proof freeze drift')
            gate=json.loads((SOURCE/'terminal_gate.json').read_text())
            if not gate['terminal_gates_passed'] or _sha256(SOURCE/'native_state.pickle')!=gate['native_state_sha256']:
                raise ValueError('actual complete C34 state drift')
            with (SOURCE/'native_state.pickle').open('rb') as f:
                (saved,decode),decode_measurement=measured(lambda:owned_load(f,expected_sha256=gate['native_state_sha256'],pool=whole,enabled=True),
                    observe=lambda stats:report.update(decode_measurement=stats))
            report['owned_decode']=decode
            emit(dict(event='authenticated_owned_checkpoint_decode_passed',decoder=decode,charged_work=whole.used))
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
            def build():
                post,final,packed,runs,result=functional(state,saved['final_hz'],pool=whole,branch=branch,observe=emit,enabled=True)
                provisional.update(result)
                payload=checkpoint_payload(saved,state.hz,post,final,packed,runs,pool=whole)
                whole.charge('c41_owned_checkpoint_schema_binding',512)
                payload.update(schema='c41_owned_functional_half_gauge_checkpoint_v1',requires_owned_readonly_decode=True,owned_decode_provenance=decode)
                result['checkpoint_closure']=checkpoint_closure(saved,payload,pool=whole,observe=emit)
                provisional.update(result)
                result['diagnostic_work']=whole.used;result['work_parts']=dict(whole.parts)
                return result,payload
            (result,payload),measurement=measured(build,observe=lambda stats:report.update(measurement=stats))
            if not result['checkpoint_closure']['strict_checkpoint_numeric_decrease']:
                raise ValueError('complete checkpoint does not strictly decrease bytes and entries')
            # Nothing is published as a completed candidate before the whole
            # source/inverse/coverage and conservative measured gates finish.
            with (DIRECTORY/'functional_checkpoint.pickle').open('xb') as f:pickle.dump(payload,f,protocol=5)
            _atomic_exclusive_json(DIRECTORY/'transformation.json',result)
            report.update(completed=True,new_HZ_constructed=True,functional=result,measurement=measurement,
                checkpoint_numeric_gate_passed=result['checkpoint_closure']['strict_checkpoint_numeric_decrease'],
                functional_checkpoint_sha256=_sha256(DIRECTORY/'functional_checkpoint.pickle'),
                transformation_sha256=_sha256(DIRECTORY/'transformation.json'),native_state_sha256=gate['native_state_sha256'])
            emit(dict(event='complete_actual_functional_HZ_saved',new_final_sha256=result['new_final_sha256'],
                checkpoint_numeric_gate_passed=report['checkpoint_numeric_gate_passed'],formal_gain=0))
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc))
            emit(dict(event='owned_checkpoint_failed_no_complete_candidate_published',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                whole_diagnostic_work=whole.used,whole_work_parts=dict(whole.parts),
                branch_work=branch.used,branch_work_parts=dict(branch.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','branch_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
