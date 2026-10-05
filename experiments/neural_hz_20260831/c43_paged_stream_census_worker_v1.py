"""One complete hash-bound nonexecuting stream census; never unpickle."""
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c43_paged_stream_census_v1 import census
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;SOURCE=EXP/'results/c34_changed_terminal_20260911_v1'
DIRECTORY=EXP/'results/c43_paged_stream_census_20260911_v1'


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered stream census')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();pool=WorkPool(256_000_000);pool.charge('c43_prepaid_evidence_events',16384)
    report=dict(completed=False,formal_gain=0,unpickler_executed=False,new_HZ_constructed=False,solver_executed=False)
    count=0
    with (DIRECTORY/'events.jsonl').open('x') as f:
        def emit(value):
            nonlocal count
            count+=1
            if count>256:raise ValueError('unpaid additional stream event')
            f.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');f.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('complete source freeze drift')
            gate=json.loads((SOURCE/'terminal_gate.json').read_text())
            if not gate['terminal_gates_passed'] or _sha256(SOURCE/'native_state.pickle')!=gate['native_state_sha256']:
                raise ValueError('original C34 artifact drift')
            emit(dict(event='complete_original_file_identity_checked_no_unpickle',bytes=gate['native_state_bytes']))
            with (SOURCE/'native_state.pickle').open('rb') as source:
                result,stats=measured(lambda:census(source,expected_sha256=gate['native_state_sha256'],
                    expected_bytes=gate['native_state_bytes'],pool=pool,observe=emit,enabled=True),
                    observe=lambda value:report.update(measurement=value))
            _atomic_exclusive_json(DIRECTORY/'census.json',result)
            report.update(completed=True,census=result,census_sha256=_sha256(DIRECTORY/'census.json'))
            emit(dict(event='complete_nonexecuting_census_saved',opcodes=result['opcode_count'],formal_gain=0))
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='stream_census_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,work_parts=dict(pool.parts),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
