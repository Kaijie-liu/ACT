"""One fixed complete synthetic consumer diagnostic, never an actual target."""
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c44_var_ids_v1 import paid
from experiments.neural_hz_20260831.c44_id_consumers_v1 import fixture,evidence,diagnostic_python_closure
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c44_id_consumers_20260911_v1'


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered synthetic consumer diagnostic')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();pool=WorkPool(256_000_000);pool.charge('c44_prepaid_evidence_events',16384)
    report=dict(completed=False,formal_gain=0,actual_source_loaded=False,new_HZ_constructed=False,
        solver_executed=False,runtime_admitted=False,cases=[])
    count=0
    with (DIRECTORY/'events.jsonl').open('x') as f:
        def emit(value):
            nonlocal count
            count+=1
            if count>128:raise ValueError('unpaid additional diagnostic event')
            f.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');f.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('complete source freeze drift')
            with paid(pool):
                for width in (4096,65536):
                    for consumers in (1,2,4):
                        case=dict(width=width,consumers=consumers);report['cases'].append(case)
                        for compact in (False,True):
                            arm='compact' if compact else 'baseline'
                            def build():
                                root=fixture(width,consumers,compact=compact,pool=pool)
                                return dict(evidence=evidence(root,pool),closure=diagnostic_python_closure(root,pool))
                            value,stats=measured(build,observe=lambda x:emit(dict(event='measured_arm',
                                width=width,consumers=consumers,arm=arm,measurement=x)))
                            case[arm]=dict(**value,measurement=stats)
                        a=case['baseline']['evidence'];b=case['compact']['evidence']
                        case['ordered_values_equal']=a['ordered_sequence_sha256']==b['ordered_sequence_sha256']
                        case['outer_aliases_equal']=a['outer_alias_matrix']==b['outer_alias_matrix']
                        if not case['ordered_values_equal'] or not case['outer_aliases_equal']:
                            raise ValueError('registered closed-consumer equivalence failed')
                        old=case['baseline']['closure']['unique_python_shallow_bytes']
                        new=case['compact']['closure']['unique_python_shallow_bytes']
                        case.update(python_byte_delta=new-old,strict_synthetic_reduction=new<old)
                        emit(dict(event='complete_case',width=width,consumers=consumers,
                            baseline_python_bytes=old,compact_python_bytes=new,whole_work=pool.used))
            report.update(completed=True,all_value_and_outer_alias_checks_passed=True,
                all_retaining_consumer_payment_guards_passed=all(c['strict_synthetic_reduction'] for c in report['cases']),
                full_HZ_LIVE_gate_proved=False,actual_native_payment_proved=False)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='diagnostic_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,work_parts=dict(pool.parts),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
