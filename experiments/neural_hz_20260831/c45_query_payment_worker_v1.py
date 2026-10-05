"""Complete synthetic query stream comparison with owned transient accounting."""
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c45_streamed_uid_queries_v1 import _Queries
from experiments.neural_hz_20260831.test_c45_streamed_uid_queries_v1 import lineage
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c45_query_payment_20260911_v1'


def run(rows,compact,pool):
    count=len(range(128,rows,256));pool.charge('c45_complete_synthetic_source_construction',512+256*count)
    source=lineage(list(range(128,rows,256)),[(101*j,j) for j in range(count)])
    seal=source.seal;index=_Queries(source,rows,pool) if compact else source
    pool.charge('c45_complete_query_value_image',8*rows+256);image=hashlib.sha256()
    for row in range(rows):
        rank=index.eq_row(row,pool=pool);uid=index.retired_to(row,pool=pool)
        image.update(struct.pack('<ii',-1 if rank is None else rank,-1 if uid is None else uid))
    if compact:proof=index.finish()
    else:
        # Include comparator's post-source validation too, not just new seals.
        pool.charge('c45_old_source_final_seal',8*sum(a.size for a in
            (source.eq_roots,source.eq_scales,source.columns,source.retired,source.tails))+256)
        source.validate();proof=dict(complete_ordered_old_queries=True)
    if source.seal!=seal:raise ValueError('source changed during query diagnostic')
    return dict(query_sha256=image.hexdigest(),rows=rows,deletions=count,proof=proof)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered query diagnostic')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);pool.charge('c45_prepaid_evidence_events',16384);started=time.monotonic()
    report=dict(completed=False,formal_gain=0,actual_source_loaded=False,new_HZ_constructed=False,
        solver_executed=False,runtime_admitted=False,cases=[])
    with (DIRECTORY/'events.jsonl').open('x') as f:
        def emit(value):
            f.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');f.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
            for rows in (65536,131072):
                case=dict(rows=rows);report['cases'].append(case)
                for compact in (False,True):
                    name='indexed' if compact else 'baseline';begin=pool.used
                    value,stats=measured(lambda:run(rows,compact,pool),observe=lambda s:emit(dict(
                        event='complete_measured_query_arm',rows=rows,arm=name,measurement=s)))
                    case[name]=dict(**value,work=pool.used-begin,measurement=stats)
                case['complete_query_image_equal']=case['baseline']['query_sha256']==case['indexed']['query_sha256']
                if not case['complete_query_image_equal']:raise ValueError('complete exact query image differs')
                case['charged_work_saved']=case['baseline']['work']-case['indexed']['work']
                emit(dict(event='complete_exact_query_case',rows=rows,charged_work_saved=case['charged_work_saved']))
            report.update(completed=True,synthetic_component_payment_passed=all(c['charged_work_saved']>0 for c in report['cases']),
                full_source_or_native_payment_proved=False,full_HZ_LIVE_gate_proved=False)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='query_diagnostic_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,work_parts=dict(pool.parts),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
