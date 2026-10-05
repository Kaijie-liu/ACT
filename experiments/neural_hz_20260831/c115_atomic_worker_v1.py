# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded complete factored-key batch, all results automatically retained."""
import faulthandler
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c109_shared_partial_v1 import fixture,MODES,GRIDS
from experiments.neural_hz_20260831.c115_atomic_fixture_v1 import complete
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c115_atomic_word_plan_20260913_v1'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((RUN/'preregistered.json').read_text())
    pool=WorkPool(256_000_000);started=time.monotonic()
    record=dict(completed=False,formal_gain=0,cases=[],source_runtime_LIVE_admitted=False)
    fatal=(RUN/'fatal.log').open('x');faulthandler.enable(file=fatal,all_threads=True)
    try:
        if any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()):
            raise ValueError('complete frozen dependency drift')
        cases=[(c,k,mode,(2,2)) for c,k in ((1,2),(3,4)) for mode in MODES]
        cases += [(1,2,'dense',grid) for grid in GRIDS]
        for index,(c,k,mode,grid) in enumerate(cases):
            state={}
            def build():
                report,before,after,old,new,expected=complete(c,k,mode,grid,pool=pool)
                state.update(old=old,new=new,expected=expected,before=before,after=after)
                report.update(channels=c,outputs=k,mode=mode,grid=list(grid))
                return report
            report,measurement=measured(build,observe=lambda _:None)
            report['measurement']=measurement
            arrays={f'before_{key}':a for key,a in state['before'].items()}
            arrays.update({f'after_{key}':a for key,a in state['after'].items()})
            unique={id(a):a for a in arrays.values()}
            report['complete_comparison_numeric_bytes']=sum(a.nbytes for a in unique.values())
            report['complete_comparison_numeric_entries']=sum(a.size for a in unique.values())
            if report['complete_comparison_numeric_entries']>64_000_000:
                raise MemoryError('whole comparison numeric entries cap exceeded')
            path=RUN/f'case_{index:02d}.npz'
            with path.open('xb') as stream:np.savez(stream,**arrays)
            report.update(artifact=path.name,artifact_sha256=_sha256(path))
            _atomic_exclusive_json(RUN/f'case_{index:02d}.json',report)
            record['cases'].append(report)
            print(json.dumps(dict(event='c115_case',index=index,mode=mode,channels=c,grid=grid,
                removed=report['omitted_factors'],nnz_delta=report['nnz_delta'],
                byte_delta=report['byte_delta'],entry_delta=report['entry_delta'],
                numeric_win=report['strict_complete_numeric_win'],removed_M=report['removed_m'],work=pool.used)),flush=True)
            del arrays,unique,state
        record.update(completed=True,work=pool.used,work_parts=pool.parts,
            numeric_winning_cases=sum(r['strict_complete_numeric_win'] for r in record['cases']),
            hash_traffic_and_all_symbolic_CPU_not_generation_tokens=True)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        faulthandler.disable();fatal.close()
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='cases'}),flush=True)
    if not record['completed'] or record['source_drift']:raise SystemExit(1)


if __name__=='__main__':main()
