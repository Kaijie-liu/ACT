"""Complete ordinary-row payment; fresh outputs held and compared in full."""
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831 import c97_prepared_row_v1 as old
from experiments.neural_hz_20260831 import c104_prepared_row_v1 as new
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c104_array_api_20260913_v1'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    frozen=json.loads((RUN/'preregistered.json').read_text())
    if any(_sha256(EXP/n)!=sha for n,sha in frozen['source_sha256'].items()):raise ValueError('frozen source drift')
    started=time.monotonic();pool=WorkPool(256_000_000)
    result=dict(completed=False,formal_gain=0,scope='ordinary_row_preparation_not_full_network_or_F_speed',shapes=[])
    try:
        for nc,nb in ((3,0),(33,0),(289,0),(33,8)):
            cv,bv=[((np.arange(n)%17)+1).astype(np.float64)/64 for n in (nc,nb)]
            cv[::2]*=-1;bv[::2]*=-1
            cp,bp=[((np.arange(n)%9)-4).astype(np.int64) for n in (nc,nb)]
            arms={};outputs={};n=nc+nb
            for name,module in (('C97',old),('C104',new)):
                observed=[];start=pool.used
                def build():
                    rows=[]
                    for _ in range(2000):
                        pool.charge('complete_original_numerical_program',32*n+64)
                        rows.append(module._prepare(cv,cp,bv,bp,.125,pool=pool))
                    return rows
                outputs[name],stats=measured(build,observe=observed.append)
                arms[name]=dict(measurement=stats,work=pool.used-start,rows=len(outputs[name]))
            for a,b in zip(outputs['C97'],outputs['C104']):
                pool.charge('complete_output_equality',8*n+64)
                if (a is None or b is None or (a.shift,a.head)!=(b.shift,b.head)
                        or np.float64(a.rhs).tobytes()!=np.float64(b.rhs).tobytes()
                        or a.continuous.tobytes()!=b.continuous.tobytes()
                        or a.binary.tobytes()!=b.binary.tobytes()):raise ValueError('complete row output differs')
            ratio=arms['C97']['measurement']['elapsed_s']/arms['C104']['measurement']['elapsed_s']
            report=dict(continuous=nc,binary=nb,arms=arms,full_rows_equal=True,speed_ratio=ratio)
            result['shapes'].append(report)
            _atomic_exclusive_json(RUN/f'row_shape_{nc}_{nb}.json',report)
            del outputs
        old_time=sum(s['arms']['C97']['measurement']['elapsed_s'] for s in result['shapes'])
        new_time=sum(s['arms']['C104']['measurement']['elapsed_s'] for s in result['shapes'])
        result.update(diagnostic_speed_ratio=old_time/new_time,old_traced_s=old_time,new_traced_s=new_time)
        if result['diagnostic_speed_ratio']<1.25 or any(s['speed_ratio']<1 for s in result['shapes']):
            raise ValueError('preregistered whole ordinary-shape payment gate failed')
        result['completed']=True
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,work=pool.used,work_parts=dict(pool.parts),
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in frozen['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'payment_result.json',result);print(json.dumps(result),flush=True)
    if not result['completed'] or result['source_drift']:raise SystemExit(1)


if __name__=='__main__':main()
