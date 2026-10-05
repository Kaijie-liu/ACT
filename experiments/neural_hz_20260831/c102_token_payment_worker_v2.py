# SPDX-License-Identifier: AGPL-3.0-or-later
"""One bounded complete ordinary-ID-shape timing gate, no archived/network HZ."""
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect as original
from experiments.neural_hz_20260831.c78_complete_roots_v1 import collect as prior
from experiments.neural_hz_20260831.c102_complete_roots_v1 import collect as changed
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c102_word_runtime_20260913_v2'


def main():
    """Retain both arms and original-oracle equality; failed timing closes the gate."""
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    frozen=json.loads((RUN/'preregistered.json').read_text())
    if any(_sha256(EXP/n)!=sha for n,sha in frozen['source_sha256'].items()):
        raise ValueError('frozen source drift before payment diagnostic')
    started=time.monotonic();result=dict(completed=False,formal_gain=0,
        scope='synthetic_complete_ordinary_integer_shape_not_original_network_or_score',
        integer_visits=6744304,distinct_integer_objects=1314088,full_numeric_hash_traffic_in_token_pool=False)
    try:
        values=list(range(257,257+1314088))
        full,tail=divmod(6744304,len(values))
        tf=SimpleNamespace(original_neural_variables={
            'shared_global_latent_lists':[list(values) for _ in range(full)]+[values[:tail]]})
        # The exact same objects remain held for both arms; this diagnostic is
        # not an original-runtime transient qualification or F-speed claim.
        expected=original(tf);pool=WorkPool(256_000_000)
        for label,collector in (('old_C78_traced',prior),('new_C102_traced',changed)):
            events=[];measurements=[];start=pool.used
            actual,stats=measured(lambda:collector(tf,pool=pool,enabled=True,observe=events.append),
                observe=measurements.append)
            report=dict(measurement=stats,token_work=pool.used-start,shared_token_work=pool.used,
                last_progress=events[-1],full_fingerprint_equal=actual.fingerprint==expected.fingerprint,
                schemas_equal=actual.schema_counts==expected.schema_counts,
                shallow_bytes_equal=actual.python_shallow_bytes==expected.python_shallow_bytes,
                unique_objects_equal=actual.unique_objects==expected.unique_objects,
                numeric_owners_equal=actual.numeric.keys()==expected.numeric.keys() and
                    all(actual.numeric[k] is v for k,v in expected.numeric.items()),
                numeric_measure_equal=asdict(actual.measure())==asdict(expected.measure()))
            result[label]=report
            _atomic_exclusive_json(RUN/(label+'.json'),report)
            if not all(report[k] for k in ('full_fingerprint_equal','schemas_equal','shallow_bytes_equal',
                    'unique_objects_equal','numeric_owners_equal','numeric_measure_equal')):
                raise ValueError('complete original C5 token/identity/owner oracle differs')
            if events[-1]['integer_visits']!=6744304 or events[-1]['integer_unique']!=1314088:
                raise ValueError('preregistered complete ordinary population differs')
        result['diagnostic_speed_ratio']=result['old_C78_traced']['measurement']['elapsed_s']/result['new_C102_traced']['measurement']['elapsed_s']
        result['timing_gate_passed']=result['diagnostic_speed_ratio']>=1.5
        if not result['timing_gate_passed']:raise ValueError('bounded payment does not demonstrate preregistered improvement')
        result['completed']=True
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,source_drift=any(
            _sha256(EXP/n)!=sha for n,sha in frozen['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'payment_result.json',result)
        print(json.dumps(result),flush=True)
    if not result['completed'] or result['source_drift']:raise SystemExit(1)


if __name__=='__main__':main()
