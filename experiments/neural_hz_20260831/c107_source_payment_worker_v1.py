# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete ordinary original/new source comparison under one prepaid bound."""
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c106_birth_emission_v1 import lift as old
from experiments.neural_hz_20260831.c107_birth_emission_v1 import lift as new
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression
from experiments.neural_hz_20260831.c91_physical_archive_v1 import fingerprint
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c107_canonical_encoding_20260913_v1'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    frozen=json.loads((RUN/'preregistered.json').read_text());started=time.monotonic()
    if any(_sha256(EXP/n)!=sha for n,sha in frozen['source_sha256'].items()):raise ValueError('source freeze drift')
    result=dict(completed=False,formal_gain=0,scope='complete_ordinary_source_not_actual_network_or_F_speed')
    upper=WorkPool(256_000_000);binding=WorkPool(64_000_000)
    upper.charge('two_complete_source_caps',128_000_000)
    upper.charge('complete_state_binding_reserve',64_000_000)
    upper.charge('complete_measurement_observation_reserve',64_000_000)
    outputs={};measurements={};observations=[]
    try:
        expr=expression(c=16,k=32,h=8);keep=np.ones(expr.n_out,bool);before=expression_binding(expr)
        for name,fn in (('C106',old),('C107',new)):
            outputs[name],measurements[name]=measured(lambda:fn(expr,keep,enabled=True,
                max_work=64_000_000,max_branch_work=64_000_000),observe=observations.append)
        def compare():
            full=numeric_layout(dict(original=expr,keep=keep,complete_outputs=outputs),binding)
            if full.resident_entries>64_000_000:raise MemoryError('complete retained diagnostic exceeds64M')
            binding.charge('complete_source_identity_comparison',4*int(full.resident_entries)+4096)
            a,b=outputs['C106']['state'],outputs['C107']['state']
            identity=fingerprint(a)
            if identity!=fingerprint(b) or expression_binding(expr)!=before:
                raise ValueError('full original source/report/owner/inverse equality failed')
            for name in ('eq_uids','ineq_uids'):
                if not np.array_equal(outputs['C106']['construction'][name],outputs['C107']['construction'][name]):
                    raise ValueError('complete actual construction UID equality failed')
            return dict(resident_bytes=full.resident_bytes,resident_entries=full.resident_entries,
                full_state_identity=identity,original_expression_unchanged=True,
                all_original_maps_owners_inverses_and_report_equal=True)
        compared,comparison_stats=measured(compare,observe=observations.append)
        ratio=measurements['C106']['elapsed_s']/measurements['C107']['elapsed_s']
        result.update(measurements=measurements,comparison=compared,comparison_measurement=comparison_stats,
            diagnostic_speed_ratio=ratio,old_traced_s=measurements['C106']['elapsed_s'],
            new_traced_s=measurements['C107']['elapsed_s'])
        if ratio<1.:raise ValueError('complete ordinary source regressed; no original network launch')
        result['completed']=True
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,aggregate_reserved_work=upper.used,
            reservations=upper.parts,binding_work=binding.used,observations=observations,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in frozen['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'payment_result.json',result);print(json.dumps(result),flush=True)
    if not result['completed'] or result['source_drift']:raise SystemExit(1)


if __name__=='__main__':main()

