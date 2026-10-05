"""One ordinary full-source cProfile comparison, no mathematical changes."""
import cProfile
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c104_birth_emission_v1 import lift
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression
from experiments.neural_hz_20260831.c91_physical_archive_v1 import fingerprint
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c105_source_profile_20260913_v1'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    frozen=json.loads((RUN/'preregistered.json').read_text())
    if any(_sha256(EXP/n)!=sha for n,sha in frozen['source_sha256'].items()):raise ValueError('source drift')
    started=time.monotonic();result=dict(completed=False,formal_gain=0,scope='ordinary_complete_source_diagnostic')
    upper=WorkPool(256_000_000);binding=WorkPool(64_000_000)
    upper.charge('two_complete_source_caps',128_000_000)
    upper.charge('complete_profile_observation_reserve',64_000_000)
    upper.charge('complete_state_binding_reserve',64_000_000)
    outputs={};measurements={};observations=[]
    try:
        expr=expression(c=16,k=32,h=12);keep=np.ones(expr.n_out,bool);before=expression_binding(expr)
        profile=cProfile.Profile()
        for name in ('plain','profiled'):
            def build():
                if name=='profiled':profile.enable()
                try:return lift(expr,keep,enabled=True,max_work=64_000_000,max_branch_work=64_000_000)
                finally:
                    if name=='profiled':profile.disable()
            outputs[name],measurements[name]=measured(build,observe=observations.append)
        entries=profile.getstats();facts=[]
        def label(code):
            if isinstance(code,str):return code
            return f'{code.co_filename}:{code.co_firstlineno}:{code.co_name}'
        calls=sum(e.callcount for e in entries);observation_work=8*calls+128*len(entries)
        if observation_work>64_000_000:raise MemoryError('complete profiler reservation exhausted')
        for e in entries:
            facts.append(dict(function=label(e.code),calls=e.callcount,recursive_calls=e.reccallcount,
                inclusive_s=e.totaltime,self_s=e.inlinetime,
                callees=[dict(function=label(c.code),calls=c.callcount,inclusive_s=c.totaltime,self_s=c.inlinetime)
                    for c in (e.calls or [])]))
        facts.sort(key=lambda e:e['self_s'],reverse=True)
        def compare():
            full=numeric_layout(dict(original=expr,keep=keep,complete_outputs=outputs,profile_facts=facts),binding)
            if full.resident_entries>64_000_000:raise MemoryError('complete diagnostic numeric population exceeds64M')
            binding.charge('complete_both_source_fingerprints',4*int(full.resident_entries)+4096)
            a,b=outputs['plain']['state'],outputs['profiled']['state']
            if fingerprint(a)!=fingerprint(b) or expression_binding(expr)!=before:
                raise ValueError('complete source/report/owner/inverse state changed under observation')
            return dict(resident_bytes=full.resident_bytes,resident_entries=full.resident_entries,
                full_state_identity=fingerprint(a),original_expression_unchanged=True,
                all_original_source_maps_owners_inverses_equal=True)
        comparison,comparison_stats=measured(compare,observe=observations.append)
        source_reports={name:{k:d['state']['fields']['report'][k] for k in
            ('total_work_upper','largest_branch_work_upper','node_count','prepared_encoding','circuit_generation')}
            for name,d in outputs.items()}
        result.update(completed=True,measurements=measurements,comparison=comparison,
            comparison_measurement=comparison_stats,source_reports=source_reports,
            profile_calls=calls,profile_function_entries=len(entries),profile_work=observation_work,
            binding_work=binding.used,profile_facts=facts)
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,observations=observations,
            aggregate_reserved_work=upper.used,reservations=upper.parts,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in frozen['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('profile_facts','source_reports')}),flush=True)
    if not result['completed'] or result['source_drift']:raise SystemExit(1)


if __name__=='__main__':main()
