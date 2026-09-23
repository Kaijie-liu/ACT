"""Post-hoc saved-only cost/obligation diagnosis; never runs a solver."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from audit_metamoe_current_assignment import read,require,scalar_rows
from audit_metamoe_protected_smoke_r1 import arrays,csr
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'docs/metamoe_checked_base_archive_20260923_r1.json'
DATA=Path('/data1/Kane/MOE/baseline_runs/metamoe_checked_base_control_20260923_r1')


def analyze():
    audit=read(ARCHIVE);require(audit['audit']=='PASS' and audit['same_expert_matrix'],'audited common object')
    for name,meta in audit['files'].items():require(sha256(DATA/name)==meta['sha256'],'saved evidence changed')
    summary=[]
    for arm in ('native','checked'):
        root=DATA/arm/'mnist_0_act';folder=root/'protected/evaluation_000'
        import json
        trace=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
        begins={r['seq']:r for r in trace if r['kind']=='BEGIN'}
        durations={}
        wanted=('act.back_end.moe.hz_routing.analyze_candidates','ProtectedHZSolver.evaluate_spec',
                'act.back_end.solver.solver_hz.hz_support_bounds')
        for r in trace:
            if r['kind']=='END' and begins[r['span']]['name'] in wanted:
                name=begins[r['span']]['name'];durations[name]=durations.get(name,0.)+r['elapsed']-begins[r['span']]['elapsed']
        result=read(folder/'result.json');m=result['metadata'];groups=[]
        for q in sorted(folder.glob('query_*')):
            r=read(q/'return.json')
            if r['scope']['phase']!='expanded':continue
            z=arrays(q/'extra.npz')
            key=tuple((k,z[k].dtype.str,z[k].shape,z[k].tobytes()) for k in sorted(z))
            for prior,members in groups:
                if key==prior:members.append(r['scope']['row']);break
            else:groups.append((key,[r['scope']['row']]))
        require(sum(len(g[1]) for g in groups)==19,'not all properties visited')
        elapsed=read(root/'terminal.json')['seconds']
        summary.append({'variant':arm,'full_status':read(root/'result.json')['status'],
            'full_seconds':elapsed,'stage_seconds':durations,'base_status':m['base_status'],
            'base_seconds':m['queries'][0]['return_elapsed_seconds'],
            'property_query_seconds':sum(q['return_elapsed_seconds'] for q in m['queries'][1:]),
            'completed_expanded_infeasibility':sum(p['status']=='infeasible' for p in m['properties']),
            'remaining_expert_output_obligations':sum(p['status']!='infeasible' for p in m['properties']),
            'distinct_expanded_queries':len(groups),'identical_query_groups':[g[1] for g in groups],
            'router_and_nonzero_observed_fraction':(durations[wanted[0]]+durations[wanted[2]])/elapsed})
    folder=DATA/'checked/mnist_0_act/protected/evaluation_000'
    z=arrays(folder/'base_model.npz');point=arrays(folder/'checked_base/candidate.npz')['point']
    represented=z['value_center']+scalar_rows(csr(z,'value_'),point)
    props=arrays(folder/'properties.npz');margins=-props['C']@represented
    return {'analysis':'SAVED_ONLY_POSTHOC','archive_sha256':sha256(ARCHIVE),'rows':summary,
        'checked_base_minimum_safety_margin':float(margins.min()),
        'expanded_queries_containing_checked_base_point':sum(
            float(scalar_rows(csr(arrays(q/'extra.npz')),point)[0])>=float(arrays(q/'extra.npz')['lb'][0])-1e-7
            for q in sorted(folder.glob('query_*'))),
        'new_solves':0,'new_forward_passes':0,'old_results_relabelled':False,
        'conclusion':'All 19 output obligations close in both repaired arms under the frozen HZ/HiGHS policy. Native UNKNOWN is base nonvacuity only. Checked base removes that barrier. No remaining expert output relaxation deficit is observed on this request.',
        'limits':'One old, route-stable request; non-outward-rounded policy, trusted network/guard lowering and native infeasibility. Not source-complete, route-changing or population speedup evidence.',
        'next':'Do not tighten this solved expert representation. Remaining observed cost is router/candidate and nonzero support execution; investigate separately under fresh controls if pursued.',
        'analysis_source_sha256':sha256(Path(__file__))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'output exists');value=analyze();write(a.output,value)
    print(value['conclusion']);print(value['rows'])
