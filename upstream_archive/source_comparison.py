"""Derived identity clarification; compare saved sources only, without solving."""
import copy
from fractions import Fraction
from pathlib import Path
from portable_proof.runtime import digest
from single_check_portable.execution import ROOT,read,save_new
from upstream_portable.study import verify,OUTPUT


def review():
    verify();rows=[]
    for rank in range(4):
        a=OUTPUT/f'rank{rank}_single_check';b=OUTPUT/f'rank{rank}_double_check'
        x=read(a/'source/manifest.json');y=read(b/'source/manifest.json')
        xr=copy.deepcopy(x['routes']);yr=copy.deepcopy(y['routes'])
        # Disclose exactly what is omitted for this semantic comparison.
        for r in (xr,yr):
            for branch in r['branches']:branch.pop('elapsed',None)
        pairs=[]
        if x['contexts'].keys()!=y['contexts'].keys():raise ValueError('context inventory differs')
        for key in x['contexts']:
            entry={'pair':x['contexts'][key]['pair']}
            for kind in ('joint_source','router_source'):
                refs=[m['contexts'][key][kind] for m in (x,y)];values=[]
                for base,ref in zip((a,b),refs):
                    p=base/'source'/ref['file']
                    if digest(p.read_bytes())!=ref['sha256']:raise ValueError('stored source hash mismatch')
                    values.append(read(p))
                entry[kind]={'single_sha256':refs[0]['sha256'],'double_sha256':refs[1]['sha256'],
                             'differing_top_level_keys':[k for k in values[0] if values[0][k]!=values[1][k]]}
            pairs.append(entry)
        row={'rank':rank,'request_equal':x['request']==y['request'],
             'route_records_equal_after_omitting_only_branch_elapsed':xr==yr,
             'common_facts_reference_equal':x['common_facts']==y['common_facts'],'pairs':pairs}
        if rank==2:
            one=read(a/'tail/check.log')['result'];two=read(b/'tail/check.log')['result']
            row['full_checked_result_equal']=one==two
            row['both_statuses']=[one['status'],two['status']]
            row['lower_bound_differences_single_minus_double']=[float(Fraction(o['lower_bound'])-Fraction(t['lower_bound']))
                                    for o,t in zip(one['obligations'],two['obligations'])]
        rows.append(row)
    return {'status':'PASS','issues':[],'rows':rows,
        'interpretation':'same request, routing and common facts, but independently generated downstream HZ coefficients can differ; not a fixed-identical-proof timing benchmark',
        'reason_for_coefficient_differences':'not isolated by this read-only comparison',
        'parent_archive_sha256':digest((ROOT/'docs/upstream_portable_v1_execution_results.json').read_bytes()),
        'new_model_or_solver_calls':0}


if __name__=='__main__':
    import json
    dest=ROOT/'docs/upstream_portable_v1_source_comparison.json'
    if dest.exists():raise FileExistsError('no overwrite')
    value=review();save_new(dest,value);print(json.dumps(value,indent=2))
