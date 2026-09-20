"""Read-only post-run evidence and arithmetic comparison, no fresh solving."""
from pathlib import Path
import time
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest
from modular_diagnostic import contract as C
from modular_diagnostic.run import audit_saved


def partial(root,name):
    path=root/name
    if not path.exists():return {'state':'MISSING','value':None,'sha256':None}
    raw=path.read_bytes()
    try:value=read(path)
    except (ValueError,UnicodeError):return {'state':'PARTIAL_UNPARSEABLE','value':None,'sha256':digest(raw)}
    return {'state':'RECORDED','value':value,'sha256':digest(raw)}


def comparison(mapping,construction,old_mapping,old_construction):
    comparable=(mapping is not None and old_mapping is not None and
                mapping.get('status')==old_mapping.get('status')=='MAPPED_HINT_ONLY')
    basis=None if not comparable else all(mapping['hint'][k]==old_mapping['hint'][k]
                                       for k in ('coordinates','rows','basic_columns','anchors'))
    system=None
    if construction is not None and old_construction is not None:
        new,old=construction.get('assembled_system_sha256'),old_construction.get('assembled_system_sha256')
        if new is not None and old is not None:system=new==old
    return {'basis_structure_equal':basis,'assembled_system_equal':system,
            'interpretation':'descriptive same-system comparison' if basis and system else
                             'basis/system not matched or unobserved; no pure arithmetic attribution',
            'matched_timing_experiment':False}


def detail(root,previous=None):
    saved={n:partial(root,n) for n in ('mapping.json','construction.json','native/raw_native.json')}
    mapping=saved['mapping.json']['value'];construction=saved['construction.json']['value']
    raw=saved['native/raw_native.json']['value']
    output={'record_states':{n:{k:v[k] for k in ('state','sha256')} for n,v in saved.items()},
        'construction':None if construction is None else {k:construction.get(k) for k in
            ('status','error','seconds','operations','stats','arithmetic','attempts','solver_calls','assembled_system_sha256')},
        'native_objective_untrusted':None if raw is None else raw.get('native_objective'),
        'comparison':None}
    if previous is not None:
        output['comparison']=comparison(mapping,construction,partial(previous,'mapping.json')['value'],
                                        partial(previous,'construction.json')['value'])
    return output


def collect():
    begin=time.monotonic();v=C.verify();root=C.OUTPUT
    launch=read(root/'launch.json')
    if launch['freeze']!=C.ref(C.FREEZE) or launch['review']!=C.ref(C.REVIEW):raise ValueError('launch identity')
    summary=audit_saved(v['jobs'],root)
    from primitive_diagnostic.contract import OUTPUT as previous
    rows=[{**row,'arithmetic_detail':detail(root/job['job_id'],previous/job['job_id'])}
          for job,row in zip(v['jobs'],summary['rows'])]
    return {'schema':'MODULAR_DIAGNOSTIC_ARCHIVE_V1','status':'PASS','issues':[],
        'execution_status':summary['status'],'execution_head':launch['head'],'freeze':C.ref(C.FREEZE),
        'sources':C.sources(),'rows':rows,'aggregates':summary['aggregates'],'cost_totals':summary['cost_totals'],
        'final_summary_publication':read(root/'summary_publication.json'),
        'artifact_sha256':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()},
        'seconds':time.monotonic()-begin,'new_solve_or_reconstruction_calls':0,
        'scope':'structural ledger/identity review, not a new feasibility check or network proof'}


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    destination=a.output.resolve()
    if not destination.is_relative_to(Path('/data1/Kane/MOE')):raise ValueError('outside workspace')
    if destination.exists():raise FileExistsError('preserve prior review')
    result=collect();save_new(destination,result)
    print(json.dumps({k:result[k] for k in ('status','execution_status','aggregates')},indent=2))
