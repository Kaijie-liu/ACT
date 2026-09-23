"""Saved-only protected query/checked-base audit. No native optimization."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.current_assignment import model_fingerprint
from audit_metamoe_protected_smoke_r1 import evaluation as native_evaluation, arrays, csr
from audit_metamoe_current_assignment import read,require,check_point,inventory
from audit_metamoe_csr_paired_r4 import check_receipt,check_candidate
from audit_conv_f0_timing import check_trace
from metamoe_csr_paired_r4 import collect_terminal
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
import metamoe_checked_base_control as control


def evaluate(folder, required, request_hash, input_hash, partial=False):
    # Fallback/native executions retain the entire original audit, unchanged.
    if not (folder/'result.json').exists():return native_evaluation(folder,required,partial=partial)
    result=read(folder/'result.json');meta=result['metadata'];queries=meta['queries']
    if not queries or queries[0].get('evidence_kind')!='CURRENT_FULL_MATRIX_POINT':
        value=native_evaluation(folder,required,partial=partial)
        if (folder/'checked_base/return.json').exists():value['checked_attempt']=read(folder/'checked_base/return.json')
        return value
    plan=read(folder/'plan.json');begin,end=plan['started_monotonic'],plan['deadline_monotonic']
    require(plan['schema']=='protected-hz-v1' and 0<plan['total_seconds']<=30 and end==begin+plan['total_seconds']
            and plan['base_fraction']==.1 and plan['base_deadline_monotonic']==begin+.1*plan['total_seconds']
            and plan['tolerance']==1e-7 and plan['numerical_policy']==sh.hz_numerical_policy_manifest(),'expert contract')
    props=meta['properties'];M=meta['required_properties'];status=result['status']
    if (folder/'late_result_rejected.json').exists():status='unknown'
    require(M==required and [p['row'] for p in props]==list(range(M)),'all properties retained')
    require(status=='unknown' or result['finished_monotonic']<end,'late conclusion')
    z=arrays(folder/'base_model.npz');ints=z['integrality'];nc=int(np.count_nonzero(ints==0));nb=int(np.count_nonzero(ints==1))
    require(np.array_equal(ints,np.r_[np.zeros(nc),np.ones(nb)]),'factor domains')
    model=sh._HZMILP(z['value_center'],csr(z,'value_'),csr(z),z['row_lb'],z['row_ub'],z['var_lb'],z['var_ub'],ints,nc,nb)
    p=arrays(folder/'properties.npz');pid=read(folder/'property_identity.json')
    coeff=(sp.csr_matrix(p['C'])@model.value_matrix).tocsr();const=p['C']@model.value_center;t=p['thresholds']
    require(coeff.shape[0]==M and np.array_equal(const,p['constants']) and pid['count']==M and
            pid['properties_sha256']==sha256(folder/'properties.npz') and meta['matrix_sha256']==sha256(folder/'base_model.npz'),
            'matrix/property bindings')
    base=queries[0];saved=read(folder/'checked_base/return.json')
    require({k:v for k,v in base.items() if k!='return_elapsed_seconds'}==
            {k:v for k,v in saved.items() if k!='return_elapsed_seconds'},'base ledger')
    require(not (folder/'checked_base/late_return_rejected.json').exists() and base['returned_status']=='feasible'
            and meta['base_status']=='feasible' and base['check']['accepted'] and not base['native_started'] and
            base['native_result'] is None and base['model_sha256']==meta['matrix_sha256'] and
            base['model_fingerprint']==model_fingerprint(model),'base evidence identity')
    require(base['assignment_scope']==base['proposal_scope'] and base['assignment_scope']['request_sha256']==request_hash
            and base['assignment_scope']['input_sha256']==input_hash and base['assignment_scope']['evaluation_nonce'], 'current request scope')
    require(begin<=base['started_monotonic']<base['finished_monotonic']<base['deadline_monotonic']==plan['base_deadline_monotonic']
            and base['started_monotonic']+base['return_elapsed_seconds']<base['deadline_monotonic'],'base budget')
    require(base['candidate_sha256']==sha256(folder/'checked_base/candidate.npz'),'base candidate hash')
    checked=check_point(z,arrays(folder/'checked_base/candidate.npz')['point'])
    require([p.name for p in sorted(folder.glob('query_*'))]==[f'query_{i:03d}' for i in range(1,len(queries))],'native query completeness')
    previous=-1;native_count=0
    for i,q in enumerate(queries[1:],1):
        qdir=folder/f'query_{i:03d}';ret=read(qdir/'return.json')
        if (qdir/'late_return_rejected.json').exists():
            ret.update(returned_status='unknown',accepted_after_receipt=False,terminal='RETURN_PUBLICATION_DEADLINE')
        require(ret==q,'native ledger');phase=q['scope']['phase'];row=q['scope']['row']
        require(phase in ('expanded','contracted') and 0<=row<M and
                row==(previous+1 if phase=='expanded' else previous),'query order')
        previous=row
        require(begin<=q['started_monotonic'] and q['deadline_monotonic']<=end and
                q['return_elapsed_seconds']>=0,'property budget')
        receipt=read(qdir/'receipt.json')
        require(all(receipt[k]==q[k] for k in ('token','scope','deadline_monotonic','artifact_hashes')),'native receipt')
        observed={f.name:sha256(f) for f in qdir.iterdir() if f.is_file() and f.name not in
                  ('receipt.json','return.json','late_return_rejected.json')}
        require(observed==q['artifact_hashes'],'native artifact hashes')
        lo=np.array([t[row]+(-1 if phase=='expanded' else 1)*plan['tolerance']-const[row]])
        if (qdir/'extra.npz').exists():
            extra=arrays(qdir/'extra.npz');req=read(qdir/'request.json')
            require(csr(extra).shape==coeff[row].shape and (csr(extra)!=coeff[row]).nnz==0 and
                    np.array_equal(extra['lb'],lo) and np.array_equal(extra['ub'],np.array([np.inf])),'property query changed')
            require(req['model_sha256']==meta['matrix_sha256'] and req['query_sha256']==sha256(qdir/'extra.npz') and
                    all(req[k]==q[k] for k in ('token','scope','deadline_monotonic')),'native query binding')
        if q['returned_status']!='unknown':
            raw=read(qdir/'native_result.json')
            require(raw==q['native_result'] and raw['finished_monotonic']<q['deadline_monotonic'] and
                    q['started_monotonic']+q['return_elapsed_seconds']<q['deadline_monotonic'],'native result/cost')
            valid=False
            if raw['candidate_sha256'] is not None:
                require(raw['candidate_sha256']==sha256(qdir/'candidate.npz'),'native candidate')
                A,lb,ub=sh._combined_constraints(model,coeff[row],lo,np.array([np.inf]))
                valid=sh._valid_milp_point(model,arrays(qdir/'candidate.npz')['x'],A,lb,ub,plan['tolerance'])
            inferred='feasible' if valid else ('infeasible' if raw['status']==2 else 'unknown')
            require(inferred==q['returned_status'],'native acceptance changed')
        require(q['native_started']==(qdir/'native_started.json').exists(),'native count');native_count+=q['native_started']
    require(native_count==meta['native_queries_started'],'native aggregate')
    for prop in props:
        if 'expanded_query' in prop:
            q=queries[prop['expanded_query']]
            require(q['scope']=={'phase':'expanded','row':prop['row']} and q['returned_status']==prop['status'],'property result binding')
        else:require(prop['status']=='NOT_STARTED','unproved property')
    if status=='certified':require(all(p['status']=='infeasible' for p in props),'incomplete SAFE')
    return {'status':status,'reason':meta['reason'],'base_status':meta['base_status'],
        'base_check':checked,'base_seconds':base['return_elapsed_seconds'],'properties':props,
        'property_status_counts':dict(Counter(p['status'] for p in props)),
        'expert_elapsed_before_publication':meta['elapsed_before_publication'],'allocation_seconds':plan['total_seconds'],
        'native_property_calls':native_count,'matrix_sha256':meta['matrix_sha256'],
        'queries':[{k:q[k] for k in ('scope','terminal','returned_status','native_started','return_elapsed_seconds')} for q in queries]}


def audit():
    began=time.monotonic();cfg=read(control.CONFIG);control.validate(cfg);root=control.OUTPUT
    summary=read(root/'summary.json');launch=read(root/'launch.json');h=sha256(control.CONFIG)
    require(summary['config_sha256']==launch['config_sha256']==h and launch['variants']==list(control.VARIANTS)
            and not launch['automatic_followup'],'batch identity')
    require([r['variant'] for r in summary['rows']]==list(control.VARIANTS),'denominator')
    reviews=[];previous_error=False
    for row in summary['rows']:
        variant=row['variant']
        if row['status']=='NOT_STARTED_AFTER_ERROR':require(previous_error,'unexplained omission');continue
        folder=root/variant/'mnist_0_act';receipt=read(folder/'receipt.json')
        require(read(folder/'terminal.json')==row and sha256(folder/'receipt.json')==row['receipt_sha256'],'terminal identity')
        check_receipt(receipt,row,cfg,control.command(cfg,variant))
        require(all(row[k]==v for k,v in collect_terminal(folder,receipt).items()),'outer precedence')
        for name in ('stdout','stderr'):require(sha256(folder/f'{name}.txt')==receipt[name+'_sha256'],'stream binding')
        require(inventory(folder/'protected')=={k.removeprefix('protected/'):v for k,v in row['protected_artifacts'].items()},'evidence inventory')
        trace=None
        if row['trace_sha256']:
            require(sha256(folder/'trace.jsonl')==row['trace_sha256'],'trace binding')
            trace=check_trace(folder/'trace.jsonl',row['seconds'],control.identity(cfg,variant),killed=receipt['status']!='COMPLETED')
        result=read(folder/'result.json') if row['result_sha256'] and not row['result_parse_error'] else None
        if result:
            require(result['config_sha256']==h and result['request_id']=='mnist_0' and result['arm']=='act' and
                    result['label']==cfg['requests'][0]['label'] and
                    result['tensor_file_sha256']==cfg['files'][cfg['requests'][0]['tensor_file']] and result['worker_seconds']<=row['seconds'],
                    'request identity')
            if receipt['status']=='COMPLETED':check_candidate(result,'act')
        ev=[evaluate(p,19,h,cfg['files'][cfg['requests'][0]['tensor_file']],receipt['status']!='COMPLETED')
            for p in sorted((folder/'protected').glob('evaluation_*'))]
        if result and receipt['status']=='COMPLETED' and 'expert_statuses' in result:
            require([e['status'] for e in ev]==list(result['expert_statuses'].values()),'expert aggregate')
        reviews.append({'variant':variant,'row':row,'result':result,'evaluations':ev,'trace':trace})
        previous_error=row['status'] in ('ERROR','SOURCE_CHANGED')
    matrix_equal=None
    if len(reviews)==2 and all(len(r['evaluations'])==1 for r in reviews):
        matrix_equal=reviews[0]['evaluations'][0]['matrix_sha256']==reviews[1]['evaluations'][0]['matrix_sha256']
        require(matrix_equal,'two arms did not solve same matrix')
    cost=read(root/'batch_cost.json')
    require(cost['config_sha256']==h and cost['charged_request_seconds']==sum(r.get('seconds',0) for r in summary['rows'])
            and cost['batch_wall_through_summary_seconds']>=cost['charged_request_seconds'],'batch cost')
    return {'audit':'PASS','issues':0,'config_sha256':h,'rows':reviews,'cost':cost,'same_expert_matrix':matrix_equal,
            'files':inventory(root),'audit_seconds':time.monotonic()-began,'new_audit_solves':0,
            'claim':'Same-source execution diagnostic under original float policy; no all-domain source proof or relaxation attribution.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'output exists');value=audit();write(a.output,value)
    print(value['audit'],[(r['variant'],r['row']['status']) for r in value['rows']])
