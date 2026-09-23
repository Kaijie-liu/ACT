"""Saved-only guarded routing feasibility, query coverage and complete cost audit."""
import argparse
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.current_assignment import model_fingerprint
from audit_metamoe_current_assignment import read,require,check_point,inventory
from audit_metamoe_protected_smoke_r1 import arrays,csr
from audit_metamoe_checked_base_control import evaluate
from audit_metamoe_csr_paired_r4 import check_receipt,check_candidate
from audit_conv_f0_timing import check_trace
from metamoe_csr_paired_r4 import collect_terminal
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
import metamoe_checked_routing_control as control


def routing_query(folder,index,enabled,request_hash,input_hash,clock,trace,partial):
    if not (folder/'result.json').exists():
        require(partial,'completed query missing result');return {'status':'PARTIAL','accepted':False}
    r=read(folder/'result.json');b=read(folder/'begin.json')
    require(r['schema']=='checked-route-feasibility-v1' and r['index']==index and r['enabled']==enabled and
            all(r[k]==v for k,v in b.items() if k not in ('status','evidence','native_invoked')),'routing initial identity')
    require(r['scope']['request_sha256']==request_hash and r['scope']['input_sha256']==input_hash
            and bool(r['scope']['evaluation_nonce']),'current routing scope')
    start,deadline,end=r['started_monotonic'],r['deadline_monotonic'],r['finished_monotonic']
    require(all(math.isfinite(t) for t in (start,deadline,end)) and clock<=start<=end and
            0<=deadline-start<=30 and r['proposal_deadline_monotonic']==min(deadline,start+3.) and
            r['tolerance']==1e-7 and r['numerical_policy']==sh.hz_numerical_policy_manifest(),'query budget/policy')
    status='unknown' if (folder/'late_result_rejected.json').exists() else r['status']
    z=None;fp=None;checked=None
    if r.get('model_sha256'):
        require(sha256(folder/'model.npz')==r['model_sha256'],'guarded model identity');z=arrays(folder/'model.npz')
        nc,nb=r['n_cont'],r['n_bin'];A=csr(z)
        require(np.array_equal(z['integrality'],np.r_[np.zeros(nc),np.ones(nb)]) and
                A.shape==(r['constraint_rows'],nc+nb) and A.nnz==r['constraint_nnz'],'no dropped guard/integrality')
        m=sh._HZMILP(z['value_center'],csr(z,'value_'),A,z['row_lb'],z['row_ub'],z['var_lb'],z['var_ub'],z['integrality'],nc,nb)
        fp=model_fingerprint(m)
    if r.get('proposal'):
        p=r['proposal'];require(enabled and p['scope']==r['scope'] and p['model_fingerprint']==fp and
                p['point_sha256']==sha256(folder/'proposal.npz'),'proposal binding')
        if p['check']['accepted']:
            checked=check_point(z,arrays(folder/'proposal.npz')['point'])
        attempt=read(folder/'attempt.json')
        require(attempt['proposal']==p and attempt['scope']==r['scope'],'attempt binding')
    if r['evidence']=='CURRENT_FULL_GUARDED_MATRIX_POINT':
        require(enabled and not r['native_invoked'] and r['status']=='feasible' and checked is not None,
                'point evidence cannot exclude/prove property')
        if status!='unknown':require(end<r['proposal_deadline_monotonic'],'late checked assignment')
    if r['native_invoked']:
        entry=read(folder/'native_entry.json')
        require(entry['scope']==r['scope'] and entry['model_sha256']==r['model_sha256'] and
                entry['deadline_monotonic']==deadline,'native binding')
        if 'native_finished_monotonic' in r:
            a,c=r['native_started_monotonic'],r['native_finished_monotonic']
            require(start<=a<deadline and a<=c<=end and
                    r['native_returned_after_local_deadline']==(c>=deadline),'native remaining deadline/overrun')
            spans=[s for s in trace['spans'] if s['name']=='act.back_end.solver.solver_hz._solve_hz_feasibility'
                   and s['end'] is not None and a<=clock+s['start']<=clock+s['end']<=c]
            require(len(spans)==1 and spans[0]['result']['status']==r['native_status'],'native trace status binding')
            if r['native_status']=='feasible':
                require(r['native_point_sha256']==sha256(folder/'native_point.npz'),'native point binding')
                check_point(z,arrays(folder/'native_point.npz')['point'])
            if r['native_status']=='infeasible':
                calls=[s for s in trace['spans'] if s['parent']==spans[0]['id'] and s['name']=='act.back_end.solver.solver_hz.milp']
                require(len(calls)==1 and calls[0]['result']['status']==2,'infeasible needs native status2')
                require(calls[0]['arguments']['integral_entries']==r['n_bin'] and
                        calls[0]['arguments']['matrix_shape']==list(A.shape),'native constraint shape')
            require(r['status']==r['native_status'] or r.get('error'),'native status substitution')
        else:require(r['status']=='unknown','incomplete native evidence')
    if status!='unknown':require(z is not None and r['evidence']!='NONE','missing feasibility evidence')
    return {'index':index,'status':status,'evidence':r['evidence'],'model_sha256':r.get('model_sha256'),
        'model_fingerprint':fp,'scope':r['scope'],'base_check':checked,'native_invoked':r['native_invoked'],
        'native_late':r.get('native_returned_after_local_deadline',False),
        'seconds_before_publication':r['elapsed_before_publication'],'declared_seconds':deadline-start}


def audit():
    began=time.monotonic();cfg=read(control.CONFIG);control.validate(cfg);root=control.OUTPUT;h=sha256(control.CONFIG)
    summary,launch=read(root/'summary.json'),read(root/'launch.json')
    require(summary['config_sha256']==launch['config_sha256']==h and launch['variants']==list(control.VARIANTS)
            and not launch['automatic_followup'] and [r['variant'] for r in summary['rows']]==list(control.VARIANTS),'batch identity')
    rows=[];blocked=False;nonces=set();accounted_seconds=0.
    for row in summary['rows']:
        variant=row['variant'];folder=root/variant/'mnist_0_act'
        if row['status']=='NOT_STARTED_AFTER_ERROR':require(blocked,'unexplained omission');continue
        require(not blocked,'executed after stop-on-error')
        receipt=read(folder/'receipt.json');killed=receipt['status']!='COMPLETED'
        require(row==read(folder/'terminal.json') and row['receipt_sha256']==sha256(folder/'receipt.json'),'terminal identity')
        check_receipt(receipt,row,cfg,control.command(cfg,variant))
        post=row['postflight_inventory_seconds']
        require(type(post) in (int,float) and math.isfinite(post) and post>=0,'postflight cost')
        accounted_seconds+=receipt['total_with_postflight_seconds']+post
        require(all(row[k]==v for k,v in collect_terminal(folder,receipt).items()),'outer precedence')
        for s in ('stdout','stderr'):require(receipt[s+'_sha256']==sha256(folder/f'{s}.txt'),'stream hash')
        observed={area+'/'+k:v for area in ('protected','routing') for k,v in inventory(folder/area).items()}
        require(observed==row['evidence_artifacts'],'evidence inventory')
        trace=None;runtime=None
        if row['trace_sha256']:
            require(row['trace_sha256']==sha256(folder/'trace.jsonl') and row['runtime_sha256']==sha256(folder/'runtime.json'),'trace/clock identity')
            runtime=read(folder/'runtime.json');require(runtime['identity']==control.identity(cfg,variant),'worker scope')
            trace=check_trace(folder/'trace.jsonl',row['seconds'],runtime['identity'],killed=killed)
        elif not killed:raise ValueError('completed request has no trace')
        result=read(folder/'result.json') if row['result_sha256'] and not row['result_parse_error'] else None
        if result:
            require(result['config_sha256']==h and result['request_id']=='mnist_0' and result['arm']=='act' and
                    result['label']==cfg['requests'][0]['label'] and result['worker_seconds']<=row['seconds'] and
                    result['tensor_file_sha256']==cfg['files'][cfg['requests'][0]['tensor_file']],'request identity')
            if not killed:check_candidate(result,'act')
        routes=[]
        for i,p in enumerate(sorted((folder/'routing').glob('query_*'))):
            require(p.name==f'query_{i:03d}' and i<2,'fixed routing query coverage')
            r=routing_query(p,i,variant=='router_checked',h,cfg['files'][cfg['requests'][0]['tensor_file']],
                            runtime['worker_started_monotonic'],trace,killed)
            if r.get('scope'):
                nonce=r['scope']['evaluation_nonce'];require(nonce not in nonces,'cross-query nonce reuse');nonces.add(nonce)
            routes.append(r)
        if result and not killed and 'candidates' in result:
            require(len(routes)==2,'missing route obligation')
            require(result['candidates']==[r['index'] for r in routes if r['status']!='infeasible'] and
                    result['excluded']==[r['index'] for r in routes if r['status']=='infeasible'] and
                    result['unresolved']==[r['index'] for r in routes if r['status']=='unknown'],'route aggregate')
        experts=[evaluate(p,19,h,cfg['files'][cfg['requests'][0]['tensor_file']],killed)
                 for p in sorted((folder/'protected').glob('evaluation_*'))]
        if result and not killed and 'expert_statuses' in result:
            require([e['status'] for e in experts]==list(result['expert_statuses'].values()),'expert aggregate')
        # Trace hash/invariants are checked above; compact timing instead of a
        # duplicate of every large solver metadata record in the Git archive.
        timing={name:value for name,value in (trace['aggregates'] if trace else {}).items()
                if name in ('act.back_end.moe.hz_routing.analyze_candidates','act.back_end.solver.solver_hz.hz_support_bounds',
                            'ProtectedHZSolver.evaluate_spec','act.back_end.solver.solver_hz.milp')}
        rows.append({'variant':variant,'status':row['status'],'seconds':row['seconds'],'result':result,
            'receipt':receipt,'routing':routes,'expert_evaluations':experts,'timing':timing,
            'trace_last_hash':trace['last_hash'] if trace else None,'right_censored_spans':trace['open_span_ids'] if trace else [],
            'postflight_inventory_seconds':row['postflight_inventory_seconds']})
        blocked=row['status'] in ('ERROR','SOURCE_CHANGED')
    same_routes=same_expert=None
    if len(rows)==2 and all(len(r['routing'])==2 and all(q.get('model_sha256') for q in r['routing']) for r in rows):
        same_routes=[q['model_sha256'] for q in rows[0]['routing']]==[q['model_sha256'] for q in rows[1]['routing']]
        require(same_routes,'route constraints differ between arms')
    if len(rows)==2 and all(len(r['expert_evaluations'])==1 and r['expert_evaluations'][0].get('matrix_sha256') for r in rows):
        same_expert=rows[0]['expert_evaluations'][0]['matrix_sha256']==rows[1]['expert_evaluations'][0]['matrix_sha256']
        require(same_expert,'expert constraints differ between arms')
    cost=read(root/'batch_cost.json')
    require(cost['config_sha256']==h and cost['charged_request_seconds']==sum(r.get('seconds',0) for r in summary['rows']) and
            cost['batch_wall_through_summary_seconds']>=accounted_seconds,'batch cost')
    return {'audit':'PASS','issues':0,'config_sha256':h,'rows':rows,'cost':cost,'same_guarded_router_matrices':same_routes,
            'same_expert_matrix':same_expert,'files':inventory(root),'audit_seconds':time.monotonic()-began,'new_audit_solves':0,
            'claim':'Single old-input routing-feasibility control. Full guard points checked; native infeasibility and source lowering trusted. Not formal source-complete or population speedup evidence.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'output exists');value=audit();write(a.output,value)
    print(value['audit'],[(r['variant'],r['status'],r['seconds']) for r in value['rows']])
