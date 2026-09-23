"""Saved-only nonzero sign and full-obligation audit, no new solver/forward."""
import argparse
from collections import Counter
from fractions import Fraction
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from act.back_end.solver import solver_hz as sh
from audit_metamoe_current_assignment import read,require,inventory
from audit_metamoe_protected_smoke_r1 import arrays,csr
from audit_metamoe_checked_base_control import evaluate
from audit_metamoe_checked_routing_control import routing_query
from audit_metamoe_csr_paired_r4 import check_receipt,check_candidate
from audit_conv_f0_timing import check_trace
from metamoe_csr_paired_r4 import collect_terminal
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
import metamoe_nonzero_precheck_control as control


def source_binding(z,r,route_model):
    """Independent coefficient mapping; not source-network/HZ inclusion proof."""
    nc,nb=r['n_cont'],r['n_bin'];no=r['n_out'];ne,ni=r['n_eq'],r['n_ineq']
    m={k:csr(z,k+'_') for k in ('Gc','Gb','Ac','Ab','Auc','Aub')}
    require(z['c'].shape==(no,) and z['b'].shape==(ne,) and z['ub'].shape==(ni,) and
        all(m[k].shape==shape for k,shape in {'Gc':(no,nc),'Gb':(no,nb),'Ac':(ne,nc),
            'Ab':(ne,nb),'Auc':(ni,nc),'Aub':(ni,nb)}.items()),'source dimensions')
    require(all(np.isfinite(v).all() for v in (z['c'],z['b'],z['ub'])) and
            all(np.isfinite(v.data).all() for v in m.values()),'nonfinite source')
    if route_model is not None:
        value=sparse.hstack((m['Gc'],2*m['Gb']),format='csr')
        A=sparse.vstack((sparse.hstack((m['Ac'],2*m['Ab'])),sparse.hstack((m['Auc'],2*m['Aub']))),format='csr')
        eq=z['b']+np.asarray(m['Ab'].sum(axis=1)).ravel()
        le=z['ub']+np.asarray(m['Aub'].sum(axis=1)).ravel()
        require(value.shape==csr(route_model,'value_').shape and (value!=csr(route_model,'value_')).nnz==0 and
            np.array_equal(z['c']-np.asarray(m['Gb'].sum(axis=1)).ravel(),route_model['value_center']) and
            A.shape==csr(route_model).shape and (A!=csr(route_model)).nnz==0 and
            np.array_equal(np.r_[eq,np.full(ni,-np.inf)],route_model['row_lb']) and
            np.array_equal(np.r_[eq,le],route_model['row_ub']) and
            np.array_equal(route_model['var_lb'],np.r_[-np.ones(nc),np.zeros(nb)]) and
            np.array_equal(route_model['var_ub'],np.ones(nc+nb)) and
            np.array_equal(route_model['integrality'],np.r_[np.zeros(nc),np.ones(nb)]),'wrong guarded routing object')
    return m


def nonzero_query(folder,row,enabled,request_hash,input_hash,clock,trace,partial,route_model=None):
    if not (folder/'result.json').exists():
        require(partial,'missing nonzero terminal');return {'accepted':False,'status':'PARTIAL'}
    r=read(folder/'result.json');begin=read(folder/'begin.json')
    require(r['schema']=='selected-score-nonzero-v1' and r['row']==row and r['enabled']==enabled and
        all(r[k]==v for k,v in begin.items() if k not in ('evidence','native_invoked')),'nonzero initial binding')
    require(r['scope']['request_sha256']==request_hash and r['scope']['input_sha256']==input_hash and
            bool(r['scope']['evaluation_nonce']),'nonzero request binding')
    start,deadline,end=r['started_monotonic'],r['deadline_monotonic'],r['finished_monotonic']
    require(all(math.isfinite(t) for t in (start,deadline,end)) and clock<=start<=end and
        0<=deadline-start<=30 and r['precheck_deadline_monotonic']==min(deadline,start+3.) and
        r['numerical_policy']==sh.hz_numerical_policy_manifest() and not r['source_complete'],'nonzero budget/policy')
    checked=None;fast=None
    if r.get('source_hz_sha256'):
        require(sha256(folder/'source_hz.npz')==r['source_hz_sha256'],'nonzero source hash')
        z=arrays(folder/'source_hz.npz');m=source_binding(z,r,route_model)
        require(0<=row<r['n_out'],'nonzero row')
        # Deliberately do not call exact_generator_sign or sparse_hz_fast_bounds.
        values=[float(v) for name in ('Gc','Gb') for v in m[name].data[m[name].indptr[row]:m[name].indptr[row+1]]]
        radius=sum((abs(Fraction(*v.as_integer_ratio())) for v in values),Fraction(0))
        c=Fraction(*float(z['c'][row]).as_integer_ratio());lo,hi=c-radius,c+radius
        checked={'lower':str(lo),'upper':str(hi),'sign':1 if lo>0 else (-1 if hi<0 else 0),'terms':len(values)}
        rad=np.asarray(abs(m['Gc']).sum(axis=1)).ravel()+np.asarray(abs(m['Gb']).sum(axis=1)).ravel()
        fl,fu=float(z['c'][row]-rad[row]),float(z['c'][row]+rad[row])
        fast={'lower':fl,'upper':fu,'accepted':math.isfinite(fl) and math.isfinite(fu) and fl<=fu and (fl>0 or fu<0)}
        if 'fast' in r:
            require(all(r['fast'][k]==v for k,v in fast.items()) and r['fast']['row']==row and
                    r['fast']['solves']==0 and not r['fast']['exact'],'fallback enclosure changed')
            attempt=read(folder/'attempt.json')
            require(attempt['scope']==r['scope'] and attempt['source_hz_sha256']==r['source_hz_sha256'] and
                    attempt['fast']==r['fast'],'precheck attempt binding')
        if 'exact_sign' in r:require(r['exact_sign']==checked,'exact sign evidence')
    result=r['result'];accepted=result['accepted']
    require(result['row']==row,'result row')
    if r['evidence']=='CURRENT_GENERATOR_BOX_NONZERO':
        require(enabled and not r['native_invoked'] and checked is not None and checked['sign']!=0 and
            fast['accepted'] and r['precheck_accepted'] and result['accepted'] and result['solves']==0 and
            result['lower']==fast['lower'] and result['upper']==fast['upper'] and
            checked['sign']==(1 if fast['lower']>0 else -1) and not result['exact'] and
            result['lower_status']==result['upper_status']==['fast_nonzero_checked'],'invalid nonzero shortcut')
        if not (folder/'late_result_rejected.json').exists():require(end<r['precheck_deadline_monotonic'],'late nonzero result')
    if r['native_invoked']:
        n=read(folder/'native_entry.json')
        require(n['scope']==r['scope'] and n.get('source_hz_sha256')==r.get('source_hz_sha256') and
                n['deadline_monotonic']==deadline,'native support binding')
        if 'native_finished_monotonic' in r:
            a,b=r['native_started_monotonic'],r['native_finished_monotonic']
            require(start<=a<deadline and a<=b<=end and 0<r['native_budget_seconds']<=deadline-a and
                    r['native_late']==(b>=deadline),'remaining support budget')
            spans=[s for s in trace['spans'] if s['name']=='act.back_end.solver.solver_hz.hz_support_bounds'
                and s['end'] is not None and a<=clock+s['start']<=clock+s['end']<=b]
            require(len(spans)==1 and spans[0]['arguments']['time_limit']==r['native_budget_seconds'], 'native support trace')
            require(result==r['native_result'] and dict(Counter(result['lower_status']))==spans[0]['result']['lower_status_counts'] and
                    dict(Counter(result['upper_status']))==spans[0]['result']['upper_status_counts'] and
                    result['solves']==spans[0]['result']['solves'],'native support return')
            if result['lower_status']==result['upper_status']==['fast_fallback'] and fast:
                require(all(result[k]==v for k,v in fast.items()),'native fallback differs')
    if accepted:require(r['evidence'] in ('CURRENT_GENERATOR_BOX_NONZERO','ORIGINAL_SUPPORT_POLICY'),'unproved nonzero')
    if (folder/'late_result_rejected.json').exists():accepted=False
    return {'row':row,'accepted':accepted,'evidence':r['evidence'],'source_hz_sha256':r.get('source_hz_sha256'),
        'scope':r['scope'],'exact_generator_sign':checked,'result':result,
        'native_invoked':r['native_invoked'],'seconds_before_publication':r['elapsed_before_publication']}


def audit():
    began=time.monotonic();cfg=read(control.CONFIG);control.validate(cfg);root=control.OUTPUT;h=sha256(control.CONFIG)
    summary,launch=read(root/'summary.json'),read(root/'launch.json')
    require(summary['config_sha256']==launch['config_sha256']==h and launch['variants']==list(control.VARIANTS)
        and not launch['automatic_followup'] and [r['variant'] for r in summary['rows']]==list(control.VARIANTS),'batch binding')
    rows=[];blocked=False;nonces=set();accounted=0.
    for row in summary['rows']:
        variant=row['variant'];folder=root/variant/'mnist_0_act'
        if row['status']=='NOT_STARTED_AFTER_ERROR':require(blocked,'unexplained omission');continue
        require(not blocked,'executed after stop-on-error')
        receipt=read(folder/'receipt.json');killed=receipt['status']!='COMPLETED'
        require(row==read(folder/'terminal.json') and row['receipt_sha256']==sha256(folder/'receipt.json'),'terminal binding')
        check_receipt(receipt,row,cfg,control.command(cfg,variant))
        post=row['postflight_inventory_seconds'];require(math.isfinite(post) and post>=0,'postflight cost')
        accounted+=receipt['total_with_postflight_seconds']+post
        require(all(row[k]==v for k,v in collect_terminal(folder,receipt).items()),'outer precedence')
        for s in ('stdout','stderr'):require(receipt[s+'_sha256']==sha256(folder/f'{s}.txt'),'stream hash')
        require({area+'/'+k:v for area in ('protected','routing','nonzero') for k,v in inventory(folder/area).items()}
                ==row['evidence_artifacts'],'evidence inventory')
        trace=None;runtime=None
        if row['trace_sha256']:
            require(row['trace_sha256']==sha256(folder/'trace.jsonl') and row['runtime_sha256']==sha256(folder/'runtime.json'),'trace binding')
            runtime=read(folder/'runtime.json');require(runtime['identity']==control.identity(cfg,variant),'worker scope')
            trace=check_trace(folder/'trace.jsonl',row['seconds'],runtime['identity'],killed=killed)
        else:require(killed,'completed without trace')
        result=read(folder/'result.json') if row['result_sha256'] and not row['result_parse_error'] else None
        if result:
            require(result['config_sha256']==h and result['request_id']=='mnist_0' and result['arm']=='act' and
                result['label']==cfg['requests'][0]['label'] and result['worker_seconds']<=row['seconds'] and
                result['tensor_file_sha256']==cfg['files'][cfg['requests'][0]['tensor_file']],'request identity')
            if not killed:check_candidate(result,'act')
        routes=[];ih=cfg['files'][cfg['requests'][0]['tensor_file']]
        for i,p in enumerate(sorted((folder/'routing').glob('query_*'))):
            require(p.name==f'query_{i:03d}' and i<2,'route coverage')
            routes.append(routing_query(p,i,True,h,ih,runtime['worker_started_monotonic'],trace,killed))
        if result and not killed and 'candidates' in result:
            require(len(routes)==2 and result['candidates']==[q['index'] for q in routes if q['status']!='infeasible'] and
                result['excluded']==[q['index'] for q in routes if q['status']=='infeasible'] and
                result['unresolved']==[q['index'] for q in routes if q['status']=='unknown'],'route aggregate')
        experts=[evaluate(p,19,h,ih,killed) for p in sorted((folder/'protected').glob('evaluation_*'))]
        if result and not killed and 'expert_statuses' in result:
            require([e['status'] for e in experts]==list(result['expert_statuses'].values()),'expert aggregate')
        nonzero=[]
        for i,p in enumerate(sorted((folder/'nonzero').glob('query_*'))):
            require(p.name==f'query_{i:03d}' and i<2,'nonzero query coverage')
            if not (p/'begin.json').exists():require(killed,'missing nonzero begin');continue
            score=read(p/'begin.json')['row'];require(type(score) is int and 0<=score<2,'score identity')
            routed=folder/f'routing/query_{score:03d}/model.npz'
            model=arrays(routed) if routed.exists() else None
            require(model is not None or killed,'unbound guarded object')
            nonzero.append(nonzero_query(p,score,variant=='support_precheck',h,ih,runtime['worker_started_monotonic'],trace,killed,model))
        for q in routes+nonzero:
            if q.get('scope'):
                nonce=q['scope']['evaluation_nonce'];require(nonce not in nonces,'cross-query nonce');nonces.add(nonce)
        if result and not killed and result.get('nonzero_obligations'):
            obligations=result['nonzero_obligations']
            require(len(nonzero)==len(obligations) and [q['row'] for q in nonzero]==[o['expert'] for o in obligations],'missing score obligation')
            for q,o in zip(nonzero,obligations):
                require(q['accepted']==o['accepted'] and all(q['result'][k]==o[k] for k in
                    ('lower','upper','lower_status','upper_status')),'score result aggregate')
        timing={name:value for name,value in (trace['aggregates'] if trace else {}).items() if name in
            ('act.back_end.moe.hz_routing.analyze_candidates','act.back_end.solver.solver_hz.hz_support_bounds',
             'ProtectedHZSolver.evaluate_spec','act.back_end.solver.solver_hz.milp')}
        rows.append({'variant':variant,'status':row['status'],'seconds':row['seconds'],'result':result,'receipt':receipt,
            'routing':routes,'expert_evaluations':experts,'nonzero':nonzero,'timing':timing,
            'trace_last_hash':trace['last_hash'] if trace else None,'right_censored_spans':trace['open_span_ids'] if trace else [],
            'postflight_inventory_seconds':post})
        blocked=row['status'] in ('ERROR','SOURCE_CHANGED')
    equality={}
    for key,field in (('routing','model_sha256'),('expert_evaluations','matrix_sha256'),('nonzero','source_hz_sha256')):
        same=None
        if len(rows)==2 and all(r[key] and all(q.get(field) for q in r[key]) for r in rows):
            same=[q[field] for q in rows[0][key]]==[q[field] for q in rows[1][key]];require(same,key+' object differs')
        equality[key]=same
    cost=read(root/'batch_cost.json')
    require(cost['config_sha256']==h and cost['charged_request_seconds']==sum(r.get('seconds',0) for r in summary['rows']) and
            cost['batch_wall_through_summary_seconds']>=accounted,'batch cost')
    return {'audit':'PASS','issues':0,'config_sha256':h,'rows':rows,'cost':cost,'same_objects':equality,
        'files':inventory(root),'audit_seconds':time.monotonic()-began,'new_audit_solves':0,
        'claim':'Single old-input cost control; exact sign of stored generator box checked. Upstream enclosure and native infeasibility still trusted; not source-complete.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'output exists');v=audit();write(a.output,v)
    print(v['audit'],[(r['variant'],r['status'],r['seconds']) for r in v['rows']])
