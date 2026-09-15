"""Pre-F0 construction-only capture and frozen exact-rational proof queries."""
import argparse
from dataclasses import asdict
import fcntl
from fractions import Fraction
import os
from pathlib import Path
import sys
import time

from scripts.conv_pre_f0_r2_contract import (ROOT,DEFAULT,FREEZE,read,save,sha,git,parents,job,sources,
                                        verify_freeze,validate_job,publication_gate)
from scripts.run_conv_sign_lp import subprocess_stage,resources
from scripts.check_conv_request_sign_lp import expected_scope,property_vector
from scripts.check_conv_pre_f0_r2 import TRUSTED,order_bounds


class CapturedPreF0(BaseException):pass


def reference(directory,name):return {'file':name,'sha256':sha(directory/name)}


def capture_worker(directory):
    from scripts.budget_contract_v2 import verify_v2
    from act.pipeline.moe.external_pair_worker import load
    from act.back_end.solver.hz_lp_export import export,snapshot
    from act.back_end.solver.lp_certificate import identity
    import act.pipeline.moe.paired_monolithic as mono
    j=validate_job(directory);req=j['parent_request'];started=time.monotonic()
    if (directory/'manifest.json').exists():raise FileExistsError('one shot')
    model,tensors=load(req);original=mono._run_monolithic;manifest={}
    def construct(*,model,center,clean_prediction,internal,config,reuse=None,budget=None):
        routes=asdict(internal['route_sets']);pairs=[list(p) for p in routes['feasible']]
        if pairs!=j['case']['expected_pairs'] or not routes['exact'] or routes['unresolved']:
            raise ValueError('frozen route coverage unavailable; no substitute')
        pair=tuple(pairs[0]);props=mono.linear_safety_rows(internal['output_spec'],10)
        if len(props)!=9:raise ValueError('wrong property count')
        rows=[]
        for i,(q,c) in enumerate(props):
            # linear_safety_rows returns CPU Torch scalars in real execution.
            # Convert only after fixing the classification template below;
            # its +/-1 and0 entries are exactly representable Python numbers.
            q=[float(v) for v in q];c=float(c)
            if list(q)!=property_vector(clean_prediction,j['competitors'][i]) or c!=0:raise ValueError('wrong property')
            proof=mono.reuse_property(reuse['facts'],reuse['scope'],pair,i) if reuse else None
            row={'pair':list(pair),'property_index':i,'competitor':j['competitors'][i],'q':list(q),'constant':c,
                 'kind':'reused' if proof else 'residual'}
            if proof:row['proof']=proof
            else:row.update(difference_lower=f'p{i}_lower',difference_upper=f'p{i}_upper')
            rows.append(row)
        support,solver=config['support'],config['solver']
        hc=mono.HybridZConfig(max_input_dim=1024,guarded_support_enabled=True,
            guarded_support_lp_neurons=int(support['lp_neurons']),guarded_support_milp_neurons=int(support['milp_neurons']),
            guarded_support_lp_time_limit=float(support['lp_time_limit']),guarded_support_milp_time_limit=float(support['milp_time_limit']),
            guarded_support_solver_backend=str(support.get('solver_backend','scipy')),
            expert_property_solver_backend=str(solver.get('backend','scipy')))
        budget.check('monolithic_pair_propagation')
        hc.guarded_support_lp_time_limit=budget.limit('monolithic_lp_support',float(support['lp_time_limit']))
        hc.guarded_support_milp_time_limit=budget.limit('monolithic_mip_support',float(support['milp_time_limit']))
        router,program=internal['router'],internal['program']
        conditioned=mono.condition_topk_set(router.output_hz,pair).hz
        entry=mono.guarded_input_topk_set(router.input_hz,router.output_hz,pair).hz
        propagated=mono.shared_input_pair_propagation(program.experts[pair[0]],program.experts[pair[1]],
            entry_hz=entry,hybridz_config=hc,expert_relation='shared_input')
        joint=propagated.joint.output_hz
        save(directory/'joint_hz.json',snapshot(joint));save(directory/'router_hz.json',snapshot(conditioned))
        manifest.update(schema='CONV_PRE_F0_RATIONAL_R2_MANIFEST',request=expected_scope(j),trusted_base=TRUSTED,
            positive_threshold=1e-7,routes=routes,common_facts=reference(directory,'common_facts.json'),
            joint_source=reference(directory,'joint_hz.json'),router_source=reference(directory,'router_hz.json'),
            expert_order=list(pair),generation_complete=False,obligations=rows,supports={})
        save(directory/'manifest.json',manifest)
        def record_support(key,kind,index,hz,q):
            budget.check('pre_f0_export');record=export(hz,q,sparse=True);name=key+'.export.json';save(directory/name,record)
            manifest['supports'][key]={'kind':kind,'property_index':index,'pair':list(pair),'request':expected_scope(j),
                'source_sha256':record['source_sha256'],'export':reference(directory,name),'status':'PENDING','certificate':None}
            save(directory/'manifest.json',manifest)
        qm=[0]*4;qm[pair[0]]=1;qm[pair[1]]=-1
        record_support('gate_lower','router_order',None,conditioned,qm)
        record_support('gate_upper','router_order',None,conditioned,[-v for v in qm])
        for row in rows:
            if row['kind']=='reused':continue
            qd=row['q']+[-v for v in row['q']]
            record_support(row['difference_lower'],'difference',row['property_index'],joint,qd)
            record_support(row['difference_upper'],'difference',row['property_index'],joint,[-v for v in qd])
        budget.check('pre_f0_capture_complete');manifest['generation_complete']=True
        save(directory/'manifest.json',manifest);raise CapturedPreF0()
    mono._run_monolithic=construct
    # These functions must never be reached by this new evidence route.
    forbidden=('build_weighted_top2_f0','compute_weighted_top2_gate_range','solve_monolithic_weighted_top2_f0')
    originals={k:getattr(mono,k) for k in forbidden}
    def no_float_f0(*a,**k):raise AssertionError('floating F0/gate/property solve forbidden')
    for k in forbidden:setattr(mono,k,no_float_f0)
    try:
        try:
            verify_v2(model,tensors['center'],req['epsilon'],read(req['config']['path']),
                journal_path=directory/'budget_journal.jsonl',started=started,
                identity={'request_proof_job_sha256':sha(directory/'job.json')},
                expected_clean_prediction=req['sample']['label'],
                checkpoint_identity={'path':req['subject']['checkpoint'],'sha256':req['subject']['checkpoint_sha256']},
                common_fact_callback=lambda v:save(directory/'common_facts.json',v))
        except CapturedPreF0:pass
        else:raise ValueError('front-end did not complete pre-F0 capture')
    finally:
        mono._run_monolithic=original
        for k,v in originals.items():setattr(mono,k,v)
    save(directory/'generation.json',manifest)
    save(directory/'capture.json',{'generation_sha256':sha(directory/'generation.json'),
        'journal_sha256':sha(directory/'budget_journal.jsonl'),'capture_seconds':time.monotonic()-started,
        'floating_F0_called':False,'weighted_property_solver_called':False,
        'source':'fresh ordered shared expert HZ before property projection; fresh conditioned router HZ'})


def proposal_worker(directory):
    from act.back_end.solver.lp_certificate import propose,identity
    from act.back_end.solver.check_hz_lp_export import check_export
    from act.back_end.solver.rational_mccormick import build
    from act.back_end.solver.check_rational_mccormick import check_construction
    j=validate_job(directory);m=read(directory/'manifest.json');started=time.monotonic();calls=[];bounds={}
    if (directory/'proposal.json').exists():raise FileExistsError('no repeat')
    def solve(key,record):
        if len(calls)>=j['protocol']['max_lp_queries']:raise ValueError('query cap')
        t=time.monotonic()
        try:cert=propose(record['lp'],time_limit=j['protocol']['solver_seconds_per_lp'])
        except ValueError as exc:
            if str(exc)!='proposal solver did not complete':raise
            cert=None
        name=key+'.certificate.json'
        if cert is not None:save(directory/name,cert)
        calls.append({'key':key,'status':'PROPOSED' if cert else 'UNAVAILABLE','seconds':time.monotonic()-t})
        save(directory/'query_log.json',calls)
        return cert,reference(directory,name) if cert else None
    order=['gate_lower','gate_upper']+[key for row in m['obligations'] if row['kind']=='residual'
                                      for key in (row['difference_lower'],row['difference_upper'])]
    for key in order:
        item=m['supports'][key]
        if item['status']!='PENDING':raise ValueError('no resume')
        record=read(directory/item['export']['file']);check_export(record,None,expected_source_sha256=item['source_sha256'])
        cert,ref=solve(key,record);checked=check_export(record,cert,expected_source_sha256=item['source_sha256'])
        item.update(status='PROPOSED' if cert else 'UNAVAILABLE',certificate=ref)
        bounds[key]=Fraction(checked['bound']['checked_lower_bound']) if cert else None
        save(directory/'manifest.json',m)
    gate=order_bounds(bounds['gate_lower'],bounds['gate_upper']);source=read(directory/m['joint_source']['file'])
    for row in m['obligations']:
        if row['kind']=='reused':continue
        lo,neg=bounds[row['difference_lower']],bounds[row['difference_upper']]
        if lo is None or neg is None:
            row.update(weighted_status='RANGE_UNAVAILABLE',weighted=None,certificate=None)
        else:
            if lo>-neg:raise ValueError('inconsistent disagreement bounds')
            diff=[str(lo),str(-neg)];gate_strings=[str(v) for v in gate];key=f"p{row['property_index']}_rational"
            record=build(source,row['q'],0,gate_strings,diff)
            check_construction(record,source_hash=identity(source),q=row['q'],offset=0,gate=gate_strings,difference=diff)
            name=key+'.export.json';save(directory/name,record);cert,ref=solve(key,record)
            check_construction(record,cert,source_hash=identity(source),q=row['q'],offset=0,gate=gate_strings,difference=diff)
            row.update(weighted_status='PROPOSED' if cert else 'UNAVAILABLE',weighted=reference(directory,name),certificate=ref,
                       gate_bounds=[str(v) for v in gate],difference_bounds=[str(v) for v in diff])
        save(directory/'manifest.json',m)
    save(directory/'proposal.json',{'manifest_sha256':sha(directory/'manifest.json'),'query_count':len(calls),
        'query_log_sha256':sha(directory/'query_log.json'),'proposal_and_local_checks_seconds':time.monotonic()-started})


def run():
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):raise ValueError('clean feature checkout required')
    f=verify_freeze();resource=resources();publication=publication_gate();DEFAULT.mkdir(exist_ok=False)
    directory=DEFAULT/f['protocol']['job_id'];directory.mkdir()
    save(directory/'job.json',{**f['job'],'protocol':f['protocol'],'freeze_sha256':sha(FREEZE)})
    rt={'schema':'CONV_PRE_F0_RATIONAL_R2_RUN','execution_head':git('rev-parse','HEAD'),'freeze_sha256':sha(FREEZE),
        'publication':publication,'started_unix':time.time(),'resource':resource,'state':'RUNNING','stages':{},'result':None,'extra_queries_queued':False}
    save(DEFAULT/'runtime.json',rt)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    for name,cap in [('capture',300),('proposal',2100),('check',600),('independent',600)]:
        args=([sys.executable,'-m','scripts.run_conv_pre_f0_r2','--'+name,str(directory)] if name in ('capture','proposal') else
              [sys.executable,'-S','-m','scripts.check_conv_pre_f0_r2',str(directory),str(directory/(name+'.json'))])
        rt['stages'][name]=subprocess_stage(args,directory,name,cap,env);save(DEFAULT/'runtime.json',rt)
        if rt['stages'][name]['state']!='COMPLETED':
            rt['state']=rt['stages'][name]['state'];save(DEFAULT/'runtime.json',rt);return
    if read(directory/'check.json')!=read(directory/'independent.json'):raise ValueError('checker disagreement')
    verify_freeze();rt.update(state='COMPLETED_CHECKED',result=read(directory/'check.json'));save(DEFAULT/'runtime.json',rt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--capture',type=Path);g.add_argument('--proposal',type=Path);a=p.parse_args()
    if a.freeze:
        if FREEZE.exists():raise FileExistsError('freeze immutable')
        save(FREEZE,{'protocol':parents()[0],'job':job(),'sources':sources()})
    elif a.capture:capture_worker(a.capture)
    elif a.proposal:proposal_worker(a.proposal)
    else:
        with (ROOT/'data/moe/results/conv_pre_f0_r2.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if DEFAULT.exists():raise FileExistsError('no resume/retry')
            try:run()
            except BaseException as exc:
                if (DEFAULT/'runtime.json').exists():
                    rt=read(DEFAULT/'runtime.json');rt.update(state='ERROR',error=repr(exc));save(DEFAULT/'runtime.json',rt)
                raise
