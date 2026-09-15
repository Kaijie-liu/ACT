"""One-shot, all-obligation supplied-HZ proof experiment, not a verifier gate."""
import argparse
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import sys
import time

from scripts.conv_request_sign_lp_contract import (ROOT,DEFAULT,FREEZE,read,save,sha,git,
    parents,jobs,sources,verify_freeze,validate_job,publication_gate)
from scripts.run_conv_sign_lp import subprocess_stage,resources
from scripts.check_conv_request_sign_lp import expected_scope,property_vector,TRUSTED


class CapturedRequest(BaseException):
    """Construction complete; deliberately no property MILP or SAFE return."""


def reference(directory,name):
    return {'file':name,'sha256':sha(directory/name)}


def capture_worker(directory):
    from scripts.budget_contract_v2 import verify_v2
    from act.pipeline.moe.external_pair_worker import load
    from act.back_end.solver.hz_lp_export import export
    import act.pipeline.moe.paired_monolithic as mono
    job=validate_job(directory); req=job['parent_request']; started=time.monotonic()
    if (directory/'manifest.json').exists():raise FileExistsError('one shot')
    model,tensors=load(req); original=mono._run_monolithic
    manifest={}
    def construct(*,model,center,clean_prediction,internal,config,reuse=None,budget=None):
        routes=asdict(internal['route_sets']); pairs=[tuple(p) for p in routes['feasible']]
        properties=mono.linear_safety_rows(internal['output_spec'],internal['program'].output_width)
        if len(properties)!=9 or [list(p) for p in pairs]!=job['case']['expected_pairs']:
            raise ValueError('frozen request routes/properties differ')
        rows=[]
        for pair in pairs:
            for i,(q,constant) in enumerate(properties):
                if list(q)!=property_vector(clean_prediction,job['competitors'][i]) or constant!=0:
                    raise ValueError('property mismatch')
                rows.append({'pair':list(pair),'property_index':i,'competitor':job['competitors'][i],
                             'q':list(q),'constant':constant,'kind':'pending'})
        manifest.update(schema='CONV_REQUEST_SIGN_LP_R1_MANIFEST',request=expected_scope(job),
            trusted_base=TRUSTED,positive_threshold=job['protocol']['positive_threshold'],
            routes=routes,common_facts=reference(directory,'common_facts.json'),
            generation_complete=False,obligations=rows)
        save(directory/'manifest.json',manifest)
        if not routes['exact'] or routes['unresolved']:raise CapturedRequest()
        support,solver=config['support'],config['solver']
        hz_config=mono.HybridZConfig(max_input_dim=1024,guarded_support_enabled=True,
            guarded_support_lp_neurons=int(support['lp_neurons']),
            guarded_support_milp_neurons=int(support['milp_neurons']),
            guarded_support_lp_time_limit=float(support['lp_time_limit']),
            guarded_support_milp_time_limit=float(support['milp_time_limit']),
            guarded_support_solver_backend=str(support.get('solver_backend','scipy')),
            expert_property_solver_backend=str(solver.get('backend','scipy')))
        for row in rows:
            proof=mono.reuse_property(reuse['facts'],reuse['scope'],tuple(row['pair']),row['property_index']) if reuse else None
            if proof is not None:row.update(kind='reused',proof=proof)
        save(directory/'manifest.json',manifest)
        router,program=internal['router'],internal['program']; propagated_pairs={}
        for pair in pairs:
            if all(r['kind']=='reused' for r in rows if r['pair']==list(pair)):continue
            budget.check('monolithic_pair_propagation')
            hz_config.guarded_support_lp_time_limit=budget.limit('monolithic_lp_support',float(support['lp_time_limit']))
            hz_config.guarded_support_milp_time_limit=budget.limit('monolithic_mip_support',float(support['milp_time_limit']))
            conditioned=mono.condition_topk_set(router.output_hz,pair).hz
            entry=mono.guarded_input_topk_set(router.input_hz,router.output_hz,pair).hz
            propagated=mono.shared_input_pair_propagation(program.experts[pair[0]],program.experts[pair[1]],
                entry_hz=entry,hybridz_config=hz_config,expert_relation=config.get('expert_relation','shared_input'))
            gates=mono.compute_weighted_top2_gate_range(conditioned,pair,
                time_limit=budget.limit('monolithic_margin',float(solver['margin_support_seconds'])))
            budget.check('monolithic_pair_complete');propagated_pairs[pair]=(conditioned,propagated,gates)
        for row in rows:
            if row['kind']=='reused':continue
            budget.check('monolithic_property');pair=tuple(row['pair'])
            conditioned,propagated,gates=propagated_pairs[pair]
            encoding=mono.build_weighted_top2_f0(propagated.joint,conditioned,pair,row['q'],row['constant'],
                difference_time_limit=budget.limit('monolithic_difference',float(solver['difference_support_seconds'])),gate_range=gates)
            record=export(encoding.output_hz,[1],sparse=True)
            name=f"pair{pair[0]}_{pair[1]}_p{row['property_index']}.export.json"
            save(directory/name,record)
            row.update(kind='lp',export=reference(directory,name),source_sha256=record['source_sha256'],
                       proposal_status='PENDING',certificate=None)
            save(directory/'manifest.json',manifest)
        budget.check('monolithic_property_complete');manifest['generation_complete']=True
        save(directory/'manifest.json',manifest);raise CapturedRequest()
    mono._run_monolithic=construct
    try:
        try:
            value=verify_v2(model,tensors['center'],req['epsilon'],read(req['config']['path']),
                journal_path=directory/'budget_journal.jsonl',started=started,
                identity={'request_proof_job_sha256':sha(directory/'job.json')},
                expected_clean_prediction=req['sample']['label'],
                checkpoint_identity={'path':req['subject']['checkpoint'],'sha256':req['subject']['checkpoint_sha256']},
                common_fact_callback=lambda v:save(directory/'common_facts.json',v))
        except CapturedRequest:pass
        else:
            if not manifest:raise ValueError('front-end did not reach registered obligations')
            save(directory/'generation_return.json',value)
    finally:mono._run_monolithic=original
    save(directory/'generation.json',manifest)
    save(directory/'capture.json',{'generation_sha256':sha(directory/'generation.json'),
        'journal_sha256':sha(directory/'budget_journal.jsonl'),'generation_complete':manifest['generation_complete'],
        'capture_seconds':time.monotonic()-started,'weighted_property_milp_called':False,
        'recipe':'fresh V2 front-end and monolithic construction prefix; all properties; no property solves',
        'production_verdict':None})


def proposal_worker(directory):
    from act.back_end.solver.lp_certificate import propose
    job=validate_job(directory);m=read(directory/'manifest.json');started=time.monotonic()
    if (directory/'proposal.json').exists():raise FileExistsError('one shot')
    for r in m['obligations']:
        if r['kind']!='lp':continue
        if r['proposal_status']!='PENDING':raise ValueError('no resume')
        t=time.monotonic()
        try:cert=propose(read(directory/r['export']['file'])['lp'],time_limit=job['protocol']['proposal_solver_seconds_per_lp'])
        except ValueError as exc:
            if str(exc)!='proposal solver did not complete':raise
            r.update(proposal_status='UNAVAILABLE',proposal_reason=str(exc))
        else:
            name=r['export']['file'].replace('.export.json','.certificate.json');save(directory/name,cert)
            r.update(proposal_status='PROPOSED',certificate=reference(directory,name))
        r['proposal_seconds']=time.monotonic()-t;save(directory/'manifest.json',m)
    save(directory/'proposal.json',{'manifest_sha256':sha(directory/'manifest.json'),
         'proposal_and_local_checks_seconds':time.monotonic()-started})


def run():
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):
        raise ValueError('clean feature checkout required')
    freeze=verify_freeze();resource=resources();publication=publication_gate()
    DEFAULT.mkdir(exist_ok=False)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    rt={'schema':'CONV_REQUEST_SIGN_LP_R1_RUN','execution_head':git('rev-parse','HEAD'),
        'freeze_sha256':sha(FREEZE),'publication':publication,'started_unix':time.time(),
        'resource_at_start':resource,'cases':[],'unattempted':[j['case'] for j in freeze['jobs']],
        'state':'RUNNING','extra_queries_queued':False}
    save(DEFAULT/'runtime.json',rt)
    for job in freeze['jobs']:
        if sources()!=freeze['sources'] or git('status','--porcelain'):raise ValueError('source drift')
        directory=DEFAULT/job['case']['job_id'];directory.mkdir()
        save(directory/'job.json',{**job,'protocol':freeze['protocol'],'freeze_sha256':sha(FREEZE)})
        row={'case':job['case'],'stages':{},'result':None};rt['cases'].append(row);rt['unattempted'].pop(0)
        save(DEFAULT/'runtime.json',rt)
        for name,seconds in [('capture',freeze['protocol']['capture_seconds']),('proposal',freeze['protocol']['proposal_outer_seconds']),
                             ('check',freeze['protocol']['checker_seconds']),('independent',freeze['protocol']['checker_seconds'])]:
            args=([sys.executable,'-m','scripts.run_conv_request_sign_lp','--'+name,str(directory)] if name in ('capture','proposal') else
                  [sys.executable,'-S','-m','scripts.check_conv_request_sign_lp',str(directory),str(directory/(name+'.json'))])
            row['stages'][name]=subprocess_stage(args,directory,name,seconds,env);save(DEFAULT/'runtime.json',rt)
            if row['stages'][name]['state']!='COMPLETED':
                rt['state']=row['stages'][name]['state'];save(DEFAULT/'runtime.json',rt);return
        if read(directory/'check.json')!=read(directory/'independent.json'):raise ValueError('checks disagree')
        row['result']=read(directory/'check.json');save(DEFAULT/'runtime.json',rt)
    parents();rt['state']='COMPLETED_CHECKED';save(DEFAULT/'runtime.json',rt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true')
    g.add_argument('--capture',type=Path);g.add_argument('--proposal',type=Path);a=p.parse_args()
    if a.freeze:
        if FREEZE.exists():raise FileExistsError('freeze immutable')
        save(FREEZE,{'protocol':parents()[0],'jobs':jobs(),'sources':sources(),
                     'scope':'All required properties on two observed requests, supplied-F0-HZ trust contract'})
    elif a.capture:capture_worker(a.capture)
    elif a.proposal:proposal_worker(a.proposal)
    else:
        with (ROOT/'data/moe/results/conv_request_sign_lp_r1.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if DEFAULT.exists():raise FileExistsError('no resume/retry')
            try:run()
            except BaseException as exc:
                if (DEFAULT/'runtime.json').exists():
                    rt=read(DEFAULT/'runtime.json');rt.update(state='ERROR',error=repr(exc));save(DEFAULT/'runtime.json',rt)
                raise
