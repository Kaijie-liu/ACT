"""Owned CPU-only endpoint workers; creation and every query share one clock."""
import argparse
from pathlib import Path
import time

from scoped_proof.io import Events, load, save, sha, tick
from scoped_proof import device_lifecycle as engine
from scripts import hz_endpoint_supervised as policy
from source_enclosure.format import identity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--invocation-sha',required=True)
    parser.add_argument('--plan-sha',required=True)
    args=parser.parse_args(); root=args.root
    inv,plan=engine.plan(policy,root,args.invocation_sha,args.plan_sha)
    phase,end,s=plan['phase'],plan['run_deadline'],inv['spec']
    events=Events(root,phase,inv['start'])

    def fault(name,delay=False):
        if s['control']!=name: return False
        events.emit('FAULT_REACHED',fault=name,invocation=inv['invocation'])
        if delay: time.sleep(max(0.,end-time.monotonic())+1)
        return True

    if phase=='admit':
        value={'status':'CPU_NO_CUDA','invocation':inv['invocation']}
    elif phase=='release':
        p=load(root/'produce_stage.json',plan['inputs']['produce'])
        if not policy.cleanup_confirmed(p): raise ValueError('producer cleanup incomplete')
        value={'status':'CPU_NO_CUDA_TO_RELEASE','invocation':inv['invocation'],
               'producer_sha256':plan['inputs']['produce']}
    elif phase=='produce':
        fault('creation_delay',True)
        costs={}
        def measured(name,fn):
            begin=time.monotonic(); result=events.call(name,fn)
            costs[name]=time.monotonic()-begin; return result
        def create():
            from act.back_end.moe.test_hz_endpoints import separation,relation
            return separation() if s['case'] in ('separation','partial') else relation()
        pairs,props,e,c=measured('create',create)
        from act.back_end.moe.hz_endpoints import prepare_request
        from act.back_end.moe.batched_support import propose_batch
        from act.back_end.moe.test_hz_endpoints import CONTEXT
        request=measured('prepare',lambda:prepare_request(pairs,props,experts=e,classes=c,context=CONTEXT,
            deadline=end,relation_mode='independent_inputs' if s['case']=='independent' else 'shared_input'))
        policy.validate_basis(request,s)
        ref=events.call('request_serialization',lambda:save(root/'prefix_request.json',request))
        proof={'schema':'GUARDED_HZ_ENDPOINT_PROOF_V1','request_sha256':identity(request),
               'pairs':[{'pair':p['pair'],'candidates':None} for p in request['pairs']]}
        prefixes={}
        def publish_prefix():
            name=f'prefix_{len(prefixes):02}.json'
            prefixes[name]=events.call('prefix_serialization',lambda:save(root/name,proof))['sha256']
        publish_prefix(); costs['propose']=0.
        for i,pair in enumerate(request['pairs']):
            tick(end)
            if s['case']=='partial' and i==len(request['pairs'])-1: break
            if fault('proposal_exception'): raise RuntimeError('registered proposal exception')
            begin=time.monotonic()
            candidates=events.call('propose_pair_'+str(i),lambda:propose_batch(
                pair['batch'],expected_batch_sha256=identity(pair['batch']),deadline=end))
            costs['propose']+=time.monotonic()-begin
            proof['pairs'][i]['candidates']=candidates; publish_prefix()
            if i==0: fault('partial_delay',True)
        if fault('wrong_source'): request['properties'][0]['q'][0]='2'
        if fault('missing_endpoint'): proof['pairs'][0]['candidates']['entries'].pop()
        value={'invocation':inv['invocation'],'spec_sha256':identity(s),'request':request,'proof':proof,
               'request_file_sha256':ref['sha256'],'prefixes':prefixes,'cost_seconds':costs}
        fault('serialization_delay',True)
    elif phase=='check':
        fault('check_delay',True)
        if fault('check_exception'): raise RuntimeError('registered checker exception')
        value=events.call('independent_exact_check',lambda:policy.check(root,inv,plan['inputs']['produce'],end))
        if fault('missing_check'): return
    else:
        fault('receive_delay',True)
        value=events.call('receive_binding',lambda:policy.receive(root,inv,plan['inputs']))
    tick(end)
    events.call('final_serialization',lambda:save(root/(phase+'.json'),value))
    tick(end); events.emit('WORKER_COMPLETE',phase=phase)


if __name__=='__main__': main()
