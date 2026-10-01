"""Owned worker, new whole-source path; numerical imports only in produce/check."""
import argparse
from pathlib import Path
import time

from scoped_proof.io import Events, load, save, tick
from scoped_proof import device_lifecycle as engine
from source_enclosure.format import identity
from scripts import hz_source_supervised as policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--invocation-sha',required=True)
    parser.add_argument('--plan-sha',required=True)
    args = parser.parse_args(); root=args.root
    inv,plan = engine.plan(policy,root,args.invocation_sha,args.plan_sha)
    phase,end,s = plan['phase'],plan['run_deadline'],inv['spec']
    events = Events(root,phase,inv['start'])

    def fault(name, delay=False):
        if s['control'] != name: return False
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
        costs=dict(create=0.,source=0.,prepare=0.,propose=0.)
        def measured(group,name,fn):
            begin=time.monotonic(); result=events.call(name,fn)
            costs[group]+=time.monotonic()-begin; tick(end); return result
        def create():
            fault('creation_delay',True)
            from scoped_source.endpoint_source_controls import cases
            return next(doc for name,doc,_ in cases() if name==s['source_case'])
        doc=measured('create','create_source_and_imports',create)
        if identity(doc)!=s['source_sha256']: raise ValueError('created declaration identity')
        source_ref=events.call('source_serialization',lambda:save(root/'prefix_source.json',doc))
        if fault('wrong_source'): doc['request']['margin']='1'
        def numerical_imports():
            from scoped_source.graph import validate,clock
            from source_enclosure.produce import box
            from scoped_source.hz_source_build import propagate,entry_for,gate,live
            from scoped_source.hz_source_check import properties,SCHEMA
            from act.back_end.moe.hz_endpoints import prepare_request
            from act.back_end.moe.batched_support import propose_batch
            return validate,clock,box,propagate,entry_for,gate,live,properties,SCHEMA,prepare_request,propose_batch
        (validate,clock,box,propagate,entry_for,gate,live,properties,SCHEMA,
         prepare_request,propose_batch)=measured('create','numerical_imports',numerical_imports)

        def input_enclosure():
            r,lo,hi=validate(doc,s['source_sha256'],clock(end))
            return r,box(lo,hi)
        r,initial=measured('source','checked_input',input_enclosure)
        allpairs=policy.roster(doc)
        lowering={'schema':'HZ_LOWERING_PREFIX_V1','source_sha256':identity(doc),'input':initial,
                  'router':None,'pairs':[],'pending_pairs':allpairs[:]}
        source_refs={}
        def publish_source():
            name=f'lowering_{len(source_refs):02}.json'
            source_refs[name]=events.call('lowering_serialization',lambda:save(root/name,lowering))['sha256']; tick(end)
        publish_source()
        router,rt=measured('source','router_propagation',lambda:propagate(doc,'router',initial,'router',end))
        lowering['router']=rt; publish_source()
        live_pairs=[]
        for pair in allpairs:
            a,b=pair
            def entry():
                if pair==allpairs[0]:
                    fault('upstream_delay',True)
                    if fault('upstream_exception'): raise RuntimeError('registered upstream error')
                return entry_for(initial,router,pair,r['experts'])
            start=measured('source',f'entry_{a}_{b}',entry)
            left,lt=measured('source',f'expert_{a}_{b}_a',lambda:propagate(doc,f'expert{a}',start,f'pair{a}-{b}/expert{a}',end))
            right,bt=measured('source',f'expert_{a}_{b}_b',lambda:propagate(doc,f'expert{b}',start,f'pair{a}-{b}/expert{b}',end))
            evidence=measured('source',f'gate_{a}_{b}',lambda:gate(router,pair))
            lowering['pairs'].append({'pair':pair,'entry':start,'a':lt,'b':bt,'gate_evidence':evidence})
            lowering['pending_pairs']=allpairs[len(lowering['pairs']):]
            def converted(): return {'pair':pair,'entry':live(start),'a':live(left),'b':live(right),'gate':evidence['bounds']}
            live_pairs.append(measured('source',f'live_pair_{a}_{b}',converted))
            publish_source()
        context={'request':identity(doc),'domain':identity(r),'guard':'ALL_TIE_LEGAL_PAIRS'}
        request=measured('prepare','prepare_endpoints',lambda:prepare_request(live_pairs,properties(r),
            experts=r['experts'],classes=r['classes'],context=context,deadline=end))
        proof={'schema':'GUARDED_HZ_ENDPOINT_PROOF_V1','request_sha256':identity(request),
               'pairs':[{'pair':p['pair'],'candidates':None} for p in request['pairs']]}
        package={'schema':SCHEMA,'source_sha256':identity(doc),'input':initial,'router':rt,
                 'pairs':lowering['pairs'],'endpoint_request':request,'proof':proof}
        package_refs={}
        def publish_package():
            name=f'endpoint_{len(package_refs):02}.json'
            package_refs[name]=events.call('endpoint_serialization',lambda:save(root/name,package))['sha256']; tick(end)
        publish_package()
        for i,pair in enumerate(request['pairs']):
            tick(end)
            if s['partial'] and i==len(request['pairs'])-1: break
            def propose():
                if fault('proposal_exception'): raise RuntimeError('registered proposal error')
                return propose_batch(pair['batch'],expected_batch_sha256=identity(pair['batch']),deadline=end)
            candidates=measured('propose',f'propose_pair_{i}',propose)
            if i==0 and fault('missing_endpoint'): candidates['entries'].pop()
            proof['pairs'][i]['candidates']=candidates; publish_package()
            if i==0: fault('partial_delay',True)
        value={'invocation':inv['invocation'],'spec_sha256':identity(s),'source_file_sha256':source_ref['sha256'],
               'source_prefixes':source_refs,'package_prefixes':package_refs,'package':package,'cost_seconds':costs}
        fault('serialization_delay',True)
    elif phase=='check':
        fault('check_delay',True)
        if fault('check_exception'): raise RuntimeError('registered checker error')
        value=events.call('independent_source_and_endpoint_check',lambda:policy.check(root,inv,plan['inputs']['produce'],end))
        if fault('missing_check'): return
    else:
        fault('receive_delay',True)
        value=events.call('receive_binding',lambda:policy.receive(root,inv,plan['inputs']))
    tick(end)
    events.call('final_serialization',lambda:save(root/(phase+'.json'),value))
    tick(end); events.emit('WORKER_COMPLETE',phase=phase)


if __name__=='__main__': main()
