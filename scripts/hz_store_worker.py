"""Owned worker for frozen checked-store supervision controls."""
import argparse
import time

from scoped_proof.io import Events, load, save, sha, tick
from scoped_proof import device_lifecycle as engine
from scoped_source.hz_request_store import Store, identity
from scoped_source.check_store_endpoints import EndpointWork, plan, SCHEMA
from scripts import hz_store_supervised as policy


def run(root,inv_sha,plan_sha):
    inv,launch=engine.plan(policy,root,inv_sha,plan_sha)
    phase=launch['phase']; end=launch['run_deadline']; s=inv['spec']; control=s['control']
    events=Events(root,phase,inv['start']); tick(end)

    def fault(name):
        if control!=name: return
        events.emit('FAULT_REACHED',control=name,invocation=inv['invocation'])
        if name.endswith('_delay'):
            while time.monotonic()<end+1: time.sleep(.05)
            tick(end)
        if name.endswith('_exception'): raise RuntimeError('registered '+name)

    if phase=='admit':
        result=dict(status='CPU_NO_CUDA',invocation=inv['invocation'])
    elif phase=='release':
        stage=load(root/'produce_stage.json',launch['inputs']['produce'])
        if not policy.cleanup_confirmed(stage): raise ValueError('producer cleanup not confirmed')
        result=dict(status='CPU_NO_CUDA_TO_RELEASE',invocation=inv['invocation'],
                    producer_sha256=sha(root/'produce_stage.json'))
    elif phase=='produce':
        costs={}
        def timed(name,fn):
            begin=time.monotonic(); value=events.call(name,fn); costs[name]=time.monotonic()-begin; return value
        def imports():
            fault('import_delay')
            import torch
            torch.set_num_threads(1)
            from scoped_source.hz_store_source import build
            from act.back_end.moe.store_endpoints import produce
            return build,produce
        build,produce=timed('imports',imports)
        work=EndpointWork(end); data=root/'evidence'; data.mkdir(exist_ok=False)
        declaration=policy.fixture(s['case'])
        if identity(declaration)!=s['source']: raise ValueError('source recipe binding')
        save(data/'source.json',declaration)
        def source():
            fault('source_delay'); fault('producer_exception')
            return build(data/'store',declaration,work=work)[0]
        store_sha=timed('source',source)
        def prepare():
            store=Store(data/'store',expected_manifest=store_sha,expected_request=s['source'],work=work)
            contract=dict(schema=SCHEMA,store=store_sha,request=s['source'],
                          **{k:s[k] for k in ('mode','classes','label','margin','semantics','ties')})
            anchor=save(data/'contract.json',contract)['sha256']
            inventory=plan(store,contract,expected_contract=anchor,declaration=declaration,work=work)
            events.emit('SOURCE_READY',source=s['source'],store=store_sha,contract=anchor,
                        counts=work.counts,required_endpoints=len(inventory.tasks))
            return store,inventory,anchor
        store,inventory,anchor=timed('plan',prepare)
        def publish_event(name,i,ref):
            if name=='before_tile':
                if i==1: fault('serialization_delay')
            else: events.emit('TILE_PUBLISHED',index=i,reference=ref,counts=work.counts,
                              contract=anchor,required_endpoints=len(inventory.tasks))
        def candidates():
            fault('candidate_delay')
            missing=[inventory.tasks[-1].slot] if s['case']=='partial' else []
            return produce(data/'proof',store,inventory,tile_width=s['tile_width'],work=work,
                           missing_slots=missing,event=publish_event)
        proof,stats=timed('candidates',candidates)
        if control=='invalid_candidate':
            # Malformed producer output is never accepted; leave the original
            # append-only evidence untouched and corrupt only its receipt hash.
            fault('invalid_candidate'); proof='0'*64
        result=dict(invocation=inv['invocation'],spec_sha256=identity(s),contract=anchor,proof=proof,
                    counts=work.counts,costs=costs,candidate=stats)
    elif phase=='check':
        def checked():
            fault('check_delay'); fault('checker_exception')
            return policy.check(root,inv,launch['inputs']['produce'],end)
        result=events.call('independent_exact_check',checked)
        if control=='missing_check':
            fault('missing_check'); events.emit('WORKER_COMPLETE',phase=phase); return
    else:
        def receive():
            fault('receive_delay'); return policy.receive(root,inv,launch['inputs'])
        result=events.call('reception',receive)
    tick(end)
    events.call('final_serialization',lambda:save(root/(phase+'.json'),result))
    tick(end); events.emit('WORKER_COMPLETE',phase=phase)


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path)
    p.add_argument('--invocation-sha',required=True); p.add_argument('--plan-sha',required=True)
    a=p.parse_args(); run(a.root,a.invocation_sha,a.plan_sha)
