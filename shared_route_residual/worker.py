"""Fresh no-site router-only worker. No native solver, real model or outputs."""
import argparse
from pathlib import Path
import sys

from scoped_proof.io import Events, load, save, tick
from source_enclosure.format import identity


def recheck(root, plan, deadline):
    doc,pre,candidates=(load(root/name) for name in ('source.json','prefix.json','candidates.json'))
    if identity(doc)!=plan['expected_source_sha256']:
        raise ValueError('frozen synthetic source binding')
    if plan['mode']=='pairwise':
        from checked_route_frontier.check import check_frontier
        result=check_frontier(doc,pre,candidates,expected_source_sha256=identity(doc),deadline=deadline)
    elif plan['mode']=='shared':
        from shared_route_residual.check import check
        result=check(doc,pre,candidates,expected_source_sha256=identity(doc),invocation=plan['invocation'],deadline=deadline)
    else:raise ValueError('mode')
    tick(deadline)
    return {'schema':'ROUTER_RESIDUAL_RECEIPT_V1','invocation':plan['invocation'],
        'mode':plan['mode'],'source_sha256':identity(doc),'prefix_sha256':identity(pre),
        'candidate_sha256':identity(candidates),'result':result,
        'complete_output_positive_proof':False,'native_float_proof':False,'new_real_positive':False}


def work(phase,root,deadline):
    plan=load(root/'plan.json');events=Events(root,phase,plan['started_monotonic'])
    if plan['mode'] not in ('pairwise','shared'):raise ValueError('mode')
    if phase=='build':
        from shared_route_residual.fixtures import fixture
        from checked_route_frontier.build import prefix
        doc=events.call('synthetic_source',lambda:fixture(plan['fixture']))
        if identity(doc)!=plan['expected_source_sha256']:raise ValueError('source')
        events.call('serialize_source',lambda:save(root/'source.json',doc))
        pre=events.call('router_source',lambda:prefix(doc,expected_source_sha256=identity(doc),deadline=deadline))
        events.call('serialize_prefix',lambda:save(root/'prefix.json',pre))
        if plan['mode']=='pairwise':
            from checked_route_frontier.build import propose_final_affine
            candidates=events.call('fixed_final_affine_candidates',lambda:propose_final_affine(doc,pre,deadline=deadline))
        else:
            from shared_route_residual.propose import propose
            candidates=events.call('fixed_shared_equality_potentials',lambda:propose(doc,pre,invocation=plan['invocation'],deadline=deadline))
        events.call('serialize_candidates',lambda:save(root/'candidates.json',candidates))
    elif phase=='check':
        result=events.call('fresh_load_source_and_router_check',lambda:recheck(root,plan,deadline))
        events.call('serialize_check',lambda:save(root/'check.json',result))
    else:raise ValueError('phase')
    tick(deadline)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('build','check'));p.add_argument('root',type=Path)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    if not sys.flags.no_site:raise ValueError('python -S required')
    def forbid(event,args):
        if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','act','highspy') or
                (a.phase=='check' and args[0] in ('checked_route_frontier.build','shared_route_residual.propose'))):
            raise ImportError('independent synthetic checker')
        if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.fork','os.exec'):
            raise PermissionError('no nested executor')
    sys.addaudithook(forbid);work(a.phase,a.root.resolve(),a.deadline)
