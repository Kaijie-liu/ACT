"""Generate/propagate/construct/publish/check under one inherited clock.

Mathematical functions and R1 serializer are unchanged. Instrumentation uses
private function globals, not persistent patches to any frozen module. Cache
exists only while checking this newly constructed source; no prior archive is
read by the worker. All phases and stats persistence are in the same budget.
"""
import argparse
from pathlib import Path
import time

from bounded_evidence.stream import save as publish
from scoped_proof.io import load,save,sha,tick
from source_enclosure.format import identity
from source_cost_supervised.worker import Journal
from parsed_source_reuse.check import clone,private_checker
from readonly_upstream.execution import validate_method


def profile(doc, root, inv, method, journal):
    from checked_route_frontier import build as router_build
    from shared_route_residual.propose import propose
    from shared_route_residual.format import binding
    from residual_proof import build
    from readonly_source.check import check as readonly_check
    deadline=inv['work_deadline_monotonic']; tick(deadline)
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); phases=[]; operations=[]; phase_name=None

    def phase(name, fn):
        nonlocal phase_name
        phase_name=name; begin=time.monotonic(); tick(deadline)
        value=journal.call('phase',name,fn); tick(deadline)
        seconds=time.monotonic()-begin
        parts=sum(x['seconds'] for x in operations if x['phase']==name)
        if parts>seconds: raise ValueError('overlapping component accounting')
        phases.append({'name':name,'seconds':seconds,'component_seconds':parts,'other_seconds':seconds-parts})
        return value

    def measured(fn, role):
        def call(*args,**kwargs):
            began=time.monotonic(); status='ERROR'
            try:
                result=journal.call('component',role,lambda:fn(*args,**kwargs))
                status='COMPLETED'; return result
            finally:
                operations.append({'phase':phase_name,'role':role,'status':status,'seconds':time.monotonic()-began})
        return call

    prefix=clone(router_build.prefix,{'network':measured(router_build.network,'router_propagation')})
    construct=clone(build.finish,{name:measured(getattr(build,name),role) for name,role in (
        ('network','expert_propagation'),('join','factor_join'),('guards','guards'),
        ('project','projection'),('output_lp','weighted_lp'))})
    def checked_functions(parser,scope):
        fn=private_checker(parser,scope)
        return clone(fn,{name:measured(fn.__globals__[name],role) for name,role in (
            ('route_check','route_recheck'),('check_network','expert_check'),('check_join','join_check'),
            ('check_guards','guard_check'),('check_projection','projection_check'),('check_outputs','weighted_lp_check'))})
    check=clone(readonly_check,{'private_checker':checked_functions})
    digest=phase('identity',lambda:identity(doc))
    pre=phase('prefix',lambda:prefix(doc,expected_source_sha256=digest,deadline=deadline))
    # Preserve the original construction token/bytes for paired differential;
    # the cache uses the fresh supervisor invocation, not this historical token.
    certificate=phase('proposal',lambda:propose(doc,pre,invocation='profile_v2',deadline=deadline))
    if type(certificate) is not dict or certificate.get('binding')!=binding(doc,pre,'profile_v2'):
        raise ValueError('bound shared certificate dictionary required')
    bundle=phase('construct',lambda:construct(doc,pre,certificate,mode='shared',invocation='profile_v2',
        expected_source_sha256=digest,deadline=deadline))
    files=phase('publish_R1',lambda:{'source':publish(root/'source.json',doc,deadline=deadline),
        'construction':publish(root/'construction.json',bundle,deadline=deadline)})
    stats_record=[]
    def retain(stats):
        stats_record.append(save(root/'parser_stats.json',{'invocation':inv['invocation'],
            'spec_sha256':inv['spec_sha256'],'method_sha256':identity(method),'stats':stats}))
    checked=phase('source_check',lambda:check(doc,bundle,invocation='profile_v2',
        expected_source_sha256=digest,deadline=deadline,enabled=method['readonly'],stats_sink=retain,
        cache_invocation=inv['invocation']))
    report={'schema':'SOURCE_COST_DIAGNOSTIC_V2','source_sha256':digest,'files':files,
        'phases':phases,'operations':operations,'source_check':checked,'complete_output_positive_proof':False,
        'native_solver_calls':0,'seconds_before_report':time.monotonic()-start,
        'upstream_method':{'schema':'READONLY_UPSTREAM_REPORT_R1','invocation':inv['invocation'],
            'spec_sha256':inv['spec_sha256'],'method_sha256':identity(method),'readonly':method['readonly'],
            'parser_stats_record':stats_record[0]}}
    save(root/'report.json',report); tick(deadline)
    return report


def run(root, expected):
    root=Path(root); inv=load(root/'invocation.json',limit=65536)
    spec=load(root/'spec.json',inv['spec_sha256'],limit=65536)
    method=load(root/'method.json',expected,limit=65536); validate_method(method)
    deadline=inv['work_deadline_monotonic']; tick(deadline)
    from source_cost_supervised.audit import validate_spec
    from source_construction_lab.fixtures import document
    validate_spec(spec); journal=Journal(root,inv)
    doc=journal.call('worker','generate',lambda:document(**spec['fixture']))
    if identity(doc)!=spec['source_sha256']: raise ValueError('generated source identity')
    journal.call('worker','profile',lambda:profile(doc,root/'profile',inv,method,journal))
    tick(deadline); records={}
    for name in ('source.json','construction.json','report.json'):
        path=root/'profile'/name; records[name]={'sha256':sha(path),'bytes':path.stat().st_size}; tick(deadline)
    records['journal.jsonl']={'sha256':sha(journal.path),'bytes':journal.path.stat().st_size}
    save(root/'candidate.json',{'schema':'SOURCE_COST_CANDIDATE_V1','invocation':inv['invocation'],
        'spec_sha256':inv['spec_sha256'],'source_sha256':spec['source_sha256'],'records':records,
        'method_sha256':expected,'complete_output_positive_proof':False,'native_solver_calls':0})
    tick(deadline)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('--method-sha256',required=True)
    a=p.parse_args(); run(a.root,a.method_sha256)
