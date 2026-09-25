"""Separate, optional saved-evidence checking segment under the existing clock.

Reuses the tested supervisor (300s total,2s reserve,8GiB,2threads), not the
old diagnostic output directory. Source generation/solving are NOT performed;
do not call these numbers end-to-end MoE verification latency.
"""
import argparse
from pathlib import Path
import time

from scoped_proof.io import PYTHON,load,save,sha,tick
from source_enclosure.format import identity
from source_cost_supervised.supervisor import supervise as watched


def validate_method(method,spec):
    if (set(method)!={'schema','enabled','source','construction','expected_result_sha256'}
            or method['schema']!='PARSED_SOURCE_SAVED_CHECK_R1' or type(method['enabled']) is not bool):
        raise ValueError('explicit bound parse-check method')
    for name in ('source','construction'):
        record=method[name]
        if (set(record)!={'path','sha256','bytes'} or type(record['bytes']) is not int or record['bytes']<=0
                or not Path(record['path']).is_absolute() or not Path(record['path']).is_relative_to('/data1/Kane/MOE')
                or Path(record['path']).is_symlink()):raise ValueError('bound local saved source')
    if method['source']['sha256']!=spec['source_sha256']:raise ValueError('source/spec mismatch')


def supervise(root,spec,method,*,budget=300.,rss_limit=8*2**30,command_hook=None):
    validate_method(method,spec)
    def command(phase,folder,deadline):
        if phase=='profile':save(folder/'method.json',method)
        args=[PYTHON,'-S','-m','parsed_source_reuse.execution',phase,str(folder),
              '--method-sha256',identity(method)]
        return command_hook(phase,folder,deadline,args) if command_hook else args
    return watched(root,spec,budget=budget,rss_limit=rss_limit,command_factory=command)


def context(root,expected):
    inv=load(root/'invocation.json',limit=65536);spec=load(root/'spec.json',inv['spec_sha256'],limit=65536)
    method=load(root/'method.json',expected,limit=65536);validate_method(method,spec)
    tick(inv['work_deadline_monotonic']);return inv,spec,method


def work(root,expected):
    from parsed_source_reuse.check import check
    began=time.monotonic();inv,spec,method=context(root,expected)
    deadline=inv['work_deadline_monotonic'];objects={};reading=time.monotonic()
    for name in ('source','construction'):
        r=method[name];path=Path(r['path'])
        if path.stat().st_size!=r['bytes']:raise ValueError('saved file size')
        objects[name]=load(path,r['sha256']);tick(deadline)
    read_seconds=time.monotonic()-reading;checking=time.monotonic();stats=[]
    def retain(value):
        stats.append(value);save(root/'parse_stats.json',{'invocation':inv['invocation'],
            'method_sha256':expected,'stats':value})
    # Bundle's historical token belongs to its checked route certificate.
    # The new cache token is the current supervised invocation, not that
    # historical certificate token. No entry is imported from another call.
    result=check(objects['source'],objects['construction'],invocation=objects['construction']['invocation'],
                 expected_source_sha256=spec['source_sha256'],deadline=deadline,
                 enabled=method['enabled'],stats_sink=retain,cache_invocation=inv['invocation'])
    check_seconds=time.monotonic()-checking
    if identity(result)!=method['expected_result_sha256']:raise ValueError('no-cache/source-check differential')
    save(root/'candidate.json',{'schema':'PARSED_SOURCE_CHECK_CANDIDATE_R1','invocation':inv['invocation'],
        'spec_sha256':inv['spec_sha256'],'method_sha256':expected,'result':result,
        'stats_record':{'sha256':sha(root/'parse_stats.json'),'bytes':(root/'parse_stats.json').stat().st_size},
        'read_seconds':read_seconds,'check_seconds':check_seconds,
        'seconds_before_candidate':time.monotonic()-began,'complete_output_positive_proof':False})
    tick(deadline)


def receive(root,expected,*,deadline=None):
    from source_cost_supervised.audit import finite
    # Offline saved audit uses a fresh, separately charged audit clock; it
    # cannot retroactively promote a timed-out online check.
    inv=load(root/'invocation.json');spec=load(root/'spec.json',inv['spec_sha256'])
    method=load(root/'method.json',expected);validate_method(method,spec)
    deadline=inv['work_deadline_monotonic'] if deadline is None else deadline;tick(deadline)
    c=load(root/'candidate.json',limit=2**20);r=c['stats_record'];path=root/'parse_stats.json'
    if path.stat().st_size!=r['bytes']:raise ValueError('stats bytes')
    stats=load(path,r['sha256'],limit=2**20)
    if (c['schema']!='PARSED_SOURCE_CHECK_CANDIDATE_R1' or c['invocation']!=inv['invocation']
            or c['spec_sha256']!=inv['spec_sha256'] or c['method_sha256']!=expected
            or stats['invocation']!=inv['invocation'] or stats['method_sha256']!=expected
            or identity(c['result'])!=method['expected_result_sha256']
            or c['complete_output_positive_proof'] is not False):raise ValueError('bound complete result')
    s=stats['stats'];p=s['parser'];timing=p['seconds']
    if (s['status']!='COMPLETED' or s['complete_output_positive_proof'] is not False or s['native_solver_queries']!=0
            or p['enabled']!=method['enabled'] or p['results_or_bounds_cached'] is not False
            or p['scope']!=[inv['invocation'],spec['source_sha256']]
            or p['failures']!=0 or p['lookups']!=p['hits']+p['parses']
            or (not p['enabled'] and (p['hits'] or p['live_entries']))
            or not s['after_close']['closed'] or s['after_close']['live_entries']!=0
            or not all(finite(v) for v in timing.values()) or not finite(p['other_seconds'])
            or abs(sum(v for k,v in timing.items() if k!='total')+p['other_seconds']-timing['total'])>1e-8
            or not all(finite(c[k]) for k in ('read_seconds','check_seconds','seconds_before_candidate'))
            or timing['total']>c['check_seconds'] or s['seconds_before_stats']>c['check_seconds']
            or c['read_seconds']+c['check_seconds']>c['seconds_before_candidate']):
        raise ValueError('parse cost/count/closure policy')
    tick(deadline)
    return {'status':'PROFILE_COMPLETE_NOT_OUTPUT_PROOF','invocation':inv['invocation'],
            'spec_sha256':inv['spec_sha256'],'method_sha256':expected,'candidate_sha256':sha(root/'candidate.json'),
            'complete_output_positive_proof':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('profile','receive'));p.add_argument('root',type=Path)
    p.add_argument('--method-sha256',required=True);a=p.parse_args()
    if a.phase=='profile':work(a.root,a.method_sha256)
    else:
        out=receive(a.root,a.method_sha256);save(a.root/'received.json',out)
        tick(load(a.root/'invocation.json')['work_deadline_monotonic'])
