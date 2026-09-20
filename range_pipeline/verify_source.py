"""Complete regenerated source with scoped ranges; NOT a positivity oracle."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import resource

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
STDLIB=tuple(Path(p).resolve() for p in sys.path if p)


def guard(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT):raise PermissionError('read-only checker')
        if not path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in STDLIB):raise PermissionError('outside moved artifact')
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.exec','os.fork'):raise PermissionError('external call prohibited')
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy','full_source','source_enclosure','source_ranges','range_pipeline'):
        raise ImportError('model/solver/producer prohibited')


def verify(root,expected_hash):
    from verify_prefix import verify as prefix_verify
    from proof_format import identity
    from step_check import check_relu,check_join
    from full_graph import validate,operator
    from delta_format import restore
    from lift_check import check as check_lift
    from obligation_check import check as check_obligations
    from range_step_check import check as check_ranged
    load=lambda n:json.loads((root/n).read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(root/'manifest.json')!=expected_hash:raise ValueError('external full manifest binding')
    m=load('manifest.json')
    if m['schema']!='RANGED_COMPLETE_EXPERT_LP_BUNDLE_V1' or m['endpoint']!='COMPLETE_SOURCE_AND_NEW_LP_CONSTRUCTIONS_NOT_POSITIVITY':
        raise ValueError('endpoint is construction only')
    policy=m['policy']
    if type(policy.get('enabled')) is not bool or policy!={'enabled':policy['enabled'],'layer':6,'row_prefix':2,'native_seconds':3.}:
        raise ValueError('registered range roster/configuration')
    for name,h in m['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=h:raise ValueError('full package integrity')
    started=time.monotonic();prefix=prefix_verify(root/'prefix',m['prefix_manifest_sha256']);prefix_seconds=time.monotonic()-started
    source=load('prefix/router_source.json');pe=load('prefix/experts.json');doc=load('full_experts.json')
    pair=validate(doc,source,pe)
    if m['request']!=source['request'] or m['pair']!=pair or prefix['pair']!=pair:raise ValueError('full request coverage')
    trace=load('trace.json');expected=[(i,j) for i in pair for j in ['coordinate']+list(range(2,9))]
    if trace['schema']!='RANGED_COMPLETE_EXPERT_TRACE_V1' or [(v['expert'],v['layer']) for v in trace['steps']]!=expected:
        raise ValueError('missing/extra/out-of-order expert layer')
    required={'full_experts.json','trace.json','joint.state.json','lp_base.json','obligations.json',
              'full_graph.py','delta_format.py','lift_check.py','obligation_check.py','verify_full.py','range_check.py','range_step_check.py'}
    required|={'prefix/manifest.json'}|{'prefix/'+n for n in load('prefix/manifest.json')['files']}
    checks=[];ends=[];pos=0
    for expert in doc['experts']:
        i=expert['expert'];shape=operator(source['request']['lower']['shape'],expert['layers'][0])[0]
        state=load(f'prefix/expert{i}_relu.state.json')
        for index in ['coordinate']+list(range(2,9)):
            row=trace['steps'][pos];pos+=1;old=state;before=shape;tag=f'expert{i}/full/{index}'
            if index=='coordinate':kind='CoordinateLift';op=[{j:1} for j in range(len(state['hz']['c']))];bias=[0]*len(op)
            else:kind=expert['layers'][index]['kind'];shape,op,bias=operator(shape,expert['layers'][index])
            if row['kind']!=kind or row['input_shape']!=before or row['output_shape']!=shape:raise ValueError('layer shape/kind')
            if kind=='Flatten':
                if row['source']!=identity(state) or row['target']!=identity(state):raise ValueError('flatten changed factor state')
                checked={'status':'CHECKED_FLATTEN_ORDER','state_sha256':identity(state)}
            else:
                name=f'expert{i}_{index}.delta.json'
                if row['file']!=name:raise ValueError('layer file binding')
                required.add(name);state=restore(old,load(name))
                if kind=='ReLU':checked=check_relu(old,state,row['proof'],tag)
                elif index==6:
                    facts=row['proof']['facts'];ctx={'request_id':identity(source['request']),'scope':f'pair{pair}/expert{i}','layer':'6'}
                    if len(facts)!=len(op) or any(f is not None and (not policy['enabled'] or j>=min(2,len(op))) for j,f in enumerate(facts)):
                        raise ValueError('fact outside registered row roster')
                    checked=check_ranged(old,state,op,bias,row['proof'],ctx,tag)
                else:checked=check_lift(old,state,op,bias,row['proof'],tag)
            checks.append({'expert':i,'layer':index,**checked})
        if shape!=[1,source['request']['classes']]:raise ValueError('incomplete expert endpoint')
        ends.append(state)
    if set(m['files'])!=required:raise ValueError('complete package file inventory')
    joint=load('joint.state.json')
    if identity(joint)!=trace['joint_sha256']:raise ValueError('new joint identity')
    joined=check_join(load('prefix/guard.state.json'),*ends,joint,trace['joint_proof'])
    obs=check_obligations(joint,load('lp_base.json'),load('obligations.json'),pair,source['request']['classes'],source['request']['clean_prediction'])
    return {'status':'CHECKED_RANGED_FULL_EXPERTS_AND_NEW_LP_CONSTRUCTIONS','range_policy':policy,
        'range_rows_checked':sum(r.get('checked_range_rows',0) for r in checks),'prefix_steps_checked':prefix['checked_steps'],
        'prefix_check_seconds':prefix_seconds,'remaining_steps_checked':len(checks),'steps':checks,'joint':joined,'outputs':obs,
        'pair':pair,'route_exclusions_checked':prefix['route_exclusions_checked'],'old_LP_certificates_used':0,
        'complete_output_positive_proof':False,'complete_strict_network_certificate':False,'production_verdict_changed':False,
        'remaining_trust':['declared graphs/parameters correspond to registered intended program','checker/interpreter execution'],
        'unclosed':['positive independently checked lower bounds on all new output LPs','native floating execution semantics'],
        'scope':'Full declared real experts enclosed and all new LP obligations checked; NO positive output conclusion.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest-hash',required=True);a=p.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:raise ValueError('python -I -S required')
    start=time.monotonic();sys.addaudithook(guard);sys.path[:0]=[str(ROOT),str(ROOT/'prefix')]
    result=verify(ROOT,a.manifest_hash);result['check_seconds']=time.monotonic()-start
    result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(result,sort_keys=True))
