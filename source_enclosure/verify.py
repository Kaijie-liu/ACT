"""Isolated source-prefix proof composition, with all required steps checked."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
STDLIB=tuple(Path(p).resolve() for p in sys.path if p)


def guard(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT):
            raise PermissionError('read-only checking')
        if not path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in STDLIB):
            raise PermissionError('outside moved proof/stdlib')
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.exec','os.fork'):
        raise PermissionError('external execution prohibited')
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy','source_enclosure'):
        raise ImportError('model/solver/producer imports prohibited')


def verify(root,manifest_hash):
    from router_check import inputs,tensor,check as check_routes,compact,digest
    from local_check import hz,conv_operator,input_cover
    from proof_format import identity
    from step_check import check_box,check_guards,check_affine,check_relu,check_join
    def load(name):return json.loads((root/name).read_bytes())
    raw=(root/'manifest.json').read_bytes()
    if digest(raw)!=manifest_hash:raise ValueError('external manifest identity')
    m=json.loads(raw);pair=m['pair']
    if m['schema']!='COMPENSATED_EXPERT_PREFIX_BUNDLE_V1' or m['endpoint']!='TWO_EXPERTS_AFTER_FIRST_CONV_RELU_NOT_CLASSIFICATION':
        raise ValueError('prefix proof, not an output safety certificate')
    source_names={'router_source.json','router_proof.json','router_hz.json','input_hz.json','experts.json'}
    source_names|={f'expert{i}_conv0.json' for i in pair}
    state_names={'input','guard','join'}|{f'expert{i}_{s}' for i in pair for s in ('affine','relu')}
    required=source_names|{n+'.state.json' for n in state_names}|{'trace.json','proof_format.py','step_check.py','verify_prefix.py','local_check.py','router_check.py'}
    if set(m['files'])!=required or set(m['source_files'])!=source_names:raise ValueError('complete source/step inventory')
    for name,h in m['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path.read_bytes())!=h:raise ValueError('file identity')
    if any(m['source_files'][n]!=m['files'][n] for n in source_names):raise ValueError('old source drift')
    source=load('router_source.json');request=m['request'];shape,_,_,image=inputs(source,request)
    routes=check_routes(source,load('router_proof.json'),expected_request=request,expected_source_sha256=digest(compact(source)))
    if len(pair)!=2 or routes['covered_pairs']!=[pair]:raise ValueError('declared sole-pair scope')
    trace=load('trace.json')
    if (trace['schema']!='COMPENSATED_PREFIX_TRACE_V1' or set(trace['states'])!=state_names or
            set(trace['certificates'])!=state_names-{'input','guard'}):raise ValueError('missing/extra prefix step')
    states={n:load(n+'.state.json') for n in state_names}
    if any(identity(states[n])!=trace['states'][n] for n in state_names):raise ValueError('state chain identity')
    certificates=trace['certificates'];checks=[]
    checks.append(check_box(image['lower'],image['upper'],states['input']))
    old=input_cover(image['lower'],image['upper'],load('input_hz.json'))
    router=hz(load('router_hz.json'))
    if router['nb'] or router['b'] or len(router['c'])!=request['experts'] or len(router['ub'])!=2*(request['experts']-2):
        raise ValueError('registered affine pair guard source')
    checks.append(check_guards(states['input'],states['guard'],router['Auc'],router['Aub'],router['ub']))
    experts=load('experts.json')
    if [e['expert'] for e in experts]!=pair:raise ValueError('expert order/coverage')
    inventory={v['name']:{k:v[k] for k in ('dtype','shape','sha256')} for v in source['state_inventory']}
    for e in experts:
        i=e['expert'];wi,w=tensor(e['weight']);bi,b=tensor(e['bias'])
        if (e['layer_index']!=0 or e['weight_name']!=f'experts.{i}.0.weight' or e['bias_name']!=f'experts.{i}.0.bias'
                or e['output_file']!=f'expert{i}_conv0.json' or e['topology_inspected'][:2]!=['Conv2d','ReLU'] or
                wi!=inventory.get(e['weight_name']) or bi!=inventory.get(e['bias_name']) or bi['shape']!=[wi['shape'][0]]):
            raise ValueError('expert prefix/parameter identity')
        op,b=conv_operator(shape,e['graph'],w,wi['shape'],b)
        checks.append(check_affine(states['guard'],states[f'expert{i}_affine'],op,b,load(e['output_file']),
                                   certificates[f'expert{i}_affine'],f'expert{i}/conv0'))
        checks.append(check_relu(states[f'expert{i}_affine'],states[f'expert{i}_relu'],
                                 certificates[f'expert{i}_relu'],f'expert{i}/relu1'))
    checks.append(check_join(states['guard'],states[f'expert{pair[0]}_relu'],states[f'expert{pair[1]}_relu'],
                             states['join'],certificates['join']))
    return {'status':'CHECKED_COMPENSATED_EXPERT_PREFIX','required_steps':len(state_names),'checked_steps':len(checks),
        'checks':checks,'old_input_inward_coordinates':old['inward_coordinates'],
        'old_input_maximum_gap':old['maximum_inward_gap'],'new_input_inward_coordinates':0,
        'pair':pair,'route_exclusions_checked':len(routes['excluded_pairs']),
        'endpoint':m['endpoint'],'output_properties_checked':0,'old_LP_certificates_used':0,
        'complete_strict_network_certificate':False,'production_verdict_changed':False,
        'uncovered':['later expert Conv/pool/linear/ReLU layers','output properties and complete weighted aggregation',
                     'old membership/common-fact proof sources'],
        'remaining_trust':['declared prefix/router graphs correspond to intended registered program',
                           'checker/interpreter execution'],
        'scope':'Exact real declared prefix enclosure on pinned represented input box, with compensated affine errors, checked ReLU ranges/graph and shared/private maps. Not original float deployment or full MoE SAFE.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest-hash',required=True);a=p.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:raise ValueError('python -I -S required')
    start=time.monotonic();sys.addaudithook(guard);sys.path.insert(0,str(ROOT))
    result=verify(ROOT,a.manifest_hash);result['check_seconds']=time.monotonic()-start
    print(json.dumps(result,sort_keys=True))
