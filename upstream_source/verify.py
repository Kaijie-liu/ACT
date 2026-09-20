"""Relocatable local upstream audit; no checkpoint, data, HZ library or solver."""
import argparse
from fractions import Fraction  # Preload stdlib numeric dependencies before IO guard.
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
            raise PermissionError('read-only check')
        if not path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in STDLIB):
            raise PermissionError('outside relocated evidence/stdlib')
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.exec','os.fork'):
        raise PermissionError('external execution prohibited')
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy'):
        raise ImportError('model/solver import prohibited')


def check(directory,expected_hash):
    from router_check import inputs,tensor,check as check_routes,compact,digest
    from local_check import input_cover,affine_error,conv_operator,pair_guards
    def load(name):return json.loads((directory/name).read_bytes())
    raw=(directory/'manifest.json').read_bytes()
    if digest(raw)!=expected_hash:raise ValueError('manifest identity')
    manifest=json.loads(raw)
    if manifest['schema']!='UPSTREAM_LOCAL_AUDIT_V1':raise ValueError('manifest schema')
    required={'router_source.json','router_proof.json','joint_hz.json','router_hz.json',
              'input_hz.json','experts.json','router_check.py','local_check.py','verify_upstream.py'}
    required|={f'expert{i}_conv0.json' for i in manifest['pair']}
    if set(manifest['files'])!=required:raise ValueError('complete local obligation inventory')
    for name,h in manifest['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path.read_bytes())!=h:
            raise ValueError('source/code file identity')
    for name,h in manifest['historical_source_sha256'].items():
        if name not in ('joint_hz.json','router_hz.json') or h!=manifest['files'][name]:
            raise ValueError('old HZ source binding')
    if set(manifest['historical_source_sha256'])!={'joint_hz.json','router_hz.json'}:raise ValueError('old HZ inventory')
    source=load('router_source.json');request=manifest['request']
    shape,_,_,image=inputs(source,request)
    routes=check_routes(source,load('router_proof.json'),expected_request=request,
                        expected_source_sha256=digest(compact(source)))
    if routes['covered_pairs']!=[manifest['pair']]:raise ValueError('local study requires registered sole pair')
    router,joint=load('router_hz.json'),load('joint_hz.json')
    if len(router['c'])!=request['experts'] or len(joint['c'])!=2*request['classes']:
        raise ValueError('historical output/property dimensions')
    guard_result=pair_guards(router,joint,manifest['pair'])
    entry=load('input_hz.json');box=input_cover(image['lower'],image['upper'],entry)
    expert_sources=load('experts.json')
    if [v['expert'] for v in expert_sources]!=manifest['pair']:raise ValueError('expert order/coverage')
    inventory={v['name']:{k:v[k] for k in ('dtype','shape','sha256')} for v in source['state_inventory']}
    experts=[]
    for item in expert_sources:
        i=item['expert']
        if (item['layer_index']!=0 or item['weight_name']!=f'experts.{i}.0.weight' or
                item['bias_name']!=f'experts.{i}.0.bias' or item['output_file']!=f'expert{i}_conv0.json'):
            raise ValueError('expert/layer/parameter binding')
        wident,weight=tensor(item['weight']);bident,bias=tensor(item['bias'])
        if inventory.get(item['weight_name'])!=wident or inventory.get(item['bias_name'])!=bident:
            raise ValueError('original expert parameter identity')
        operator,constants=conv_operator(shape,item['graph'],weight,wident['shape'],bias)
        if bident['shape']!=[wident['shape'][0]]:raise ValueError('bias shape')
        experts.append({'expert':i,'layer_index':0,**affine_error(entry,load(item['output_file']),operator,constants)})
    return {'status':'CHECKED_LOCAL_UPSTREAM_AUDIT_NOT_NETWORK_PROOF',
        'input_cover':box,'saved_pair_guards':guard_result,'first_conv_steps':experts,
        'original_parameter_routes':routes['covered_pairs'],
        'all_upstream_steps_checked':False,'complete_strict_network_certificate':False,
        'production_verdict_changed':False,
        'unclosed':['input-to-HZ containment if reported gaps are nonzero',
            'subsequent expert layers and per-ReLU independently checked range/encoding evidence',
            'historical shared/private factor maps and intermediate-state identity chain',
            'membership-guard/common-interval sources used by the reused output obligation',
            'declared graph correspondence and deployed floating semantics'],
        'scope':'Original source bindings + actual saved pair guards + NEW local first-conv controls. No historical full expert trace, no discharged global upstream assumption.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest-hash',required=True);a=p.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:raise ValueError('requires python -I -S')
    start=time.monotonic();sys.addaudithook(guard);sys.path.insert(0,str(ROOT))
    result=check(ROOT,a.manifest_hash);result['check_seconds']=time.monotonic()-start
    print(json.dumps(result,sort_keys=True))
