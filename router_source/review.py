"""Independent stored execution audit plus a second relocated complete check."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.checker import compact,digest
from router_source.build import save


def review(root):
    start=time.monotonic();terminal=json.loads((root/'terminal.json').read_bytes())
    pub=json.loads((root/'publication.json').read_bytes());ex=json.loads((root/'execution.json').read_bytes())
    freeze=ROOT/'docs/router_source_v1_freeze.json'
    if sha(freeze)!=ex['freeze_sha256'] or sha(root/'terminal.json')!=pub['terminal_sha256']:
        raise ValueError('protocol/terminal identity')
    for name,h in json.loads(freeze.read_bytes())['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('runtime source identity')
    for name,h in terminal['inventory'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('artifact identity changed')
    if (terminal['budget_seconds']!=300 or ex['budget_seconds']!=300 or
            not terminal['elapsed_before_publication']<=pub['seconds']<300 or
            (root/'publication_timeout.json').exists()):raise ValueError('publication deadline')
    prev=0.
    for i,s in enumerate(terminal['stages']):
        if (s['phase']!=('build','check')[i] or not prev<=s['start_seconds']<=s['end_seconds']<=pub['seconds'] or
            s!=json.loads((root/(s['phase']+'_stage.json')).read_bytes())):
            raise ValueError('stage order/cost identity')
        prev=s['end_seconds']
    if terminal['error'] or len(terminal['stages'])!=2 or any(s['state']!='COMPLETED' for s in terminal['stages']):
        return {'status':'PASS','issues':[],'outcome':terminal['status'],
                'complete':False,'failed_execution_preserved':True,'new_solver_calls':0}
    destination=root/'review_relocated';shutil.copytree(root/'relocated',destination)
    extension=sha(destination/'extension.json');log=root/'fresh_review.log'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    owner=execute([ACT,'-I','-S',str(destination/'verify_with_router.py'),'--extension-hash',extension],
                  log,time.monotonic()+180,env)
    if owner['state']!='COMPLETED':raise ValueError('fresh complete check failed')
    result=json.loads(log.read_bytes())
    original=terminal['check']
    if {k:v for k,v in result.items() if k!='check_seconds'}!={k:v for k,v in original.items() if k!='check_seconds'}:
        raise ValueError('fresh mathematical result disagrees')
    if terminal['status']!=result['status'] or terminal['complete_strict_network_certificate'] or terminal['production_verdict_changed']:
        raise ValueError('claim mismatch')
    # Mutations are independent postterminal controls, not new solver requests.
    controls=[]
    for name in ('missing_route','wrong_parameter','wrong_pool'):
        target=root/('mutation_'+name);shutil.copytree(destination,target)
        e=json.loads((target/'extension.json').read_bytes())
        if name=='missing_route':
            file='router_proof.json';obj=json.loads((target/file).read_bytes());obj['routes'].pop()
        else:
            file='router_source.json';obj=json.loads((target/file).read_bytes())
            if name=='wrong_parameter':
                # Swap two intact byte strings with valid lengths; model identity
                # still pins the original weight and bias tensors.
                import base64,struct
                t=obj['parameters'][obj['graph']['bias']];raw=bytearray(base64.b64decode(t['bytes']))
                raw[:8]=struct.pack('<d',123.);t['bytes']=base64.b64encode(raw).decode()
            else:obj['graph']['divisor_override']=3
            e['source_identity']=digest(compact(obj))
            p=json.loads((target/'router_proof.json').read_bytes());p['source_sha256']=e['source_identity']
            (target/'router_proof.json').write_bytes(compact(p));e['files']['router_proof.json']=sha(target/'router_proof.json')
        (target/file).write_bytes(compact(obj));e['files'][file]=sha(target/file)
        (target/'extension.json').write_bytes(compact(e))
        owner=execute([ACT,'-I','-S',str(target/'verify_with_router.py'),'--extension-hash',sha(target/'extension.json')],
            root/('mutation_'+name+'.log'),time.monotonic()+10,env)
        if owner['state']!='ERROR':raise ValueError('mutated proof not rejected: '+name)
        controls.append({'name':name,'state':'REJECTED','seconds':owner['seconds']})
    generation=json.loads((root/'generation.json').read_bytes())
    return {'status':'PASS','issues':[],'outcome':result['status'],'execution_head':ex['head'],
        'terminal_sha256':pub['terminal_sha256'],'extension_sha256':extension,
        'route_check':result['route_check'],'required_obligations':result['required_obligations'],
        'minimum_lower_bound':result['minimum_lower_bound'],'remaining_trusted_base':result['remaining_trusted_base'],
        'complete_strict_network_certificate':False,'production_verdict_changed':False,
        'new_solver_calls':0,'new_network_forward_calls':0,'new_HZ_propagations':0,
        'stored_source_total_seconds':pub['seconds'],'generation':generation,'stages':terminal['stages'],
        'fresh_relocated_check_seconds':result['check_seconds'],'mutation_controls':controls,
        'postterminal_review_seconds':time.monotonic()-start,
        'scope':result['scope']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    a=p.parse_args();root=a.root.resolve();r=review(root);save(root/'review.json',r);print(json.dumps(r,indent=2))
