"""Fresh relocated local checks, identity/cost review and semantic mutations."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.checker import compact
from router_source.build import save


def compact_result(result):
    result=copy.deepcopy(result)
    result.pop('check_seconds',None)
    result['input_cover'].pop('failures')
    for row in result['first_conv_steps']:row.pop('row_error_bounds')
    return result


def review(root):
    start=time.monotonic();terminal=json.loads((root/'terminal.json').read_bytes())
    pub=json.loads((root/'publication.json').read_bytes());execution=json.loads((root/'execution.json').read_bytes())
    freeze=ROOT/'docs/upstream_source_v1_freeze.json'
    if sha(freeze)!=execution['freeze_sha256'] or sha(root/'terminal.json')!=pub['terminal_sha256']:
        raise ValueError('freeze/terminal identity')
    for category in ('sources','artifacts'):
        for name,h in json.loads(freeze.read_bytes())[category].items():
            if sha(ROOT/name)!=h:raise ValueError('bound source changed')
    for name,h in terminal['inventory'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=h:raise ValueError('artifact changed')
    if (terminal['budget_seconds']!=300 or execution['budget_seconds']!=300 or
            not terminal['elapsed_before_publication']<=pub['seconds']<300 or (root/'publication_timeout.json').exists()):
        raise ValueError('publication budget')
    prev=0
    for i,s in enumerate(terminal['stages']):
        if (i>=2 or s['phase']!=('build','check')[i] or
                not prev<=s['start_seconds']<=s['end_seconds']<=pub['seconds'] or
                s!=json.loads((root/(s['phase']+'_stage.json')).read_bytes())):raise ValueError('stage cost identity')
        prev=s['end_seconds']
    if terminal['error'] or len(terminal['stages'])!=2 or any(s['state']!='COMPLETED' for s in terminal['stages']):
        return {'status':'PASS','issues':[],'outcome':terminal['status'],'complete':False,'failed_execution_preserved':True}
    destination=root/'review_relocated';shutil.copytree(root/'relocated',destination)
    manifest=sha(destination/'manifest.json');env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    def run(path,log):
        return execute([ACT,'-I','-S',str(path/'verify_upstream.py'),'--manifest-hash',sha(path/'manifest.json')],
                       log,time.monotonic()+180,env)
    row=run(destination,root/'fresh_review.log')
    if row['state']!='COMPLETED':raise ValueError('fresh local check did not complete')
    fresh=json.loads((root/'fresh_review.log').read_bytes());original=terminal['check']
    if {k:v for k,v in fresh.items() if k!='check_seconds'}!={k:v for k,v in original.items() if k!='check_seconds'}:
        raise ValueError('full exact diagnostics differ')
    if (terminal['status']!=fresh['status'] or fresh['complete_strict_network_certificate'] or
            fresh['production_verdict_changed']):raise ValueError('claim promotion')
    controls=[]
    for name in ('missing_expert','wrong_expert_parameter','wrong_frame'):
        target=root/('mutation_'+name);shutil.copytree(destination,target)
        m=json.loads((target/'manifest.json').read_bytes())
        file='input_hz.json' if name=='wrong_frame' else 'experts.json';obj=json.loads((target/file).read_bytes())
        if name=='missing_expert':obj.pop()
        if name=='wrong_expert_parameter':obj[0]['weight_name']='experts.0.0.weight'
        if name=='wrong_frame':obj['frame_id']+=1
        (target/file).write_bytes(compact(obj));m['files'][file]=sha(target/file)
        (target/'manifest.json').write_bytes(compact(m))
        rejected=run(target,root/('mutation_'+name+'.log'))
        if rejected['state']!='ERROR':raise ValueError('semantic mutation not rejected')
        controls.append({'name':name,'state':'REJECTED','seconds':rejected['seconds']})
    return {'status':'PASS','issues':[],'execution_head':execution['head'],
        'terminal_sha256':pub['terminal_sha256'],'manifest_sha256':manifest,
        'outcome':fresh['status'],'result':compact_result(fresh),
        'generation':json.loads((root/'generation.json').read_bytes()),'stages':terminal['stages'],
        'stored_source_total_seconds':pub['seconds'],'fresh_relocated_check_seconds':fresh['check_seconds'],
        'mutations':controls,'postterminal_review_seconds':time.monotonic()-start,
        'complete_strict_network_certificate':False,'production_verdict_changed':False}


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();result=review(root);save(root/'review.json',result);print(json.dumps(result,indent=2))
