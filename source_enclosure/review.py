"""Independent terminal review, relocated recheck and semantic negative cases."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.build import save
from router_source.checker import compact


def review(root):
    start=time.monotonic();t=json.loads((root/'terminal.json').read_bytes())
    p=json.loads((root/'publication.json').read_bytes());e=json.loads((root/'execution.json').read_bytes())
    f=ROOT/'docs/source_enclosure_v1_freeze.json'
    if sha(f)!=e['freeze_sha256'] or sha(root/'terminal.json')!=p['terminal_sha256']:raise ValueError('freeze/terminal identity')
    for group in ('sources','artifacts'):
        for name,h in json.loads(f.read_bytes())[group].items():
            if sha(ROOT/name)!=h:raise ValueError('bound source drift')
    for name,h in t['inventory'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=h:raise ValueError('artifact changed')
    if t['budget_seconds']!=300 or e['budget_seconds']!=300:raise ValueError('budget changed')
    if not t['elapsed_before_publication']<=p['seconds']<300 or (root/'publication_timeout.json').exists():
        raise ValueError('publication deadline')
    prev=0
    for i,s in enumerate(t['stages']):
        if (i>=2 or s['phase']!=('build','check')[i] or s['cap_seconds']!=(90,180)[i] or
                not prev<=s['start_seconds']<=s['end_seconds']<=p['seconds'] or
                s!=json.loads((root/(s['phase']+'_stage.json')).read_bytes())):raise ValueError('stage accounting')
        prev=s['end_seconds']
    if t['error'] or len(t['stages'])!=2 or any(s['state']!='COMPLETED' for s in t['stages']):
        return {'status':'PASS','issues':[],'outcome':t['status'],'complete':False,'failed_execution_preserved':True}
    moved=root/'review_relocated';shutil.copytree(root/'relocated',moved);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    def run(path,log):
        return execute([ACT,'-I','-S',str(path/'verify_prefix.py'),'--manifest-hash',sha(path/'manifest.json')],
            log,time.monotonic()+180,env)
    r=run(moved,root/'fresh_review.log')
    if r['state']!='COMPLETED':raise ValueError('fresh source composition failed')
    fresh=json.loads((root/'fresh_review.log').read_bytes())
    if {k:v for k,v in fresh.items() if k!='check_seconds'}!={k:v for k,v in t['check'].items() if k!='check_seconds'}:
        raise ValueError('exact recheck disagrees')
    if (fresh['checked_steps']!=7 or fresh['required_steps']!=7 or fresh['output_properties_checked']!=0 or
            fresh['complete_strict_network_certificate'] or fresh['production_verdict_changed'] or t['status']!=fresh['status']):
        raise ValueError('prefix/full claim mismatch')
    mutations=[];i=fresh['pair'][0]
    for mode in ('missing_step','error_bound','relu_range','join_map'):
        dst=root/('mutation_'+mode);shutil.copytree(moved,dst);file='trace.json'
        trace=json.loads((dst/file).read_bytes())
        if mode=='missing_step':trace['states'].pop(f'expert{i}_relu')
        if mode=='error_bound':trace['certificates'][f'expert{i}_affine']['error_bounds'][0]='0'
        if mode=='relu_range':trace['certificates'][f'expert{i}_relu']['ranges'][0]=['0','0']
        if mode=='join_map':trace['certificates']['join']['maps']['right_c'][-1]=0
        (dst/file).write_bytes(compact(trace));m=json.loads((dst/'manifest.json').read_bytes());m['files'][file]=sha(dst/file)
        (dst/'manifest.json').write_bytes(compact(m));rejected=run(dst,root/('mutation_'+mode+'.log'))
        if rejected['state']!='ERROR':raise ValueError('semantic tamper not rejected: '+mode)
        mutations.append({'name':mode,'state':'REJECTED','seconds':rejected['seconds']})
    return {'status':'PASS','issues':[],'execution_head':e['head'],'terminal_sha256':p['terminal_sha256'],
        'manifest_sha256':sha(moved/'manifest.json'),'result':{k:v for k,v in fresh.items() if k!='check_seconds'},
        'generation':json.loads((root/'generation.json').read_bytes()),'stages':t['stages'],
        'stored_source_total_seconds':p['seconds'],'fresh_relocated_check_seconds':fresh['check_seconds'],
        'semantic_mutations':mutations,'postterminal_review_seconds':time.monotonic()-start,
        'complete_strict_network_certificate':False,'production_verdict_changed':False}


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();result=review(root);save(root/'review.json',result);print(json.dumps(result,indent=2))
