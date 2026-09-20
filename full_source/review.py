"""Independent saved terminal accounting and isolated relocated construction review."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.build import save


def review(root):
    start=time.monotonic();load=lambda n:json.loads((root/n).read_bytes())
    e,t,p=load('execution.json'),load('terminal.json'),load('publication.json');freeze=ROOT/'docs/full_source_v1_freeze.json'
    if sha(freeze)!=e['freeze_sha256'] or sha(root/'terminal.json')!=p['terminal_sha256']:raise ValueError('terminal/freeze identity')
    f=json.loads(freeze.read_bytes())
    for group in ('sources','artifacts'):
        for name,h in f[group].items():
            if sha(ROOT/name)!=h:raise ValueError('frozen source drift')
    for name,h in t['inventory'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=h:raise ValueError('terminal artifact drift')
    if e['budget_seconds']!=300 or t['budget_seconds']!=300:raise ValueError('budget drift')
    prev=0
    for i,s in enumerate(t['stages']):
        if (i>=2 or s['phase']!=('build','check')[i] or not prev<=s['start_seconds']<=s['end_seconds']<=t['elapsed_before_publication'] or
                s!=load(s['phase']+'_stage.json')):raise ValueError('stage accounting')
        prev=s['end_seconds']
    late=p['seconds']>=300 or (root/'publication_timeout.json').exists()
    if t['complete_output_positive_proof'] or t['complete_strict_network_certificate'] or t['production_verdict_changed']:
        raise ValueError('construction improperly upgraded to SAFE')
    result={'status':'PASS','issues':[],'execution_head':e['head'],'outcome':'TIMEOUT' if late else t['status'],
        'terminal_sha256':p['terminal_sha256'],'total_seconds':p['seconds'],'stages':t['stages'],
        'complete_output_positive_proof':False,'complete_strict_network_certificate':False,'production_verdict_changed':False}
    if late or t['error'] or len(t['stages'])!=2 or any(s['state']!='COMPLETED' for s in t['stages']):
        result.update(complete=False,failed_execution_preserved=True)
        result['partial_progress']=[json.loads(q.read_bytes()) for q in sorted(root.glob('progress_*.json'))]
        return result
    if not t['elapsed_before_publication']<=p['seconds']<300:raise ValueError('publication accounting')
    moved=root/'review_relocated';shutil.copytree(root/'relocated',moved)
    row=execute([ACT,'-I','-S',str(moved/'verify_full.py'),'--manifest-hash',sha(moved/'manifest.json')],
                root/'fresh_review.log',time.monotonic()+298,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    if row['state']!='COMPLETED':raise ValueError('relocated complete check not completed')
    fresh=load('fresh_review.log');strip=lambda r:{k:v for k,v in r.items() if k not in ('check_seconds','prefix_check_seconds','peak_rss_kib')}
    if strip(fresh)!=strip(t['check']) or fresh['status']!=t['status']:raise ValueError('fresh review disagreement')
    if fresh['remaining_steps_checked']!=16 or fresh['outputs']['obligations']!=9 or fresh['outputs']['lower_bounds_checked']!=0:
        raise ValueError('real scope/obligation inventory')
    result.update(complete=True,result=strip(fresh),generation=load('generation.json'),
        manifest_sha256=sha(moved/'manifest.json'),fresh_relocated_check_seconds=fresh['check_seconds'],
        fresh_relocated_peak_rss_kib=fresh['peak_rss_kib'],
        separate_review_seconds=time.monotonic()-start)
    return result


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();result=review(root);save(root/'review.json',result);print(json.dumps(result,indent=2))
