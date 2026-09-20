"""Source-preserving independent recheck, full terminal costs, and failure retention."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.build import save


def strip(result):
    return {k:v for k,v in result.items() if k not in ('check_seconds','source_check_seconds','bound_check_seconds','peak_rss_kib')}


def review(root):
    started=time.monotonic();load=lambda n:json.loads((root/n).read_bytes())
    e,t,p=load('execution.json'),load('terminal.json'),load('publication.json');fpath=ROOT/'docs/full_bounds_v1_freeze.json'
    if sha(fpath)!=e['freeze_sha256'] or sha(root/'terminal.json')!=p['terminal_sha256']:raise ValueError('frozen terminal identity')
    f=json.loads(fpath.read_bytes())
    for group in ('sources','artifacts'):
        for name,h in f[group].items():
            if sha(ROOT/name)!=h:raise ValueError('frozen code/source changed')
    for name,h in t['inventory'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=h:raise ValueError('raw artifact drift')
    if e['budget_seconds']!=300 or t['budget_seconds']!=300 or t['production_verdict_changed'] or t['deployed_floating_point_proof']:
        raise ValueError('budget/claim drift')
    previous=0;order=['prepare','propose','seal','check'];caps=[15,180,10,300]
    for i,row in enumerate(t['stages']):
        if (i>=4 or row['phase']!=order[i] or row['cap_seconds']!=caps[i] or
                not previous<=row['start_seconds']<=row['end_seconds']<=t['elapsed_before_publication'] or row!=load(order[i]+'_stage.json')):
            raise ValueError('full phase accounting')
        previous=row['end_seconds']
    late=p['seconds']>=300 or (root/'publication_timeout.json').exists()
    result={'status':'PASS','issues':[],'execution_head':e['head'],'outcome':'TIMEOUT' if late else t['status'],
        'total_seconds':p['seconds'],'terminal_sha256':p['terminal_sha256'],'stages':t['stages'],
        'production_verdict_changed':False,'deployed_floating_point_proof':False}
    entered=[load(q.name) for q in sorted(root.glob('entered_*.json'))]
    if [r['competitor'] for r in entered]!=list(range(1,len(entered)+1)) or any(r['native_seconds']!=16 for r in entered) or len(entered)>9:
        raise ValueError('fixed one-call roster/native cap')
    result['native_calls_entered']=len(entered)
    if late or t['error'] or len(t['stages'])!=4 or t['stages'][-1]['state']!='COMPLETED':
        result.update(complete=False,failed_execution_preserved=True,check=t['check']);return result
    if not t['elapsed_before_publication']<=p['seconds']<300:raise ValueError('terminal publication accounting')
    result['solver_environment']=load('solver_environment.json') if (root/'solver_environment.json').exists() else None
    if result['solver_environment'] is None:
        if entered:raise ValueError('missing entered-call environment')
    elif result['solver_environment']['options']!={'threads':1,'time_limit':16.} or result['solver_environment']['method']!='highs':
        raise ValueError('solver configuration drift')
    src=root/'relocated';m=json.loads((src/'manifest.json').read_bytes());proposals=[]
    for entry in m['outcomes']:
        if entry['file'] is None:continue
        row=json.loads((src/entry['file']).read_bytes())
        if row['solver']['native_limit_seconds']!=16:raise ValueError('changed native cap')
        proposals.append({'competitor':row['competitor'],'candidate_available':row['candidate'] is not None,
                          'solver':row['solver'],'elapsed_before_publication':row['elapsed_before_publication']})
    moved=root/'review_relocated';shutil.copytree(src,moved)
    fresh=execute([ACT,'-I','-S',str(moved/'verify_bounds.py'),'--manifest-hash',sha(moved/'manifest.json')],
        root/'fresh_review.log',time.monotonic()+298,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    if fresh['state']!='COMPLETED':raise ValueError('independent relocated recheck incomplete')
    checked=load('fresh_review.log')
    if strip(checked)!=strip(t['check']):raise ValueError('fresh mathematical recheck differs')
    if checked['required_obligations']!=9 or checked['positive_bounds']+checked['nonpositive_bounds']+checked['missing_bounds']!=9:
        raise ValueError('complete fixed denominator')
    expected_positive=checked['complete_declared_real_output_proof'] and t['status']=='CHECKED_POSITIVE_DECLARED_REAL_MOE'
    if t['complete_declared_real_output_proof']!=expected_positive:raise ValueError('positive aggregation mismatch')
    result.update(complete=True,result=strip(checked),proposals=proposals,sealed=load('sealed.json'),
        preparation=load('preparation.json'),manifest_sha256=sha(moved/'manifest.json'),
        fresh_relocated_check_seconds=checked['check_seconds'],fresh_source_check_seconds=checked['source_check_seconds'],
        fresh_bound_check_seconds=checked['bound_check_seconds'],fresh_peak_rss_kib=checked['peak_rss_kib'],
        postterminal_review_seconds=time.monotonic()-started)
    if (root/'proposal.json').exists():
        result['proposal']=load('proposal.json')
        if result['proposal']['calls']!=len(entered) or result['proposal']['native_seconds_per_call']!=16:
            raise ValueError('native call accounting mismatch')
    return result


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();result=review(root);save(root/'review.json',result);print(json.dumps(result,indent=2))
