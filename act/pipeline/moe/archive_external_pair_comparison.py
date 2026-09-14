"""Archive frozen external comparison; no new bound queries or replacements."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean,median
import subprocess

from act.pipeline.moe.experiment1 import PROJECT_ROOT,_sha256
from act.pipeline.moe.paired_followup import save
from act.pipeline.moe.review_external_pair_comparison import audit

HEAD='0de4fe1c77cf5491faf75db6bbd0ef354ee9c821'
BASE=PROJECT_ROOT/'data/moe/results'
OUTPUT=PROJECT_ROOT/'act/pipeline/moe/results/external_pair_comparison_review_20260914_r1.json'


def ref(path):return {'path':str(path),'sha256':_sha256(path)}


def execution_source():
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',HEAD,'act'],cwd=PROJECT_ROOT,text=True)
    digest=hashlib.sha256()
    for name in sorted(n for n in names.splitlines() if n.endswith('.py')):
        digest.update(f'{name}:{_sha256(PROJECT_ROOT/name)}\n'.encode())
    return digest.hexdigest()


def describe(rows):
    """All observations contribute; states and evidence classes stay separate."""
    output={}
    for arm in ('adaptive','crown'):
        values=[r for r in rows if r['method']==arm]
        costs=[r['wall_seconds'] for r in values]
        output[arm]={'denominator':len(values),'states':dict(Counter(r['status'] for r in values)),
            'total_observed_seconds':sum(costs),'mean_observed_seconds':mean(costs),
            'median_observed_seconds':median(costs),'outer_timeouts':sum(r['outer_timeout'] for r in values),
            'per_state':{state:{'count':sum(r['status']==state for r in values),
                'mean_observed_seconds':mean(r['wall_seconds'] for r in values if r['status']==state)}
                for state in sorted({r['status'] for r in values})}}
    return output


def build():
    source=execution_source();result={'execution_head':HEAD,'execution_source_sha256':source,'experiments':{}}
    for phase in ('smoke','full'):
        root=BASE/f'external_pair_{phase}_20260914_r1';rt=json.loads((root/'runtime.json').read_text())
        if rt['git_head']!=HEAD or rt['source_sha256']!=source:raise ValueError('frozen source drift')
        checked=audit(root)
        if checked!=json.loads((root/'audit.final.json').read_text()):raise ValueError('re-audit differs from saved audit')
        result['experiments'][phase]={'runtime':ref(root/'runtime.json'),'audit':ref(root/'audit.final.json'),
            'summary':checked,'input_preparation_seconds_excluded_equally':rt['input_preparation_seconds_excluded_equally'],
            'raw_hashes':{str(p.relative_to(root)):_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    root=BASE/'external_pair_full_20260914_r1';rows=[json.loads(v) for v in (root/'rows.jsonl').read_text().splitlines()]
    summary=result['experiments']['full']['summary'];details=[];positives=[]
    for model,table in summary['models'].items():
        for item in table['rows']:
            rank=item['rank'];record={'model':model,**item}
            a,b=[next(r for r in rows if (r['model'],r['rank'],r['method'])==(model,rank,arm)) for arm in ('adaptive','crown')]
            record['adaptive_terminal']=ref(root/a['job_id']/'terminal.json');record['crown_terminal']=ref(root/b['job_id']/'terminal.json')
            record['adaptive_seconds']=a['wall_seconds'];record['crown_seconds']=b['wall_seconds']
            record['adaptive_package']=a['package'];record['outer_timeout']=a['outer_timeout']
            if a['package']:
                ep=Path(a['package'])/'evidence.json';e=json.loads(ep.read_text())
                record.update(adaptive_evidence=ref(ep),decision_tier=e['verdict']['decision_tier'],reason=e['verdict']['reason'])
            else:record.update(decision_tier=None,reason='OUTER_REQUEST_TIMEOUT')
            extpath=root/b['job_id']/'external.json';external=json.loads(extpath.read_text())
            record['external_record']=ref(extpath);record['external_reason']=external['reason']
            obligations=[{'pair':p['pair'],'property_index':i,'lower':lo,'upper':p['upper'][i]}
                         for p in external['pairs'] for i,lo in enumerate(p['lower'])]
            record['crown_minimum_obligation']=min(obligations,key=lambda r:r['lower']) if obligations else None
            record['crown_nonpositive_obligations']=[p for p in obligations if p['lower']<=1e-7]
            record['pair_bounds_completed']=len(external['pairs']);details.append(record)
            if (a['status']=='SAFE') != (b['status']=='POSITIVE'):
                positives.append({**record,'positive_only':'adaptive' if a['status']=='SAFE' else 'crown'})
    # Two same-input models are not independent observations, and the combined
    # positive inventory below intentionally does not assert equal proof levels.
    strata={}
    for label in ('single','multiple','unavailable'):
        selected=[r for r in details if ('unavailable' if r['pair_count'] is None else 'single' if r['pair_count']==1 else 'multiple')==label]
        strata[label]={'model_input_pairs':len(selected),'adaptive_policy_safe':sum(r['adaptive']=='SAFE' for r in selected),
            'crown_numerical_positive':sum(r['crown']=='POSITIVE' for r in selected),
            'adaptive_only_positive':sum(r['adaptive']=='SAFE' and r['crown']!='POSITIVE' for r in selected),
            'crown_only_positive':sum(r['crown']=='POSITIVE' and r['adaptive']!='SAFE' for r in selected)}
    result.update(status='PASS',issues=[],exact_reaudit_equals_saved=True,full_costs=describe(rows),
        full_requests=details,positive_discordances=positives,route_strata=strata,
        positive_intersection=sum(r['adaptive']=='SAFE' and r['crown']=='POSITIVE' for r in details),
        scope='Ten observed images, three fixed same-family models; complete-cost hybrid external comparison. HZ-policy SAFE and CROWN numerical filters are unequal evidence classes. No new holdout, native proof, solver retuning or outcome-selected portfolio.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write and OUTPUT.exists():raise ValueError('refusing to overwrite archive')
    result=build()
    if a.write:save(OUTPUT,result)
    elif result!=json.loads(OUTPUT.read_text()):raise ValueError('archive does not reconstruct')
    print(json.dumps({'status':result['status'],'costs':result['full_costs'],'strata':result['route_strata']},indent=2))
