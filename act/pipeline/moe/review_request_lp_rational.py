"""Read-only recheck of R3 construction and complete request, no builder."""
import argparse
import json
from pathlib import Path
from act.pipeline.moe.experiment1 import PROJECT_ROOT,_sha256
from act.pipeline.moe.paired_followup import save
from act.pipeline.moe.check_request_lp import check_directory,RATIONAL_TRUSTED
from act.back_end.solver.check_hz_lp_export import check_export


def review(root):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
    reference=json.loads((PROJECT_ROOT/'act/pipeline/moe/results/request_lp_order_review_20260914_r2.json').read_text())
    for name,h in reference['raw_hashes'].items():
        if _sha256(root/'parent'/name)!=h:raise ValueError('parent artifact changed')
    old=json.loads((root/'parent/manifest.json').read_text())
    if m['request']!=old['request'] or m['routes']!=old['routes']:raise ValueError('request or route coverage changed')
    def read(ref):
        path=(root/ref['file']).resolve()
        if not path.is_relative_to(root) or _sha256(path)!=ref['sha256']:raise ValueError('artifact mismatch')
        return json.loads(path.read_text())
    for key,proof in old['proofs'].items():
        item=m['proofs'][key]
        for field in ('export','certificate'):
            if proof[field] and item[field]['sha256']!=proof[field]['sha256']:raise ValueError('historical proof changed')
        check_export(read(item['export']),read(item['certificate']) if item['certificate'] else None,
                     expected_source_sha256=item['hz_sha256'])
    new_keys=set(m['proofs'])-set(old['proofs'])
    if len(new_keys)!=3:raise ValueError('query inventory changed')
    changes=[]
    for before,after in zip(old['obligations'],m['obligations']):
        if before['kind']=='reused':
            if before!=after:raise ValueError('reuse changed')
        else:
            if {k:v for k,v in before.items() if k!='source'}!={k:v for k,v in after.items() if k!='source'}:
                raise ValueError('ranges or obligation scope changed')
            if after['source'] not in new_keys:raise ValueError('floating F0 still used')
            item=m['proofs'][after['source']]
            changes.append({'pair':after['pair'],'property_index':after['property_index'],
                            'status':item['status'],'checked_lower_bound':item.get('checked_lower_bound'),
                            'lambda_bounds':after['lambda_bounds'],'difference_bounds':after['difference_bounds']})
    result=check_directory(root,expected_request_id=old['request_id'])
    if result!=json.loads((root/'check.json').read_text()) or result['trusted_base']!=RATIONAL_TRUSTED:
        raise ValueError('aggregation mismatch')
    return {'audit':'PASS','issues':[],'classification':'DIRECT_RATIONAL_CONSTRUCTION_SAME_FROZEN_REQUEST',
            'check':result,'new_queries':len(new_keys),'residuals':changes,
            'removed_trusted_assumption':'F0_outer_HZ_construction_and_floating_coefficients',
            'launch':json.loads((root/'launch.json').read_text()),'terminal':json.loads((root/'terminal.json').read_text()),
            'raw_root':str(root),'raw_hashes':{str(p.relative_to(root)):_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()},
            'scope':'Rational construction and necessary output LP bounds checked from supplied shared HZ; upstream propagation/guards/route exclusions trusted, not native floating-point proof.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=review(a.root);save(a.output,r);print(json.dumps(r['check'],indent=2))
