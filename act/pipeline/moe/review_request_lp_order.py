"""Independently recheck the parent and every new order-refined LP artifact."""
import argparse
import json
from pathlib import Path
from fractions import Fraction
from act.pipeline.moe.request_lp_order import validated_parent, PARENT_HASH
from act.pipeline.moe.request_lp_control import frozen_request
from act.pipeline.moe.check_request_lp import check_directory
from act.pipeline.moe.paired_followup import save
from act.pipeline.moe.experiment1 import PROJECT_ROOT, _sha256
from act.back_end.solver.lp_certificate import identity
from act.back_end.solver.check_hz_lp_export import check_export


def build():
    parent=validated_parent();root=PROJECT_ROOT/'data/moe/results/request_lp_order_20260914_r2'
    manifest=json.loads((root/'manifest.json').read_text());launch=json.loads((root/'launch.json').read_text())
    if _sha256(root/'base/manifest.json')!=PARENT_HASH or manifest['parent']['sha256']!=PARENT_HASH:
        raise ValueError('parent identity mismatch')
    if manifest['routes']!=parent['routes'] or manifest['request']!=parent['request']:
        raise ValueError('request/route partition changed')
    def read(ref):
        path=(root/ref['file']).resolve()
        if not path.is_relative_to(root) or _sha256(path)!=ref['sha256']:raise ValueError('artifact mismatch')
        return json.loads(path.read_text())
    proofs={}
    for key,item in manifest['proofs'].items():
        export=read(item['export']);cert=read(item['certificate']) if item['certificate'] else None
        checked=check_export(export,cert,expected_source_sha256=item['hz_sha256'])
        value=checked['bound']['checked_lower_bound'] if cert else None
        if value is not None and value!=item['checked_lower_bound']:raise ValueError('claimed metadata changed')
        proofs[key]={'kind':item['kind'],'scope':item['scope'],'property_index':item['property_index'],
                     'checked_lower_bound':value,'hz_sha256':item['hz_sha256'],'new':key not in parent['proofs']}
        if key in parent['proofs']:
            old=parent['proofs'][key]
            for field in ('export','certificate'):
                if old[field] and item[field]['sha256']!=old[field]['sha256']:raise ValueError('parent proof changed')
    checked=check_directory(root,expected_request_id=identity(frozen_request()))
    if checked!=json.loads((root/'check.json').read_text()):raise ValueError('aggregation mismatch')
    old_rows={(tuple(r['pair']),r['property_index']):r for r in parent['obligations']}
    comparisons=[]
    for row in manifest['obligations']:
        old=old_rows[(tuple(row['pair']),row['property_index'])]
        if old['kind']=='reused' and row!=old:raise ValueError('frozen reuse fact changed')
        if old['kind']=='residual':
            for field in ('difference_lower','difference_upper','difference_bounds'):
                if row.get(field)!=old[field]:raise ValueError('disagreement evidence changed')
            comparisons.append({'pair':row['pair'],'property_index':row['property_index'],
                                'old_lower_bound':parent['proofs'][old['source']]['checked_lower_bound'],
                                'new_lower_bound':proofs[row['source']]['checked_lower_bound'] if row['kind']=='residual' else None,
                                'lambda_bounds':row.get('lambda_bounds'),'kind':row['kind']})
    new_count=sum(p['new'] for p in proofs.values())
    if new_count>5:raise ValueError('query budget expanded')
    return {'classification':'SAME_REQUEST_ORDER_ONLY_PROOF_FOLLOWUP_NOT_PERFORMANCE',
            'launch':launch,'terminal':json.loads((root/'terminal.json').read_text()),'parent_manifest_sha256':PARENT_HASH,
            'check':checked,'new_lp_queries':new_count,'proof_count':len(proofs),'proofs':proofs,
            'comparisons':comparisons,'obligations':manifest['obligations'],
            'raw_root':str(root),'raw_hashes':{str(p.relative_to(root)):_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()},
            'audit':'PASS','issues':[],
            'scope':'Every supplied export/dual and complete obligation aggregation rechecked, including old negative evidence. Upstream translations remain trusted.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=build();save(args.output,result)
    print(json.dumps({'audit':result['audit'],'check':result['check'],'new_lp_queries':result['new_lp_queries']},indent=2))
