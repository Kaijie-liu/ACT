"""Read-only reconstruction of the fixed ACT-only rational proof experiment."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from act.back_end.solver.lp_certificate import identity
from act.back_end.solver.check_hz_lp_export import check_export
from act.back_end.solver.check_rational_mccormick import check_construction
from act.pipeline.moe.check_request_lp import check_directory
from act.pipeline.moe.request_lp_cases import selected_request, sha, save

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT = PROJECT/'data/moe/results/request_lp_act_only_20260915_r1'
ARCHIVE = Path(__file__).parent/'results/request_lp_act_only_review_20260915_r1.json'


def review(root):
    root=Path(root).resolve(); runtime=json.loads((root/'runtime.json').read_text())
    cfg=runtime['config']
    config_path=PROJECT/'act/pipeline/moe/configs/request_lp_act_only_r1.json'
    if sha(config_path)!=runtime['config_sha256'] or json.loads(config_path.read_text())!=cfg:
        raise ValueError('protocol drift')
    if len(runtime['cases'])!=3 or runtime['borrowed_computed_proof_facts']!=0:
        raise ValueError('incomplete case ledger')
    result={'execution_head':runtime['head'],'classification':'POST_SELECTED_FIXED_MECHANISM_CASES',
        'borrowed_computed_proof_facts':0,'raw_input_preparation_prior_all_ten_seconds':0.2768000243231654,
        'cases':[],'issues':[]}
    for case,entry in zip(cfg['cases'],runtime['cases']):
        if case!=entry['case']:raise ValueError('case order/substitution')
        directory=root/f"{case['model']}_{case['dataset_index']}"
        request,_=selected_request(cfg,case,PROJECT); rid=identity(request)
        job=json.loads((directory/'job.json').read_text())
        terminal=json.loads((directory/'terminal.json').read_text())
        if job['request']!=request or job['request_id']!=rid or terminal!={k:v for k,v in entry.items() if k!='case'}:
            raise ValueError('request/terminal ledger mismatch')
        if terminal['status']!='COMPLETED' or terminal['returncode']!=0:
            raise ValueError('this archive expects three completed generations, not missing attempts')
        checked=check_directory(directory,expected_request_id=rid)
        if checked!=terminal['check'] or checked!=json.loads((directory/'check.json').read_text()):
            raise ValueError('independent request check mismatch')
        manifest=json.loads((directory/'manifest.json').read_text())
        proof_values={}; binaries={}
        for key,item in manifest['proofs'].items():
            def read(ref):
                path=(directory/ref['file']).resolve()
                if not path.is_relative_to(directory) or sha(path)!=ref['sha256']:
                    raise ValueError('proof reference drift')
                return json.loads(path.read_text())
            record=read(item['export']); cert=read(item['certificate']) if item['status']=='CHECKED' else None
            if item['kind']=='rational_weighted':
                r=check_construction(record,cert,source_hash=item['hz_sha256'],q=record['q'],
                    offset=0,gate=record['gate'],difference=record['difference'])
            else:r=check_export(record,cert,expected_source_sha256=item['hz_sha256'])
            value=Fraction(r['bound']['checked_lower_bound']) if cert else None
            if value is not None and str(value)!=item['checked_lower_bound']:
                raise ValueError('stored bound differs from independent arithmetic')
            proof_values[key]=value; binaries[key]=record['n_relaxed_binaries']
        rows=[]
        for row in manifest['obligations']:
            keys=row.get('sources',[]) if row['kind']=='reused' else [row['source']] if row['kind']=='residual' else []
            vals=[proof_values[k] for k in keys]
            value=min(vals) if vals and all(v is not None for v in vals) else None
            rows.append({'pair':row['pair'],'property_index':row['property_index'],'kind':row['kind'],
                'checked_lower_bound':str(value) if value is not None else None,
                'positive':value is not None and value>Fraction.from_float(1e-7),
                'source_proofs':keys,'source_relaxed_binaries':[binaries[k] for k in keys]})
        if sum(r['positive'] for r in rows)!=checked['required']-checked['counts']['unknown']:
            raise ValueError('obligation count mismatch')
        result['cases'].append({'model':case['model'],'dataset_index':case['dataset_index'],
            'request_id':rid,'routes':manifest['routes']['feasible'],'check':checked,'obligations':rows,
            'proof_exports_independently_checked':len(proof_values),
            'generation_seconds':terminal['generation_seconds'],
            'independent_check_seconds':terminal['independent_check_seconds'],
            'raw_directory_bytes':sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()),
            'source_relaxed_binary_counts':sorted(set(binaries.values()))})
    result['raw_hashes']={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    result['status']='PASS'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args(); value=review(args.root)
    if args.write:save(ARCHIVE,value)
    elif value!=json.loads(ARCHIVE.read_text()):raise ValueError('archive reconstruction differs')
    print(json.dumps({'status':value['status'],'cases':[{k:v for k,v in c.items() if k not in ('obligations','request_id')}
                                                      for c in value['cases']]},indent=2))
