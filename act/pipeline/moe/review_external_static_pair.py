"""Provenance and numerical-record audit, NOT an independent CROWN bound proof."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from act.pipeline.moe.external_static_pair import ROOT, TOOL, COMMITS, PAIRS, request_identity, git, sha, dump


def validate(summary):
    request,rid=request_identity();launch=summary['launch']
    if launch['request']!=request or launch['request_id']!=rid or launch['commits']!=COMMITS:
        raise ValueError('request/tool identity changed')
    if launch['pairs']!=[list(p) for p in PAIRS]:raise ValueError('pair inventory changed')
    expected={'method':'CROWN','bound_opts':{'conv_mode':'matrix'},'dtype':'float64','device':'cpu'}
    if launch['config']!=expected or launch['timeout_each_seconds']!=120:raise ValueError('frozen config changed')
    results=summary['results']
    if [r['pair'] for r in results]!=[list(p) for p in PAIRS]:raise ValueError('missing/duplicate/reordered pair')
    positive=0;completed=0
    C=[]
    for competitor in range(10):
        if competitor==5:continue
        row=[0]*10;row[5]=1;row[competitor]=-1;C.append(row)
    for r in results:
        if r.get('formal_SAFE') is not False:raise ValueError('unsupported formal SAFE label')
        if r['status']!='NUMERICAL_BOUND_RETURNED_NOT_FORMAL_SAFE':continue
        if r['request']!=request or r['request_id']!=rid or r['C']!=[C]:raise ValueError('property/input changed')
        if r['dtype']!='float64' or r['device']!='cpu':raise ValueError('numeric semantics changed')
        if not Path(r['auto_lirpa_file']).is_relative_to(TOOL/'auto_LiRPA'):raise ValueError('wrong frontend')
        if r['forced_branch_max_error']>1e-10 or r['lowered_max_error']>1e-10:raise ValueError('conformance failure')
        lower,upper=r['bounds']['lower'],r['bounds']['upper']
        if len(lower)!=9 or len(upper)!=9 or not all(math.isfinite(v) for v in lower+upper):
            raise ValueError('incomplete/nonfinite bounds')
        if len(r['concrete_margins'])!=5:raise ValueError('probe omission')
        for probe in r['concrete_margins']:
            if len(probe)!=9 or any(not math.isfinite(v) for v in probe):raise ValueError('invalid probe')
            if any(not lo-1e-10<=v<=hi+1e-10 for lo,v,hi in zip(lower,probe,upper)):
                raise ValueError('bound/probe inconsistency')
        count=sum(v>1e-7 for v in lower)
        if count!=r['positive_rows'] or r['required_rows']!=9:raise ValueError('row counts incorrect')
        positive+=count;completed+=9
    expected_status=('ALL_LISTED_STATIC_OBLIGATIONS_NUMERICALLY_POSITIVE' if completed==18 and positive==18
                     else 'NOT_ALL_STATIC_OBLIGATIONS_POSITIVE')
    if (summary['positive_rows']!=positive or summary['required_rows']!=18
            or summary['status']!=expected_status or summary['formal_SAFE'] is not False):
        raise ValueError('aggregate inflation')
    return {'completed_rows':completed,'positive_rows':positive,'required_rows':18,'status':expected_status}


def build():
    root=ROOT/'data/moe/results/external_static_pair_20260914_r2'
    summary=json.loads((root/'summary.json').read_text());launch=summary['launch']
    for name,h in summary['raw_hashes'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=h:raise ValueError('raw artifact changed')
    for path,commit in COMMITS.items():
        if git(path,'rev-parse','HEAD')!=commit or git(path,'status','--porcelain'):raise ValueError('external source drift')
    for name,key in [('act/pipeline/moe/external_static_pair.py','worker_sha256'),('act/back_end/moe/static_pair.py','adapter_sha256')]:
        source=subprocess.check_output(['git','-C',str(ROOT),'show',launch['execution_head']+':'+name])
        if hashlib.sha256(source).hexdigest()!=launch[key]:raise ValueError('execution source drift')
    subject=json.loads((ROOT/'act/pipeline/moe/configs/schedule_confirmation_selection_r2.json').read_text())['models']['seed0']
    for r in summary['results']:
        if r!=json.loads((root/f"pair_{r['pair'][0]}_{r['pair'][1]}.json").read_text()):raise ValueError('summary differs from raw')
        if 'model_state' in r and r['model_state']!=subject['model_state']:raise ValueError('model-state changed')
    aggregate=validate(summary)
    return {'classification':'EXTERNAL_REAL_MODEL_STATIC_WHOLE_BOX_CONTROL_NOT_FULL_SYSTEM_BENCHMARK',
            'audit':'PASS','issues':[],'aggregate_recomputed':aggregate,'raw_root':str(root),
            'summary_sha256':sha(root/'summary.json'),
            'scope':'Source,request,property,finite-conformance and recorded-bound consistency audit; NOT independent outward-rounded CROWN proof.',**summary}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=build();dump(args.output,result)
    print(json.dumps({'audit':result['audit'],'aggregate':result['aggregate_recomputed']},indent=2))
