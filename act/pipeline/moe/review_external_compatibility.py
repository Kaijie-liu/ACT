"""Read-only provenance/result audit for the separate frozen frontend probe."""
import argparse
import json
from pathlib import Path
from act.pipeline.moe.external_compatibility import ROOT, TOOL, COMMITS, CASES, git, sha, dump


def build():
    root=ROOT/'data/moe/results/external_compatibility_20260914_r1'
    summary=json.loads((root/'summary.json').read_text());launch=summary['launch']
    if tuple(launch['cases'])!=CASES or launch['commits']!=COMMITS: raise ValueError('frozen cases/source mismatch')
    if set(summary['results'])!=set(CASES): raise ValueError('case omission')
    for name,h in summary['raw_hashes'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or sha(p)!=h: raise ValueError('raw file drift')
    for path,commit in COMMITS.items():
        if git(path,'rev-parse','HEAD')!=commit or git(path,'status','--porcelain'): raise ValueError('tool drift')
    for name,h in launch['anchors'].items():
        if sha(TOOL/name)!=h: raise ValueError('source anchor drift')
    # Validate the ACT worker at execution commit, not mutable current sources.
    import subprocess,hashlib
    original=subprocess.check_output(['git','-C',str(ROOT),'show',launch['execution_head']+':act/pipeline/moe/external_compatibility.py'])
    if hashlib.sha256(original).hexdigest()!=launch['worker_sha256']: raise ValueError('worker drift')
    for case,result in summary['results'].items():
        if result!=json.loads((root/(case+'.json')).read_text()): raise ValueError('summary mismatch')
        if result.get('formal_SAFE',False): raise ValueError('unsupported formal label')
        if 'auto_lirpa_file' in result and not Path(result['auto_lirpa_file']).is_relative_to(TOOL/'auto_LiRPA'):
            raise ValueError('wheel substituted for pinned source')
        if result['status']=='NUMERICAL_BOUND_RETURNED_NOT_FORMAL_SAFE':
            lo,hi=result['bounds']['lower'][0][0],result['bounds']['upper'][0][0]
            if not lo<=result['grid_range'][0]<=result['grid_range'][1]<=hi: raise ValueError('bound/grid inconsistency')
            if result['max_finite_probe_error']!=0: raise ValueError('finite probe disagreement')
    return {'classification':'PINNED_FRONTEND_COMPATIBILITY_NOT_EXTERNAL_BENCHMARK',
            'audit':'PASS','issues':[],'scope':'Provenance, recorded phase and finite-probe consistency; not independent verification of numerical bounds.',
            'summary_sha256':sha(root/'summary.json'),'raw_root':str(root),
            'unsupported_operation_log':[line for line in (root/'dynamic_top2.log').read_text().splitlines()
                                         if 'Name: onnx::' in line],**summary}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=build();dump(args.output,result)
    print(result['audit'],len(result['issues']))
