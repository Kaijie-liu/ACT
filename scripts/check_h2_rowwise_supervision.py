"""Combine frozen call re-audit with the separately recorded control outcome.

No solver/model execution. PASS means this fixed control stage passed, not real
capacity, native-float safety or an independent proof that unit tests executed.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,sha
from scripts.archive_h2_rowwise_supervision import derive


def require_outcome(value):
    expected={'tests':18,'success':True,'failures':[],'errors':[],'skipped':[],'real_requests':0}
    if (value!=expected or type(value.get('tests')) is not int or
            type(value.get('success')) is not bool or type(value.get('real_requests')) is not int):
        raise ValueError('complete successful frozen control outcome required')


def check(root):
    outcome_path=root/'test_outcome.json'; digest=sha(outcome_path)
    outcome=load(outcome_path,digest); require_outcome(outcome)
    archive=derive(root)  # full 38-call roster, identities, cost, math, common sources
    if sha(outcome_path)!=digest: raise ValueError('control outcome changed during audit')
    return {'schema':'HR_SUPERVISION_ACCEPTANCE_V1','status':'PASS','archive':archive,
            'control_outcome':outcome,'control_outcome_sha256':digest,
            'acceptance_checker_sha256':sha(Path(__file__)),
            'real_capacity_admitted':False,'new_real_certificate':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path); group.add_argument('--check',type=Path)
    a=p.parse_args(); result=check(a.root)
    if a.output:
        if not a.output.resolve().is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('output outside project')
        with a.output.open('x') as stream: json.dump(result,stream,sort_keys=True,indent=2); stream.write('\n')
    elif load(a.check)!=result: raise ValueError('acceptance archive changed')
    print(json.dumps({'status':'PASS','controls':result['control_outcome']['tests'],
        'calls':result['archive']['call_count'],'rechecked_packages':len(result['archive']['mathematical_rechecks']),
        'statuses':result['archive']['statuses'],'real_capacity_admitted':False},sort_keys=True))
