"""Fresh exact residual checks and relocated original-LP checker; no elimination."""
import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from single_check_portable.execution import ROOT,ACT,read,save_new
from portable_proof.runtime import digest
from modular_basis.controls import sources,old


def compare_checked(checked,reference):
    execution={'seconds','isolated','site_disabled','solver_or_model_imported'}
    if (set(checked)!=set(reference)|execution or checked['isolated'] is not True or
            checked['site_disabled'] is not True or checked['solver_or_model_imported'] is not False or
            not math.isfinite(checked['seconds']) or not 0<=checked['seconds']<=30):
        raise ValueError('isolated execution contract')
    if {k:v for k,v in checked.items() if k not in execution}!=reference:
        raise ValueError('isolated exact original-LP check differs')


def review(receipt,destination):
    begin=time.monotonic();old();c=read(receipt)
    if c['status']!='PASS' or c['sources']!=sources():raise ValueError('current controls required')
    base=Path(c['control_artifact_root'])
    actual={str(p.relative_to(base)):digest(p.read_bytes()) for p in sorted(base.rglob('*')) if p.is_file()}
    if actual!=c['artifact_sha256']:raise ValueError('control evidence drift')
    solved=unresolved=0
    for path in sorted(base.glob('*/system.json')):
        s=read(path);r=read(path.parent/'result.json')
        if r['status']=='CANDIDATE_ONLY':
            x=list(map(F,r['solution']));m=len(s['rhs'])
            if len(x)!=m or len(s['rows'])!=m:raise ValueError('system dimensions')
            for row,rhs in zip(s['rows'],s['rhs']):
                if any(not 0<=j<m for j,_ in row) or sum((F(v)*x[j] for j,v in row),F(0))!=F(rhs):
                    raise ValueError('original exact system residual')
            ref=path.parent/'reference.json'
            if ref.exists() and 'old_solution' in read(ref) and list(map(F,read(ref)['old_solution']))!=x:
                raise ValueError('differential result')
            solved+=1
        else:
            if r['solution'] is not None or r['network_SAFE'] or r['network_UNSAFE']:
                raise ValueError('unresolved system promoted')
            unresolved+=1
    moved=Path(tempfile.mkdtemp(prefix='modular_basis_review_',dir=ROOT/'data/moe/results'))
    checks=[]
    for bundle in sorted(base.glob('*/bundle.json')):
        if bundle.parent.name=='moved':continue
        proposal=read(bundle.parent/'proposal.json');b=read(bundle)
        source=read(bundle.parent/'input.json')
        if proposal['bundle']!=b or b['lp']!=source['lp'] or b['statement']!=source['statement']:
            raise ValueError('original LP identity')
        dest=moved/bundle.parent.name;dest.mkdir()
        shutil.copyfile(bundle,dest/'bundle.json');shutil.copyfile(ROOT/'lp_sandwich/check.py',dest/'verify.py')
        process=subprocess.run([ACT,'-I','-S',str(dest/'verify.py'),str(dest/'bundle.json'),
              '--bundle-sha256',digest((dest/'bundle.json').read_bytes()),
              '--statement-sha256',proposal['statement_sha256'],'--timeout-seconds','30'],
              cwd=dest,capture_output=True,text=True,check=True,timeout=35)
        checked=json.loads(process.stdout);reference=read(bundle.parent/'check_reference.json')
        compare_checked(checked,reference)
        save_new(dest/'check.json',checked);checks.append({'case':bundle.parent.name,'checked':checked})
    old()
    if c['sources']!=sources():raise ValueError('review source drift')
    out={'status':'PASS','issues':[],'receipt_sha256':digest(receipt.read_bytes()),'sources':sources(),
        'artifact_files':len(actual),'exact_system_checks':solved,'unresolved_systems_preserved':unresolved,
        'moved_isolated_checks':checks,'moved_root':str(moved),
        'moved_artifact_sha256':{str(p.relative_to(moved)):digest(p.read_bytes())
                                for p in sorted(moved.rglob('*')) if p.is_file()},
        'seconds':time.monotonic()-begin,'real_LP_reconstructions':0,'new_eliminations':0,'native_calls':0,
        'scope':'analytic exact residuals and full original-LP feasibility only, no real-network proof'}
    save_new(destination,out)
    return {k:out[k] for k in ('status','issues','artifact_files','exact_system_checks','unresolved_systems_preserved')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(review(a.controls,a.output),indent=2))
