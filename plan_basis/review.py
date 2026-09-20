"""Fresh saved-equation and relocated original-LP checks, no new elimination."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from single_check_portable.execution import ROOT,ACT,read,save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity
from modular_basis.review import compare_checked
from .controls import sealed,sources


def check_system(s,r):
    if r['network_SAFE'] or r['network_UNSAFE']:raise ValueError('network overclaim')
    if r['status']!='CANDIDATE_ONLY':
        if r['solution'] is not None:raise ValueError('partial point promoted')
        return False
    x=list(map(F,r['solution']));m=len(s['rhs'])
    if len(x)!=m or len(s['rows'])!=m:raise ValueError('system dimension')
    for row,rhs in zip(s['rows'],s['rhs']):
        if any(type(j) is not int or not 0<=j<m for j,_ in row):raise ValueError('column')
        if sum((F(v)*x[j] for j,v in row),F(0))!=F(rhs):raise ValueError('exact equation residual')
    return True


def check_diagnostics(r):
    if sum(r['costs']['operations'].values())!=r['operations']:raise ValueError('operation accounting')
    for row in r['stats']['rounds']:
        progress=row.get('reconstruction_progress')
        if progress is None:continue
        n=progress['successful_prefix'];failed=progress['failed_coordinate']
        if type(n) is not int or n<0:raise ValueError('prefix count')
        if progress['complete_vector']:
            if failed is not None or progress['status'] not in ('EXACT_SYSTEM_RESIDUAL_ZERO','REJECTED_BY_EXACT_RESIDUAL','AWAITING_EXACT_RESIDUAL'):
                raise ValueError('complete progress')
        elif failed!=n or progress['status'] not in ('INCOMPLETE','INTERRUPTED'):
            raise ValueError('partial progress')


def review(receipt,output):
    begin=time.monotonic();sealed();r=read(receipt)
    if r['status']!='PASS' or r['sources']!=sources():raise ValueError('current passing controls required')
    roots=[Path(p) for p in r['artifact_roots']]
    actual={str(p.relative_to(ROOT)):digest(p.read_bytes()) for root in roots for p in sorted(root.rglob('*')) if p.is_file()}
    if actual!=r['artifact_sha256']:raise ValueError('source artifact mutation')
    solved=unresolved=0;diffs=0
    for root in roots:
        for path in sorted(root.glob('*/system.json')):
            s=read(path);result=read(path.parent/'result.json')
            success=check_system(s,result);solved+=success;unresolved+=not success
            if root==roots[0]:
                check_diagnostics(result)
                ref=path.parent/'reference.json'
                if ref.exists():
                    for name,v in read(ref).items():
                        check_system(s,v)
                        if v['solution']!=result['solution']:raise ValueError('on/off/frozen differential')
                        diffs+=1
    base=Path(tempfile.mkdtemp(prefix='plan_basis_review_',dir=ROOT/'data/moe/results'))
    checks=[]
    for i,root in enumerate(roots):
        for bundle in sorted(root.glob('*/bundle.json')):
            reference=bundle.parent/'check_reference.json'
            if not reference.exists():reference=bundle.parent/'checked.json'
            if not reference.exists():reference=bundle.parent/'check.json'
            if not reference.exists():raise ValueError('missing original checker record')
            ref=read(reference)
            # Stored isolated outputs include execution flags; independently
            # compare mathematical fields and recheck the new invocation flags.
            ref={k:v for k,v in ref.items() if k not in ('seconds','isolated','site_disabled','solver_or_model_imported')}
            dest=base/f'{i}_{bundle.parent.name}';dest.mkdir()
            shutil.copyfile(bundle,dest/'bundle.json');shutil.copyfile(ROOT/'lp_sandwich/check.py',dest/'verify.py')
            data=read(bundle)
            run=subprocess.run([ACT,'-I','-S',str(dest/'verify.py'),str(dest/'bundle.json'),
                  '--bundle-sha256',digest((dest/'bundle.json').read_bytes()),'--statement-sha256',identity(data['statement']),
                  '--timeout-seconds','30'],cwd=dest,capture_output=True,text=True,check=True,timeout=35)
            checked=json.loads(run.stdout);compare_checked(checked,ref)
            save_new(dest/'checked.json',checked)
            checks.append({'case':dest.name,'result':checked})
    # Independent reviewer mutation controls operate on copies, not artifacts.
    s={'rows':[[[0,'3']]],'rhs':['1']}
    ok={'status':'CANDIDATE_ONLY','solution':['1/3'],'network_SAFE':False,'network_UNSAFE':False}
    if not check_system(s,ok):raise ValueError('review positive control')
    rejected=0
    for change in ({'solution':['0']},{'network_SAFE':True},{'status':'LIMIT'}):
        try:check_system(s,{**ok,**change})
        except ValueError:rejected+=1
    if rejected!=3:raise ValueError('review mutation controls')
    sealed()
    if r['sources']!=sources():raise ValueError('review source drift')
    result={'status':'PASS','issues':[],'receipt':str(receipt),'receipt_sha256':digest(receipt.read_bytes()),
            'sources':sources(),'artifact_files':len(actual),'exact_system_checks':solved,
            'unresolved_preserved':unresolved,'new_arm_differentials':diffs,'mutation_rejections':rejected,
            'moved_checks':checks,'moved_root':str(base),
            'moved_hashes':{str(p.relative_to(base)):digest(p.read_bytes()) for p in sorted(base.rglob('*')) if p.is_file()},
            'seconds':time.monotonic()-begin,'real_LP_calls':0,'new_eliminations':0,'native_calls':0,
            'scope':'exact saved-system residuals and original LP checks, not proof of every internal field operation'}
    save_new(output,result)
    print(json.dumps({k:result[k] for k in ('status','issues','artifact_files','exact_system_checks','unresolved_preserved','new_arm_differentials')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();review(a.controls,a.output)
