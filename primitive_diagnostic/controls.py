"""Retain batch tests and independently replay their ledgers; no real requests."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from single_check_portable.execution import ROOT,ACT,read,save_new
from portable_proof.runtime import digest
from primitive_diagnostic import contract as C
from primitive_diagnostic.run import audit_saved


def inventory(base):
    return {str(p.relative_to(base)):digest(p.read_bytes()) for p in sorted(base.rglob('*')) if p.is_file()}


def run():
    C.verify_sealed();before=C.sources();n=1;begin=time.monotonic();log=io.StringIO()
    while (ROOT/f'docs/primitive_diagnostic_controls_attempt{n:03}.json').exists():n+=1
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'primitive_diagnostic.tests','sparse_diagnostic_archive.tests']))
    C.verify_sealed()
    if before!=C.sources():raise ValueError('control source drift')
    from primitive_diagnostic.tests import ARTIFACT_ROOT,JOBS,VALID_BATCHES
    status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    records=[{'root':p,'summary':audit_saved(JOBS,Path(p))} for p in VALID_BATCHES] if status=='PASS' else []
    destination=ROOT/f'docs/primitive_diagnostic_controls_attempt{n:03}.json'
    save_new(destination,{'status':status,'sources':before,'tests_run':result.testsRun,'log':log.getvalue(),
        'seconds':time.monotonic()-begin,'artifact_root':str(ARTIFACT_ROOT),'artifact_sha256':inventory(ARTIFACT_ROOT),
        'jobs':JOBS,'batches':records,'integration':C.ref(C.INTEGRATION),'integration_review':C.ref(C.INTEGRATION_REVIEW),
        'real_LP_solves':0,'real_reconstructions':0,
        'scope':'analytic LP batch, accelerated synthetic timeout, archive/launch mutations; not efficacy'})
    print(log.getvalue());print(destination,status,flush=True)
    if status!='PASS':raise SystemExit(1)


def review(controls,destination):
    if destination.exists():raise FileExistsError('retain prior review')
    begin=time.monotonic();C.verify_sealed();c=read(controls)
    if c['status']!='PASS' or c['sources']!=C.sources():raise ValueError('current passing controls required')
    base=Path(c['artifact_root'])
    if inventory(base)!=c['artifact_sha256']:raise ValueError('control artifact changed')
    moved=Path(tempfile.mkdtemp(prefix='primitive_diagnostic_review_',dir=ROOT/'data/moe/results'))
    checks=[]
    for i,b in enumerate(c['batches']):
        folder=Path(b['root']);summary=audit_saved(c['jobs'],folder)
        if summary!=b['summary']:raise ValueError('fresh cost/terminal mismatch')
        for row in summary['rows']:
            if not row['complete_independent_check']:continue
            source=folder/row['job_id'];target=moved/(str(i)+'_'+row['job_id'])
            shutil.copytree(source/'portable',target);packing=read(source/'packing.json')
            proc=subprocess.run([ACT,'-I','-S',str(target/'verify.py'),str(target/'bundle.json'),
                '--bundle-sha256',packing['bundle_sha256'],'--statement-sha256',packing['statement_sha256'],
                '--timeout-seconds','10'],cwd=target,capture_output=True,text=True,check=True,timeout=15)
            result=json.loads(proc.stdout);expected=read(source/'check.log')
            if {k:v for k,v in result.items() if k!='seconds'}!={k:v for k,v in expected.items() if k!='seconds'}:
                raise ValueError('independent original-LP outcome changed')
            save_new(target/'review.json',result);checks.append(result)
    C.verify_sealed()
    if C.sources()!=c['sources']:raise ValueError('review source drift')
    result={'status':'PASS','issues':[],'controls':C.ref(controls),'sources':C.sources(),
        'artifact_files':len(c['artifact_sha256']),'batch_ledgers':len(c['batches']),
        'terminal_rows':sum(len(b['summary']['rows']) for b in c['batches']),
        'moved_checks':checks,'moved_root':str(moved),'moved_sha256':inventory(moved),
        'seconds':time.monotonic()-begin,'new_native_or_reconstruction_calls':0}
    save_new(destination,result)
    return {k:result[k] for k in ('status','issues','artifact_files','batch_ledgers','terminal_rows')}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.review:print(json.dumps(review(a.review,a.output),indent=2))
    else:run()
