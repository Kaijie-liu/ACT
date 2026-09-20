"""Execution-only addendum; the original scientific freeze remains immutable."""
import argparse
from pathlib import Path
import subprocess
import time

from lp_sandwich.check import strict_json
from soplex_fidelity.io import save,sha
from soplex_execution.runtime import ROOT,PYTHON
from soplex_execution.supervisor import FREEZE,ADDENDUM,verify_freeze
from soplex_execution.audit import audit_job


def prepare(raw,attempt):
    if ADDENDUM.exists():raise FileExistsError(ADDENDUM)
    f=strict_json(FREEZE.read_bytes());r=strict_json((raw/'controls.json').read_bytes())
    if Path(f['output']).exists():raise FileExistsError('real result identity already used')
    if r['status']!='PASS' or r['failures'] or r['errors'] or r['tests']!=18 or r['real_queries'] or r['analytic_native_queries']!=10:
        raise ValueError('controls gate')
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('tested source drift')
    for name,digest in r['artifacts'].items():
        if sha(raw/name)!=digest:raise ValueError('control artifact drift')
    for name,digest in f['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('historical source drift')
    controls=ROOT/f'docs/soplex_execution_controls_attempt{attempt}.json';save(controls,r)
    start=time.monotonic()
    p=subprocess.run([str(PYTHON),'-m','unittest','lp_sandwich.tests'],cwd=ROOT,capture_output=True,text=True,timeout=60)
    regression=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-start,
                    analytic_scipy_queries=4,real_queries=0,checker_sha256=sha(ROOT/'lp_sandwich/check.py'))
    regression_path=ROOT/f'docs/soplex_execution_checker_regressions_attempt{attempt}.json'
    save(regression_path,regression)
    if p.returncode or 'Ran 21 tests' not in p.stderr:raise ValueError('original checker regressions')
    paths=set(Path(__file__).parent.glob('*.py'))
    paths.update(ROOT/name for name in f['sources'])
    paths.update((ROOT/'evidence_cohort/ownership.py',
                  ROOT/'lp_sandwich/__init__.py',ROOT/'lp_sandwich/check.py',FREEZE,
                  ROOT/'docs/soplex_execution_v1.md',controls,regression_path))
    for k in ('soplex','reader','settings','installation'):paths.add(Path(f['runtime'][k]['path']))
    for k in ('original_parent','protocol','import_receipt','import_review','controls','independent_checker'):
        ref=f[k]
        if sha(ref['path'])!=ref['sha256']:raise ValueError('original protocol dependency drift')
        paths.add(Path(ref['path']))
    for job in f['jobs']:
        if sha(job['export']['path'])!=job['export']['sha256']:raise ValueError('original input drift')
    result=dict(schema='SOPLEX_EXECUTION_ADDENDUM_V1',execution_ready=True,
        launch_authorized_by='User: 补齐外层监督、候选输出接收与终态成本审计，再执行。',
        protocol_sha256=sha(FREEZE),bindings={str(p):sha(p) for p in sorted(paths)},
        controls_path=str(controls),regressions_path=str(regression_path),preparation_attempt=attempt,
        policy=f['policy'],jobs=[j['job_id'] for j in f['jobs']],output=f['output'],
        execution_changes='Only supervised execution/admission/accounting; no scientific changes.',
        controls=dict(new=18,unchanged_checker=21,analytic_soplex_queries=10,analytic_scipy_queries=4,real_queries=0),
        candidate_contract='Complete exact named rational primal only, original LP independent check; no native status trust.')
    save(ADDENDUM,result);return result


def review():
    start=time.monotonic();f,a=verify_freeze();c=strict_json(Path(a['controls_path']).read_bytes())
    raw=Path(c['raw_root']);root=raw/'release_review';root.mkdir()
    runtime={k:f['runtime'][k] for k in ('soplex','reader','settings')};runtime['checker']=f['independent_checker']
    rows=[]
    for row in c['native_rows']:
        spec=strict_json((raw/row['case']/'spec.json').read_bytes())
        v=audit_job(spec['job'],raw/row['case'],root/row['case'],f['policy'],runtime)
        if v['upper_bound']!=row['upper_bound']:raise ValueError('control upper disagreement')
        rows.append(dict(case=row['case'],status=v['status'],upper_bound=v['upper_bound']))
    result=dict(status='PASS',issues=[],execution_ready=True,addendum_sha256=sha(ADDENDUM),
                rows=rows,seconds=time.monotonic()-start,native_queries=0,real_queries=0,
                scope='Frozen execution identities, 10 fresh isolated analytic LP checks, terminal and cost audit; not real-study results.')
    save(ROOT/'docs/soplex_execution_v1_readiness_review.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','review']);p.add_argument('--controls',type=Path)
    p.add_argument('--attempt',default='002',choices=['002','003'])
    a=p.parse_args();result=prepare(a.controls,a.attempt) if a.action=='prepare' else review();print(result.get('status','FROZEN'))
