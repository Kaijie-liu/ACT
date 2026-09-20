"""Freeze the scientific protocol, explicitly not an executable launch gate."""
import argparse
from pathlib import Path
import sys

from lp_sandwich.check import strict_json
from soplex_fidelity.io import load_job,save,sha
from soplex_fidelity.run import ROOT,PARENT,READER,roster

FREEZE=ROOT/'docs/soplex_finite_comparison_v1_freeze.json'
PREFIX=Path('/data1/Kane/MOE/envs/soplex-8.0.3')
OUTPUT=ROOT/'data/moe/results/soplex_diagnostic_real_20260920_v1'


def ref(path):return dict(path=str(path),sha256=sha(path))


def prepare():
    if FREEZE.exists() or OUTPUT.exists():raise FileExistsError('new protocol/result identities required')
    evidence=ROOT/'docs/soplex_large_import_attempt001.json'
    review=ROOT/'docs/soplex_large_import_review_attempt001.json'
    controls=ROOT/'docs/soplex_fidelity_controls_attempt002.json'
    r=strict_json(evidence.read_bytes());v=strict_json(review.read_bytes());c=strict_json(controls.read_bytes())
    if any(a['status']!='PASS' for a in (r,v,c)) or v['issues'] or v['receipt_sha256']!=sha(evidence):
        raise ValueError('readback/control gates')
    if c['reader_sha256']!=sha(READER) or r['reader_sha256']!=sha(READER):raise ValueError('reader drift')
    for name,digest in c['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('control source drift')
    inv=strict_json((ROOT/'docs/soplex_compat_installation_v1.json').read_bytes())
    if (sha(PREFIX/'bin/soplex')!=inv['files']['bin/soplex']['sha256'] or
        sha(ROOT/'lp_sandwich/check.py')!=inv['historical_checker_sha256']):raise ValueError('sealed runtime drift')
    jobs=roster()
    for job in jobs:load_job(job)
    if [j['job_id'] for j in jobs]!=[row['job_id'] for row in r['rows']]:raise ValueError('roster mismatch')
    result=dict(schema='SOPLEX_FINITE_COMPARISON_PROTOCOL_V1',status='PROTOCOL_FROZEN_NOT_EXECUTED',
        execution_ready=False,launch_authorized=False,jobs=jobs,original_parent=ref(PARENT),
        protocol=ref(ROOT/'docs/soplex_finite_comparison_v1.md'),
        import_receipt=ref(evidence),import_review=ref(review),controls=ref(controls),
        runtime=dict(interpreter=sys.executable,soplex=ref(PREFIX/'bin/soplex'),
            reader=ref(READER),settings=ref(PREFIX/'src/settings/exact.set'),
            upstream_commit='13e2ab2467e0016d02116802ac4dc7a89560dbc1',
            installation=ref(ROOT/'docs/soplex_compat_installation_v1.json')),
        independent_checker=ref(ROOT/'lp_sandwich/check.py'),output=str(OUTPUT),
        policy=dict(total_seconds=300,proposal_seconds=218,work_seconds=298,
            workers=1,threads=1,nice=10,native_attempts=1,retry=False,resume=False,
            source_math_unchanged=True,offset='omit exact additive constant natively; original checker restores it',
            memory_address_space_bytes=8*1024**3,per_file_bytes=128*1024**2,
            point_file_bytes=64*1024**2,job_output_bytes=512*1024**2,output_rational_bits=4096,
            resource=dict(minimum_ram_gib=16,minimum_disk_gib=5,maximum_load_per_core=.5,
                          poll_seconds=30,wait_limit_seconds=86400),
            native_time='remaining absolute proposal window, not an additional per-call window',
            comparison='full original LP, native basis choice allowed; historical Python results descriptive only',
            primal_only=True,no_native_status_acceptance=True,network_verdict=False,
            failure='ERROR stops with later NOT_RUN_AFTER_ERROR; TIMEOUT/LIMIT/unresolved continue'),
        readiness_missing=['rational CLI-output admission controls','unified deadline/memory/output supervisor',
                           'partial evidence and terminal/cost audit','hash-bound execution addendum'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()},
        optimization_calls=0)
    save(FREEZE,result);return result


def review():
    f=strict_json(FREEZE.read_bytes())
    for key in ('original_parent','protocol','import_receipt','import_review','controls','independent_checker'):
        r=f[key]
        if sha(r['path'])!=r['sha256']:raise ValueError(key+' drift')
    for key in ('soplex','reader','settings','installation'):
        r=f['runtime'][key]
        if sha(r['path'])!=r['sha256']:raise ValueError(key+' drift')
    if f['jobs']!=roster():raise ValueError('selection drift')
    for job in f['jobs']:load_job(job)
    for name,digest in f['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source drift')
    if (f['status']!='PROTOCOL_FROZEN_NOT_EXECUTED' or f['execution_ready'] or
        f['launch_authorized'] or OUTPUT.exists() or not f['readiness_missing']):
        raise ValueError('freeze is not launch permission')
    out=dict(status='PASS',issues=[],freeze=ref(FREEZE),jobs=4,optimization_calls=0,
             execution_ready=False,scope='Protocol/input/runtime identity, not execution-supervisor validation.')
    save(ROOT/'docs/soplex_finite_comparison_v1_freeze_review.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','review']);a=p.parse_args()
    result=prepare() if a.action=='prepare' else review();print(result['status'])
