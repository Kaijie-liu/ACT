"""One <=300s PGD/ledger integration control, no trained-model inference."""
import time
ENTRY=time.monotonic()
import io
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
from scoped_proof.io import load,save,sha
from scoped_proof.owned_session import execute
from architecture_s1_execution import FILES,Artifacts,finite_queue,digest,THREAD_ENV,read_record
import genbab_cpu_ready as backend


def native(root,deadline):
    fd=os.open(os.devnull,os.O_WRONLY);os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
    try:
        backend.limits(deadline);backend.probe()
        from act.util.typing_compat import install_typing_override
        install_typing_override()
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
        import unittest
        import test_architecture_attack as m
        m.DEADLINE=deadline
        output=io.StringIO()
        names=load(root/'freeze.json').get('unit_tests')
        suite=(unittest.TestSuite(m.Controls(n) for n in names) if names
            else unittest.defaultTestLoader.loadTestsFromModule(m))
        result=unittest.TextTestRunner(stream=output,verbosity=2,failfast=True).run(
            suite)
        save(root/'unit.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
            details=output.getvalue(),clock=time.monotonic(),deadline=deadline))
        return 0 if result.wasSuccessful() else 1
    except BaseException as exc:
        save(root/'error.json',dict(kind=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        return 1


def run(root,previous=None):
    root.mkdir()
    files=FILES+('scripts/architecture_s1_public_control.py','tests/test_architecture_attack.py','scripts/multilayer_stage0.py')
    freeze=dict(schema='S1_PUBLIC_PGD_CONTROL_V1',synthetic_control=True,source_files={p:sha(ROOT/p) for p in files},
        backend_sources_sha256=digest(backend.source_identity()),batch_seconds=300,
        cases=['lossless endpoint/interior witnesses','corrupt witness rejection','no-witness is not safe',
            'radius step and legacy default differential','center replay','deadline',
            'complete 24000-domain PGD roster','physical input helper','two-worker PGD/verification integration'],
        trained_calls=0)
    if previous is not None:
        prior=load(previous/'unit.json')
        if prior['status']!='FAIL' or prior['tests']!=6:
            raise ValueError('preserved R1 fixture-expectation failure')
        freeze.update(prior_control=dict(root=str(previous),unit_sha256=sha(previous/'unit.json'),
            complete_sha256=sha(previous/'complete.json'),unchanged_passed_unit_cases=5),
            unit_tests=['test_same_witness_seed_and_old_default','test_zero_failure_is_not_safety'])
    save(root/'freeze.json',freeze)
    env=dict(os.environ,**THREAD_ENV)
    session=execute([str(backend.ENVIRONMENT/'bin/python'),'-I','-B',str(Path(__file__).resolve()),
        'native',str(root),str(ENTRY+90)],run_deadline=ENTRY+94,cleanup_deadline=ENTRY+98,
        env=env,rss_limit=None,development_policy=backend.POLICY)
    save(root/'unit_session.json',session)
    unit=load(root/'unit.json') if (root/'unit.json').exists() else {}
    if session['status']!='COMPLETED' or unit.get('status')!='PASS':
        save(root/'complete.json',dict(status='FAIL',phase='unit',seconds=time.monotonic()-ENTRY));return 1
    store=Artifacts.create(root/'requests')
    requests=[dict(id='pgd_attack',case='attack',mode='pgd',indices=[0,1],radius=2),
        dict(id='pgd_zero',case='zero',mode='pgd',indices=[0,1],radius=8),
        dict(id='dense_complete',case='positive',mode='complete'),
        dict(id='moe_incomplete',case='tie',mode='incomplete')]
    summary=finite_queue(store.root,requests,root/'freeze.json',workers=2,seconds=60,timing_reserve=(5,3,1))
    ledger=[read_record(store.root/(r['id']+'.ledger.json.gz')) for r in requests]
    ok=summary['status']=='COMPLETE' and [r['status'] for r in ledger]==[
        'AUXILIARY_COMPLETE','AUXILIARY_COMPLETE','POSITIVE','POSITIVE']
    for r in ledger:
        terminal=store.root/r['id']/'terminal.json.gz'
        ok &= sha(terminal)==r['terminal_sha256'] and read_record(terminal)['status']==r['status']
    physical=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    ok &= time.monotonic()<ENTRY+300 and physical<128*2**20
    save(root/'complete.json',dict(status='PASS' if ok else 'FAIL',tests=unit['tests'],
        native_requests=len(requests),auxiliary_domains=4,seconds=time.monotonic()-ENTRY,
        source_files=freeze['source_files'],physical_bytes=physical,summary=summary,
        trained_calls=0,artifacts={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}))
    print(json.dumps(dict(status='PASS' if ok else 'FAIL',tests=unit['tests'],native_requests=4,
        seconds=time.monotonic()-ENTRY)))
    return 0 if ok else 1


if __name__=='__main__':
    if sys.argv[1]=='native':raise SystemExit(native(Path(sys.argv[2]),float(sys.argv[3])))
    raise SystemExit(run(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)>2 else None))
