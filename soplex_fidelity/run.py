"""Read-only native import study. This module cannot launch LP optimization."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from lp_sandwich.check import strict_json,deadline_tick
from soplex_fidelity.io import load_job,save,sha,write_lp,verify_readback

ROOT=Path(__file__).resolve().parents[1]
READER=Path('/data1/Kane/MOE/envs/soplex-8.0.3/bin/read_only_rational_v1')
PARENT=ROOT/'docs/lp_diagnostic_v1_freeze.json'
IDS=['input220_p0','input222_p1','input230_p2','input232_p0']


def roster():
    original=strict_json(PARENT.read_bytes())['jobs']
    later=strict_json((ROOT/'docs/modular_diagnostic_v1_freeze.json').read_bytes())['jobs']
    if [j['job_id'] for j in original]!=IDS or original!=later:raise ValueError('original roster drift')
    return original


def limits():
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(128*1024**2,128*1024**2))
    os.nice(10)


def native_import(input_path,output_path,root,timeout=60):
    start=time.monotonic()
    command=['/usr/bin/time','-f','%e %M','-o',str(root/'native.resources'),
             str(READER),str(input_path),str(output_path)]
    timed_out=False
    with (root/'native.log').open('xb') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                           start_new_session=True,preexec_fn=limits)
        try:p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(p.pid,signal.SIGKILL);p.wait()
    result=dict(command=command,returncode=p.returncode,timeout=timed_out,
                wall_seconds=time.monotonic()-start,optimization_calls=0,
                address_space_limit_bytes=8*1024**3,file_limit_bytes=128*1024**2)
    usage=root/'native.resources'
    result['native_peak_rss_kib']=None;result['native_reported_seconds']=None
    if usage.exists():
        values=usage.read_text().splitlines()
        if values:
            parts=values[-1].split()
            if len(parts)==2:
                try:result.update(native_reported_seconds=float(parts[0]),native_peak_rss_kib=int(parts[1]))
                except ValueError:pass
    save(root/'native_process.json',result)
    if timed_out:raise TimeoutError('read-only import timeout')
    if p.returncode:raise ValueError('native reader failed; see preserved log')
    return result


def study(output,receipt):
    if output.exists() or receipt.exists():raise FileExistsError('new import-only identity required')
    output.mkdir();rows=[];start=time.monotonic();failed=False
    for job in roster():
        if failed:
            rows.append(dict(job_id=job['job_id'],status='NOT_RUN_AFTER_ERROR'));continue
        dest=output/job['job_id'];dest.mkdir();begin=time.monotonic();phase='load';cost={}
        try:
            tick=deadline_tick(begin+180)
            t=time.monotonic();lp=load_job(job,tick);cost['load_validate']=time.monotonic()-t
            phase='export';t=time.monotonic();size=write_lp(lp,dest/'input.lp',tick);cost['export']=time.monotonic()-t
            phase='import';proc=native_import(dest/'input.lp',dest/'readback.txt',dest)
            cost['native_import']=proc['wall_seconds'];tick()
            phase='compare';t=time.monotonic();comparison=verify_readback(lp,dest/'readback.txt',tick)
            cost['exact_comparison']=time.monotonic()-t
            result=dict(job_id=job['job_id'],status='PASS',comparison=comparison,
                        native=proc,export_bytes=size,readback_bytes=(dest/'readback.txt').stat().st_size,
                        original_lp_sha256=job['statement']['lp_sha256'],statement_sha256=job['statement_sha256'])
        except Exception as exc:
            result=dict(job_id=job['job_id'],status='ERROR',phase=phase,error=repr(exc));failed=True
        result.update(cost_seconds=cost,observed_seconds=time.monotonic()-begin,optimization_calls=0)
        save(dest/'comparison.json',result);rows.append(result)
    result=dict(schema='SOPLEX_LARGE_IMPORT_ONLY_V1',status='FAIL' if failed else 'PASS',rows=rows,
        parent=dict(path=str(PARENT),sha256=sha(PARENT)),reader_sha256=sha(READER),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()},
        checker_sha256=sha(ROOT/'lp_sandwich/check.py'),raw_root=str(output),
        artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
        preparation_seconds=time.monotonic()-start,real_native_imports=sum('native' in r for r in rows),
        optimization_calls=0,candidates=0,scope='Exact input readback only; no LP solution or network conclusion.')
    save(receipt,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();r=study(a.output,a.receipt);print(json.dumps(r['rows'],indent=2))
    raise SystemExit(0 if r['status']=='PASS' else 1)
