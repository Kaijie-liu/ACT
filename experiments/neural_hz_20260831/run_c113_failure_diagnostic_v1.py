# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded frozen failure diagnosis; no revision of C113 qualification gates."""
import json
import os
from pathlib import Path
import pickle
import resource
import faulthandler
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c113_complete_source_diagnostic_v1 import complete_source
from experiments.neural_hz_20260831.c113_masked_plan_diagnostic_v1 import diagnose
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c113_failure_diagnostic_20260913_v1'
PREVIOUS=EXP/'results/c113_once_prepared_20260913_v1'


def worker():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((RUN/'preregistered.json').read_text());started=time.monotonic()
    reserve=WorkPool(256_000_000);reserve.charge('three_complete_source_calls',96_000_000)
    reserve.charge('dense_complete_source_proof',64_000_000)
    reserve.charge('masked_complete_source_plan_and_proofs',64_000_000)
    reserve.charge('observation_reserve',32_000_000)
    dense=WorkPool(64_000_000);masked=WorkPool(64_000_000)
    record=dict(completed=False,formal_gain=0,C113_v1_qualification_passed=False)
    fatal=(RUN/'fatal.log').open('x');faulthandler.enable(file=fatal,all_threads=True)
    try:
        if any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
        for name,pool,fn in (('dense',dense,lambda:complete_source('dense',pool=dense)),
                             ('masked',masked,lambda:diagnose(pool=masked))):
            record['active_stage']=name;held={}
            def build():
                data,roots=fn();held['complete_roots']=roots;return data
            data,measurement=measured(build,observe=lambda m:record.update(active_measurement=m))
            archive=RUN/(name+'_complete_roots.pickle')
            with archive.open('xb') as stream:
                pickle.dump(held['complete_roots'],stream,protocol=5);stream.flush();os.fsync(stream.fileno())
            data.update(measurement=measurement,archive=archive.name,archive_sha256=_sha256(archive))
            _atomic_exclusive_json(RUN/(name+'.json'),data)
            record[name]=data
            print(json.dumps(dict(event='c113_failure_diagnosis_complete',stage=name,wall_s=measurement['elapsed_s'],work=pool.used)),flush=True)
            del held
        record['completed']=True
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        faulthandler.disable();fatal.close()
        record.update(wall_s=time.monotonic()-started,aggregate_reserved_work=reserve.used,
            reservations=reserve.parts,dense_work=dense.used,dense_parts=dense.parts,
            masked_work=masked.used,masked_parts=masked.parts,
            source_drift=any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('dense','masked')}),flush=True)
    if not record['completed'] or record['source_drift']:raise SystemExit(1)


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((PREVIOUS/'preregistered.json').read_text());done=json.loads((PREVIOUS/'exit.json').read_text())
    if (_sha256(PREVIOUS/'exit.json')!='65001272f2f784a782e249853eca8468f273cf36803052813cbc664a16473e81'
        or done['all_stages_passed'] or done['tests_count']!=3286 or done['tests_exit']!=1
        or done['source_drift'] or done['provenance_drift']):raise ValueError('exact failed C113 prerequisite required')
    hashes=dict(prior['source_sha256'])
    for name,h in done['artifacts'].items():hashes[str((PREVIOUS/name).relative_to(EXP))]=h
    names=['C113_FAILURE_DIAGNOSTIC_PREREG_20260913.md','c113_complete_source_diagnostic_v1.py',
        'c113_masked_plan_diagnostic_v1.py','run_c113_failure_diagnostic_v1.py',
        str((PREVIOUS/'exit.json').relative_to(EXP))]
    hashes.update({n:_sha256(EXP/n) for n in names})
    if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('complete dependency or failure result drift')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('production provenance drift')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        CPU_threads=1,GPU=False,address_space_bytes=16*1024**3,transient_bytes=1024**3,entries_cap=64_000_000,
        aggregate_reserved_work=256_000_000,worker_seconds=240,complete_failed_geometries_unchanged=True,
        independent_witness_checks_not_removed=True,C113_v1_remains_closed=True,
        source_runtime_LIVE_admitted=False,new_solver_run_authorized=False,formal_gain=0))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    started=time.monotonic();record=dict(all_stages_passed=False,formal_gain=0)
    try:
        with (RUN/'worker.log').open('x') as stream:
            job=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
                env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=job.returncode
        result=json.loads((RUN/'result.json').read_text())
        if job.returncode or not result['completed'] or result['source_drift']:raise ValueError('complete diagnosis failed')
        record['all_stages_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(p.relative_to(RUN)):_sha256(p) for p in RUN.rglob('*') if p.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['all_stages_passed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker()
    else:main()
