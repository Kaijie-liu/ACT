"""Exclusive60s diagnostic supervisor; reuse only exact unchanged qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c105_source_profile_20260913_v1'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    old=EXP/'results/c104_array_api_20260913_v1';prior=json.loads((old/'preregistered.json').read_text());done=json.loads((old/'exit.json').read_text())
    if (_sha256(old/'exit.json')!='65477ea097be278bc508fcc51cbcb3e52b20a7c1a6ad8eacd983d08b70a9b217'
        or done['tests_count']!=2941 or done['tests_exit'] or done['source_drift'] or done['provenance_drift']):
        raise ValueError('exact previous full qualification missing')
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('unchanged qualification dependency drift')
    names=['C105_SOURCE_PROFILE_PREREG_20260913.md','c105_source_profile_worker_v1.py','run_c105_source_profile_v1.py']
    names += [str((old/n).relative_to(EXP)) for n in ('preregistered.json','exit.json')]
    hashes.update({n:_sha256(EXP/n) for n in names})
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('production provenance drift')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    RUN.mkdir();command=[sys.executable,str(EXP/'c105_source_profile_worker_v1.py')]
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,command=command,
        reused_unchanged_tests=2941,reused_test_wall_s=done['test_wall_s'],worker_wall_cap_s=60,
        geometry=dict(c=16,k=32,h=12),aggregate_work_cap=256_000_000,source_caps=[64_000_000,64_000_000],
        profile_reserve=64_000_000,binding_reserve=64_000_000,formal_gain=0,real_network_launched=False))
    start=time.monotonic();record=dict(completed=False,formal_gain=0)
    try:
        with (RUN/'worker.log').open('x') as f:
            done=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['worker_exit']=done.returncode
        result=json.loads((RUN/'result.json').read_text())
        if done.returncode or not result['completed'] or result['source_drift']:raise ValueError('complete diagnostic rejected')
        record['completed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-start,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={f.name:_sha256(f) for f in RUN.iterdir() if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['completed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
