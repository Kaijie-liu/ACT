"""Exclusive complete inherited qualification; no unregistered target launch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c64_birth_generator_20260913_v1'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    old=json.loads((EXP/'results/c31_prepared_generator_20260911_v1/preregistered.json').read_text())
    last=json.loads((EXP/'results/c63_birth_routing_20260913_v1/preregistered.json').read_text())
    hashes=dict(last['source_sha256'])
    for name,sha in old['source_sha256'].items():
        if name in hashes and hashes[name]!=sha:raise ValueError('inherited source conflict')
        hashes[name]=sha
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=['C64_BIRTH_GENERATOR_PREREG_20260913.md','c64_compact_boundary_dp_v1.py',
        'c64_gauged_products_v1.py','c64_birth_quotient_v1.py','c64_birth_emission_v1.py',
        'test_c64_birth_emission_v1.py',Path(__file__).name,'C63_OWNED_BIRTH_HANDOFF_20260913.md',
        'CHECKPOINT_C63_BIRTH_ROUTING_20260913_SHA256SUMS',
        'results/c63_birth_routing_20260913_v1/result.json','results/c63_birth_routing_20260913_v1/exit.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=list(dict.fromkeys([*old['tests'],*last['focused_tests'],'test_c64_birth_emission_v1.py']))
    if len(old['tests'])!=49 or len(last['focused_tests'])!=7 or len(tests)!=57:raise ValueError('complete inherited file inventory differs')
    provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        stage='fresh_owned_birth_generator_complete_qualification',tests=tests,test_file_count=len(tests),
        test_wall_cap_s=60,worker_cap_s=240,cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,
        transient_cap_bytes=1024**3,entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        unchanged_radix_auxiliary_cap=16384,unchanged_radix_added_entries_cap=131072,unchanged_radix_work_cap=16_000_000,
        actual_generator_launch_authorized_by_this_runner=False,formal_gain=0))
    started=time.monotonic();record=dict(qualification_passed=False,actual_target_started=False,formal_gain=0)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=60)
        with (DIRECTORY/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if '::' in s and s.startswith(('experiments/','act/'))]
        files={s.split('::',1)[0] for s in ids}
        expected_files={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if collection.returncode or not ids or len(ids)!=len(set(ids)) or files!=expected_files:
            raise ValueError('complete exact test collection failed')
        _atomic_exclusive_json(DIRECTORY/'inventory.json',dict(nodeids=ids,count=len(ids),files=sorted(files),frozen_before_execution=True))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(command,60)
        with (DIRECTORY/'tests.log').open('x') as stream:
            checked=subprocess.run([*command,'--junitxml='+str(DIRECTORY/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=checked.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(DIRECTORY/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if checked.returncode or sorted(actual)!=sorted(ids) or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped')):
            raise ValueError('complete inherited/new exactness gate failed; no actual generator')
        record['qualification_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={f.name:_sha256(f) for f in DIRECTORY.iterdir() if f.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['qualification_passed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
