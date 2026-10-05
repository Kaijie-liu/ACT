"""Freeze full qualification, proof transfer, one original native run, restore."""
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
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c74_native_binding_20260913_v2'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((EXP/'results/c74_native_binding_20260913_v1/preregistered.json').read_text())
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('complete inherited source drift')
    names=['C74_NATIVE_BINDING_PREREG_20260913.md','c74_native_binding_v1.py','c74_live_runtime_v1.py',
        'c74_live_worker_v2.py','c74_proof_transfer_worker_v2.py','test_c74_native_binding_v1.py',
        Path(__file__).name,'C73_OUTER_QUERY_AUDIT_20260913.md','C73_NATIVE_INTEGRATION_HANDOFF_20260913.md',
        'CHECKPOINT_C73_NATIVE_PROOF_20260913_SHA256SUMS',
        'results/c73_outer_query_20260913_v1/component/native.pickle',
        'results/c73_outer_query_20260913_v1/component/result.json',
        'results/c73_outer_query_20260913_v1/restore/result.json',
        'results/c73_outer_query_20260913_v1/exit.json']
    names += ['C74_EVENT_CLOCK_CORRECTION_PREREG_20260913.md','test_c74_event_clock_v2.py',
        'results/c74_native_binding_20260913_v1/exit.json',
        'results/c74_native_binding_20260913_v1/qualification.json',
        'results/c74_native_binding_20260913_v1/result.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=[*prior['tests'],'test_c74_event_clock_v2.py']
    if len(tests)!=82 or len(set(tests))!=82:raise ValueError('complete qualification files required')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('branch/production candidate drift')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    command=list(json.loads((EXP/'results/c9_live_relu_20260906_v1/preregistered.json').read_text())['command'])
    command[:3]=[sys.executable,str(EXP/'c74_live_worker_v2.py'),str(RUN)]
    command[command.index('--output')+1]=str(RUN/'result.json')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=1841,command=command,cpu_threads=1,gpu_enabled=False,
        address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,entries_cap=64_000_000,
        whole_work_cap=256_000_000,branch_work_cap=200_000_000,radix_caps=[16384,131072,16_000_000],
        test_wall_cap_s=60,prepare_wall_cap_s=60,native_wall_cap_s=240,restore_wall_cap_s=60,
        source_hash_authentication='separate_complete_C32_style_diagnostics_all_calls_reported',
        fresh_original_network_native_authorized=True,archived_HZ_input_to_native_authorized=False,
        terminal_solve_or_promotion_authorized=False,historical_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,qualification_passed=False)
    try:
        testcmd=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*testcmd,'--collect-only'],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        expected={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=1841 or len(set(ids))!=1841
                or {s.split('::',1)[0] for s in ids}!=expected):raise ValueError('complete exact collection differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(count=len(ids),nodeids=ids,frozen_before_test_execution=True))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(testcmd,60)
        with (RUN/'tests.log').open('x') as stream:
            tested=subprocess.run([*testcmd,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tested.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (tested.returncode or sorted(actual)!=sorted(ids)
                or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete qualification failed; native not started')
        record['qualification_passed']=True
        for stage,cmd,seconds in (
            ('prepare',[sys.executable,str(EXP/'c74_proof_transfer_worker_v2.py'),'prepare'],60),
            ('native',command,240),
            ('restore',[sys.executable,str(EXP/'c74_proof_transfer_worker_v2.py'),'restore'],60)):
            record[stage+'_started']=True
            with (RUN/(stage+'.log')).open('x') as stream:
                proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=seconds)
            record[stage+'_exit']=proc.returncode
            if proc.returncode:raise ValueError(stage+' gate rejected')
            path=RUN/('qualification.json' if stage=='native' else stage+'_result.json')
            result=json.loads(path.read_text())
            if not result.get('passed' if stage=='native' else 'completed'):raise ValueError(stage+' result incomplete')
            print(json.dumps(dict(event=stage+'_passed',wall_s=result['wall_s'])),flush=True)
        record['all_declared_stages_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record.get('all_declared_stages_passed') or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
