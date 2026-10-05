"""Exclusive full qualification, new circuit/input proof, then one fresh terminal."""
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
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c100_fresh_circuit_terminal_20260913_v3'
NAMES=['C100_FRESH_RUNTIME_PREREG_20260913.md','c100_native_binding_v1.py',
    'c100_live_runtime_v1.py','c100_native_terminal_v1.py','c100_runtime_final_binding_v1.py',
    'c100_terminal_observer_v1.py','c100_terminal_preflight_v1.py','c100_prepare_worker_v1.py',
    'c100_changed_terminal_worker_v1.py','test_c100_circuit_terminal_v1.py','run_c100_terminal_supervisor_v1.py']


def main():
    if RUN.exists():raise FileExistsError(RUN)
    previous=EXP/'results/c99_circuit_native_20260913_v2'
    old=json.loads((previous/'preregistered.json').read_text());done=json.loads((previous/'exit.json').read_text())
    if (not done['all_declared_stages_passed'] or done['tests_count']!=2527
        or done['source_drift'] or done['provenance_drift']):raise ValueError('qualified complete C99v2 required')
    hashes=dict(old['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('frozen dependency drift')
    names=NAMES+['C100_POST_SPLICE_PAYMENT_PREREG_20260913.md','c100_live_runtime_v2.py',
        'test_c100_post_splice_payment_v2.py','c100_prepare_worker_v3.py',
        'c100_changed_terminal_worker_v3.py','run_c100_terminal_supervisor_v3.py']+['C100_FRESH_INPUT_PREREG_20260913.md','c100_original_input_v2.py',
        'test_c100_original_input_v2.py','c100_prepare_worker_v2.py','c100_changed_terminal_worker_v2.py',
        'run_c100_terminal_supervisor_v2.py']+[str(p.relative_to(EXP)) for p in
        (previous/'preregistered.json',previous/'exit.json',previous/'preflight/result.json',previous/'component/result.json',previous/'restore/result.json') if p.exists()]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=old['tests']+['test_c100_circuit_terminal_v1.py','test_c100_original_input_v2.py','test_c100_post_splice_payment_v2.py']
    if len(tests)!=117 or len(set(tests))!=117:raise ValueError('incomplete inherited test files')
    provenance=_provenance(ROOT)
    if provenance!=old['provenance']:raise ValueError('production branch/candidate drift')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    terminal_command=list(json.loads((EXP/'results/c83_changed_terminal_20260913_v1/preregistered.json').read_text())['command'])
    terminal_command[:3]=[sys.executable,str(EXP/'c100_changed_terminal_worker_v3.py'),str(RUN)]
    terminal_command[terminal_command.index('--output')+1]=str(RUN/'result.json')
    if '--stop-after-layer' in terminal_command or terminal_command[terminal_command.index('--solver-timeout')+1]!='45':
        raise ValueError('original full-terminal command differs')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        command=terminal_command,tests=tests,required_test_count=2550,complete_test_wall_cap_s=60,stage_worker_wall_cap_s=240,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_bytes=1024**3,
        whole_work_cap=256_000_000,branch_work_cap=200_000_000,entries_cap=64_000_000,
        shared_radix_caps=[16384,131072,16_000_000],native_payload_separate_C32_boundary=True,
        stages=['all_inherited_and_new_tests','complete_new_native_original_input_proof','fresh_original_network_terminal'],
        historical_writes_authorized=False,fresh_runtime_authorized=True,archived_numeric_HZ_runtime_input=False,
        ordinary_terminal_45s_with_base_feasibility=True,promotion_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(all_declared_stages_passed=False,formal_gain=0)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        files={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=2550 or len(set(ids))!=2550
            or {n.split('::',1)[0] for n in ids}!=files):raise ValueError('exact inherited/new test inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(nodeids=ids,count=len(ids),frozen_before_test_execution=True))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            finished=subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=finished.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (finished.returncode or sorted(actual)!=sorted(ids)
            or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete original/new qualification failed')
        print(json.dumps(dict(event='all_tests_passed',tests=len(ids),wall_s=record['test_wall_s'])),flush=True)
        record['prepare_started']=True
        with (RUN/'prepare_worker.log').open('x') as stream:
            worker=subprocess.run([sys.executable,str(EXP/'c100_prepare_worker_v3.py')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['prepare_exit']=worker.returncode
        prepared=json.loads((RUN/'prepare_result.json').read_text())
        print(json.dumps(dict(event='prepare_exit',exit=worker.returncode,completed=prepared['completed'],
            failure=prepared.get('failure'),bound=prepared.get('bound'))),flush=True)
        if worker.returncode or not prepared['completed']:
            raise ValueError('complete new source/input/affine preparation failed; fresh worker not launched')
        bound=json.loads((RUN/'native_bound.json').read_text())
        if not bound['fits']:raise ValueError('complete runtime bound rejected')
        links={name:_sha256(RUN/name) for name in ('source_proof.json','transfer_proof.json','final_proof.json',
            'live_inputs.json','native_bound.json','prepare_result.json')}
        _atomic_exclusive_json(RUN/'prepared_inputs.json',links)
        record['fresh_terminal_started']=True
        with (RUN/'worker.log').open('x') as stream:
            worker=subprocess.run(terminal_command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['fresh_terminal_exit']=worker.returncode
        if worker.returncode:raise ValueError('fresh original terminal rejected')
        audit=json.loads((RUN/'terminal_audit.json').read_text())
        if (not audit['terminal_gates_passed'] or not audit['ordinary_terminal_returned'] or audit.get('failure')):
            raise ValueError('complete fresh original terminal incomplete')
        if any(_sha256(RUN/n)!=sha for n,sha in links.items()):raise ValueError('prepared text source binding changed')
        record['verdict']=audit['concrete_result_verdict']
        record['all_declared_stages_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['all_declared_stages_passed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
