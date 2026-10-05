"""Freeze complete qualification and costs, then one bounded fresh terminal."""
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
from experiments.neural_hz_20260831.c83_runtime_final_binding_v1 import load,final_extra
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c83_changed_terminal_20260913_v1'
FINAL_SHA='4257d022e5c268d534959b8f6cf084b4ca372d1748504f156f81bf37be30c0c5'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((EXP/'results/c82_native_terminal_20260913_v2/preregistered.json').read_text())
    end=json.loads((EXP/'results/c82_native_terminal_20260913_v2/exit.json').read_text())
    hashes=dict(prior['source_sha256'])
    if (not end['all_declared_stages_passed'] or end['tests_count']!=1970 or end['tests_exit']!=0
            or end['source_drift'] or end['provenance_drift']
            or any(_sha256(EXP/n)!=h for n,h in hashes.items())):
        raise ValueError('complete independent qualification/source prerequisite changed')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('branch/production candidate drift')
    names=['C83_FRESH_TERMINAL_PREREG_20260913.md','c83_runtime_final_binding_v1.py',
        'c83_terminal_observer_v1.py','c83_changed_terminal_worker_v1.py',
        'test_c83_runtime_terminal_v1.py',Path(__file__).name,
        'C82_FRESH_TERMINAL_HANDOFF_20260913.md','CHECKPOINT_C82_NATIVE_TERMINAL_20260913_SHA256SUMS',
        'results/c82_native_terminal_20260913_v2/preregistered.json',
        'results/c82_native_terminal_20260913_v2/exit.json',
        'results/c82_native_terminal_20260913_v2/result.json',
        'results/c82_native_terminal_20260913_v2/final_proof.json',
        'results/c74_native_binding_20260913_v2/qualification.json',
        'results/c74_native_binding_20260913_v2/native_bound.json',
        'results/c74_native_binding_20260913_v2/preregistered.json',
        'results/c74_native_binding_20260913_v2/source_proof.json',
        'results/c74_native_binding_20260913_v2/transfer_proof.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=list(prior['tests'])
    if len(tests)!=88 or len(set(tests))!=88:raise ValueError('complete old test file list required')
    tests.append('test_c83_runtime_terminal_v1.py');required_count=1992
    old=EXP/'results/c74_native_binding_20260913_v2'
    native=json.loads((old/'qualification.json').read_text())
    old_bound=json.loads((old/'native_bound.json').read_text())
    final=load((EXP/'results/c82_native_terminal_20260913_v2/final_proof.json').read_bytes(),FINAL_SHA)
    if not native['passed'] or not native['whole_live_path_proved'] or not old_bound['fits']:
        raise ValueError('complete actual C74 native prerequisite missing')
    # C82 preparation executed exactly two original topology-signature calls.
    topology=final['diagnostic_work_parts']['terminal_suffix_topology_metadata']
    if topology%2 or (topology//2-256)%64:raise ValueError('complete original topology population differs')
    layers=(topology//2-256)//64;extra=final_extra(layers)
    bound=dict(whole=old_bound['whole']+extra,branch=old_bound['branch']+extra,
        original_native_whole=old_bound['whole'],original_native_branch=old_bound['branch'],
        actual_layer_count=layers,terminal_extra=extra,scope_observer=512,source_publication=1024,
        original_suffix_topology=256+64*layers,source_native_payload_separately_paid=True,
        full_authentication_separately_reported=True,formal_gain=0)
    bound['fits']=bound['whole']<=256_000_000 and bound['branch']<=200_000_000
    if not bound['fits']:raise ValueError('complete fresh-terminal bound does not fit')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    command=list(json.loads((old/'preregistered.json').read_text())['command'])
    command[:3]=[sys.executable,str(EXP/'c83_changed_terminal_worker_v1.py'),str(RUN)]
    command[command.index('--output')+1]=str(RUN/'result.json')
    if command.count('--stop-after-layer')!=1:raise ValueError('original native command stop differs')
    at=command.index('--stop-after-layer')
    if command[at+1]!='78':raise ValueError('original measurement source stop differs')
    del command[at:at+2]
    if command[command.index('--solver-timeout')+1]!='45':raise ValueError('original solver deadline differs')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'native_bound.json',bound)
    _atomic_exclusive_json(RUN/'live_inputs.json',dict(final_proof_sha256=FINAL_SHA,
        native_bound_sha256=_sha256(RUN/'native_bound.json'),
        native_lowered_n_cont=native['native_ingestion']['lowered_n_cont'],
        native_lowered_n_bin=native['native_ingestion']['lowered_n_bin']))
    hashes.update({str((RUN/n).relative_to(EXP)):_sha256(RUN/n) for n in ('native_bound.json','live_inputs.json')})
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=required_count,command=command,cpu_threads=1,gpu_enabled=False,
        address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,entries_cap=64_000_000,
        whole_work_cap=256_000_000,branch_work_cap=200_000_000,test_wall_cap_s=60,
        worker_wall_cap_s=240,solver_time_limit_s=45,radix_caps=[16384,131072,16_000_000],
        source_hash_authentication='separate_complete_C32_style_diagnostics_all_calls_reported',
        fresh_original_network_native_authorized=True,archived_HZ_input_authorized=False,
        ordinary_terminal_authorized=True,promotion_authorized=False,historical_writes_authorized=False,
        complete_final_checkpoint_restoration_not_yet_qualified=True,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,qualification_passed=False)
    try:
        testcmd=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collected=subprocess.run([*testcmd,'--collect-only'],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as s:s.write(collected.stdout)
        ids=[s for s in collected.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        expected={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collected.returncode or len(ids)!=required_count or len(set(ids))!=required_count
                or {s.split('::',1)[0] for s in ids}!=expected):raise ValueError('complete collection differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(count=len(ids),nodeids=ids,frozen_before_test_execution=True))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(testcmd,60)
        with (RUN/'tests.log').open('x') as s:
            tested=subprocess.run([*testcmd,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=s,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tested.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (tested.returncode or sorted(actual)!=sorted(ids)
                or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete qualification failed; original network not started')
        record['qualification_passed']=True
        print(json.dumps(dict(event='complete_tests_passed',count=len(ids),wall_s=record['test_wall_s'],bound=bound)),flush=True)
        record['fresh_terminal_started']=True
        with (RUN/'worker.log').open('x') as s:
            proc=subprocess.run(command,cwd=ROOT,env=env,stdout=s,stderr=subprocess.STDOUT,timeout=240)
        record['fresh_terminal_exit']=proc.returncode
        if proc.returncode:raise ValueError('fresh original terminal rejected')
        result=json.loads((RUN/'terminal_audit.json').read_text())
        if (not result['ordinary_terminal_returned'] or not result['terminal_gates_passed']
                or result.get('failure')):raise ValueError('fresh terminal incomplete')
        record.update(all_declared_stages_passed=True,ordinary_terminal_returned=True,
                      verdict=result['concrete_result_verdict'])
        print(json.dumps(dict(event='fresh_terminal_returned',verdict=record['verdict'])),flush=True)
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record.get('all_declared_stages_passed') or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
