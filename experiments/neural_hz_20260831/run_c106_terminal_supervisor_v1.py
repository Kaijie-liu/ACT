# SPDX-License-Identifier: AGPL-3.0-or-later
"""All original tests, complete-source payment and one unchanged-gate terminal."""
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
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c106_defining_rows_20260913_v1'
NAMES=['C106_DEFINING_ROWS_PREREG_20260913.md','C106_NATIVE_BUILD_20260913.json',
    'C106_LOCAL_DRAFT_CORRECTION_20260913.md','c106_defining_rows_native_v1.c',
    '_c106_defining_rows_v1.cpython-313-x86_64-linux-gnu.so','c106_defining_rows_v1.py',
    'c106_birth_emission_v1.py','c106_live_runtime_v1.py','c106_changed_terminal_worker_v1.py',
    'c106_source_payment_worker_v1.py','test_c106_defining_rows_v1.py',
    'test_c106_complete_source_v1.py','test_c106_defining_rows_v2.py',
    'test_c106_complete_source_v2.py','run_c106_terminal_supervisor_v1.py']


def main():
    if RUN.exists():raise FileExistsError(RUN)
    predecessor=EXP/'results/c104_array_api_20260913_v1'
    prior=json.loads((predecessor/'preregistered.json').read_text())
    done=json.loads((predecessor/'exit.json').read_text())
    if (_sha256(predecessor/'exit.json')!='65477ea097be278bc508fcc51cbcb3e52b20a7c1a6ad8eacd983d08b70a9b217'
        or done['tests_count']!=2941 or done['tests_exit'] or done.get('timeout_s')!=240
        or done['source_drift'] or done['provenance_drift']):
        raise ValueError('complete honestly closed C104 qualification missing')
    hashes=dict(prior['source_sha256'])
    diagnosis=EXP/'results/c105_source_profile_20260913_v1'
    diagnostic=json.loads((diagnosis/'exit.json').read_text())
    if (_sha256(diagnosis/'exit.json')!='5753eacf9d25973e1fea129261bed264bf7a0bb89565a0782051d9ad2a25085c'
        or not diagnostic['completed'] or diagnostic['worker_exit']
        or diagnostic['source_drift'] or diagnostic['provenance_drift']):
        raise ValueError('complete bounded C105 observation missing')
    if any(_sha256(diagnosis/n)!=sha for n,sha in diagnostic['artifacts'].items()):
        raise ValueError('C105 complete result drift')
    hashes.update(json.loads((diagnosis/'preregistered.json').read_text())['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source/ABI/proof drift')
    previous=EXP/'results/c100_fresh_circuit_terminal_20260913_v4'
    old=json.loads((previous/'preregistered.json').read_text())
    links=json.loads((previous/'prepared_inputs.json').read_text())
    if any(_sha256(previous/n)!=sha for n,sha in links.items()):raise ValueError('complete original text proof chain drift')
    if not json.loads((previous/'prepare_result.json').read_text())['completed']:
        raise ValueError('qualified complete C100 source/input/property proof missing')
    names=NAMES+[str((predecessor/n).relative_to(EXP)) for n in ('preregistered.json','exit.json','terminal_gate.json')]
    names += [str((diagnosis/n).relative_to(EXP)) for n in ('preregistered.json','exit.json','result.json')]
    hashes.update({n:_sha256(EXP/n) for n in names})
    build=json.loads((EXP/'C106_NATIVE_BUILD_20260913.json').read_text())
    if any(_sha256(Path(n))!=sha for n,sha in build['dependency_sha256'].items()):
        raise ValueError('new complete row compiler/header/binary drift')
    hashes.update(build['dependency_sha256'])
    tests=prior['tests']+['test_c106_defining_rows_v2.py','test_c106_complete_source_v2.py']
    if len(tests)!=130 or len(set(tests))!=130:raise ValueError('complete inherited test set differs')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance'] or provenance!=old['provenance']:raise ValueError('production provenance drift')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    terminal_command=list(prior['command'])
    terminal_command[:3]=[sys.executable,str(EXP/'c106_changed_terminal_worker_v1.py'),str(RUN)]
    terminal_command[terminal_command.index('--output')+1]=str(RUN/'result.json')
    if '--stop-after-layer' in terminal_command or terminal_command[terminal_command.index('--solver-timeout')+1]!='45':
        raise ValueError('full original ordinary terminal command differs')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        command=terminal_command,tests=tests,required_test_count=2989,complete_test_wall_cap_s=60,stage_worker_wall_cap_s=240,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_bytes=1024**3,
        whole_work_cap=256_000_000,branch_work_cap=200_000_000,entries_cap=64_000_000,
        shared_radix_caps=[16384,131072,16_000_000],native_payload_separate_C32_boundary=True,
        stages=['all_inherited_and_new_tests','complete_original_and_new_source_payment','fresh_original_network_terminal'],
        payment_diagnostic_wall_cap_s=60,payment_speed_ratio_floor=1.0,
        payment_geometry=dict(c=16,k=32,h=8),diagnostic_source_caps=[64_000_000,64_000_000],
        diagnostic_shared_token_cap=256_000_000,complete_traversals=4,
        full_numeric_hashes_unchanged=True,numeric_hash_traffic_in_token_pool=False,
        all_CPU_work_in_generation_cap=False,qualified_proof_directory=str(previous),
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
        if (collection.returncode or len(ids)!=2989 or len(set(ids))!=2989
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
        record['payment_started']=True
        with (RUN/'payment_worker.log').open('x') as stream:
            paid=subprocess.run([sys.executable,str(EXP/'c106_source_payment_worker_v1.py')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        record['payment_exit']=paid.returncode
        payment=json.loads((RUN/'payment_result.json').read_text())
        print(json.dumps(dict(event='payment_diagnostic_returned',completed=payment['completed'],
            ratio=payment.get('diagnostic_speed_ratio'),failure=payment.get('failure'))),flush=True)
        if paid.returncode or not payment['completed'] or payment['source_drift']:
            raise ValueError('changed exact affine preparation has no qualified payment; no network launched')
        record['qualified_C100_text_proof_reused']=True
        record['fresh_numeric_source_still_required']=True
        if any(_sha256(previous/n)!=sha for n,sha in links.items()):
            raise ValueError('qualified source/input/property text proof changed after tests')
        record['fresh_terminal_started']=True
        with (RUN/'worker.log').open('x') as stream:
            worker=subprocess.run(terminal_command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['fresh_terminal_exit']=worker.returncode
        if worker.returncode:raise ValueError('fresh original terminal rejected')
        audit=json.loads((RUN/'terminal_audit.json').read_text())
        if (not audit['terminal_gates_passed'] or not audit['ordinary_terminal_returned'] or audit.get('failure')):
            raise ValueError('complete fresh original terminal incomplete')
        if any(_sha256(previous/n)!=sha for n,sha in links.items()):raise ValueError('prepared text source binding changed')
        if len(audit['complete_traversal_receipts'])!=4 or any(
                not receipt['measurement']['measured_transient_gate'] for receipt in audit['complete_traversal_receipts']):
            raise ValueError('complete measured original-runtime traversals missing')
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

