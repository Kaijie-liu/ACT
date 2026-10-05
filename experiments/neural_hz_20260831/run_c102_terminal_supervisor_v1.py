# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full equivalence, traced payment diagnostic, then one fresh C102 terminal."""
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
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c102_word_runtime_20260913_v1'
NAMES=['C102_WORD_RUNTIME_PREREG_20260913.md','C102_NATIVE_BUILD_20260913.json',
    'c102_integer_prefix_native_v1.c','_c102_integer_prefix_v1.cpython-313-x86_64-linux-gnu.so',
    'c102_complete_roots_v1.py','c102_runtime_roots_v1.py','test_c102_complete_roots_v1.py',
    'test_c102_runtime_roots_v1.py','c102_token_payment_worker_v1.py',
    'c102_changed_terminal_worker_v1.py','run_c102_terminal_supervisor_v1.py']


def main():
    if RUN.exists():raise FileExistsError(RUN)
    previous=EXP/'results/c100_fresh_circuit_terminal_20260913_v4'
    old=json.loads((previous/'preregistered.json').read_text());done=json.loads((previous/'exit.json').read_text())
    if (_sha256(previous/'exit.json')!='560da91d69b4b03a0ae0d4b7f14b70a1902aee1eff5fba973ea72bda26106cba'
        or _sha256(previous/'preregistered.json')!='a5aa6df0cd5f9c311b952e64ccf4ff238b16bdeebb4b46d1ac970df06f00802f'
        or done['tests_count']!=2550 or done['prepare_exit']!=0 or done.get('timeout_s')!=240
        or done['source_drift'] or done['provenance_drift']):
        raise ValueError('qualified C100 proof with honestly recorded terminal timeout required')
    links=json.loads((previous/'prepared_inputs.json').read_text())
    if any(_sha256(previous/n)!=sha or done['artifacts'][n]!=sha for n,sha in links.items()):
        raise ValueError('complete C100 prepared input chain drift')
    if not json.loads((previous/'prepare_result.json').read_text())['completed']:
        raise ValueError('complete source/input/final affine preparation required')
    if _sha256(previous/'terminal_gate.json')!=done['artifacts']['terminal_gate.json']:
        raise ValueError('original actual LIVE qualification record drift')
    gate=json.loads((previous/'terminal_gate.json').read_text())
    if not gate['terminal_gates_passed'] or not gate['physical_decrease'] or not gate['final_native_ingestion']['passed']:
        raise ValueError('C100 actual full LIVE and native prerequisite missing')
    del gate
    c101=EXP/'results/c101_runtime_roots_20260913_v1'
    c101_prior=json.loads((c101/'preregistered.json').read_text())
    if (_sha256(c101/'exit.json')!='ac30dcf01e59cbcf02743b5baa7da9c6b57f55bad15780e2400bc6d66db114d4'
        or _sha256(c101/'preregistered.json')!='1730c99fd2ed8a4ace95df629b43d644b8af50205b3f27e5020a296bd71938a1'):
        raise ValueError('closed C101 predecessor drift')
    c101_done=json.loads((c101/'exit.json').read_text())
    if c101_done['tests_count']!=2556 or c101_done['tests_exit'] or c101_done['source_drift'] or c101_done['provenance_drift']:
        raise ValueError('complete previous qualification missing')
    hashes=dict(c101_prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('frozen dependency drift')
    names=NAMES+[str((previous/n).relative_to(EXP)) for n in
        ('preregistered.json','exit.json','prepared_inputs.json','terminal_gate.json',*links)]
    names += [str((c101/n).relative_to(EXP)) for n in ('preregistered.json','exit.json')]
    hashes.update({n:_sha256(EXP/n) for n in names})
    build=json.loads((EXP/'C102_NATIVE_BUILD_20260913.json').read_text())
    if any(_sha256(Path(n))!=sha for n,sha in build['dependency_sha256'].items()):
        raise ValueError('new compiled source/binary/header provenance differs')
    hashes.update(build['dependency_sha256'])
    tests=c101_prior['tests']+['test_c102_complete_roots_v1.py','test_c102_runtime_roots_v1.py']
    if len(tests)!=120 or len(set(tests))!=120:raise ValueError('incomplete inherited test files')
    provenance=_provenance(ROOT)
    if provenance!=old['provenance']:raise ValueError('production branch/candidate drift')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    terminal_command=list(old['command'])
    terminal_command[:3]=[sys.executable,str(EXP/'c102_changed_terminal_worker_v1.py'),str(RUN)]
    terminal_command[terminal_command.index('--output')+1]=str(RUN/'result.json')
    if '--stop-after-layer' in terminal_command or terminal_command[terminal_command.index('--solver-timeout')+1]!='45':
        raise ValueError('original full-terminal command differs')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        command=terminal_command,tests=tests,required_test_count=2583,complete_test_wall_cap_s=60,stage_worker_wall_cap_s=240,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_bytes=1024**3,
        whole_work_cap=256_000_000,branch_work_cap=200_000_000,entries_cap=64_000_000,
        shared_radix_caps=[16384,131072,16_000_000],native_payload_separate_C32_boundary=True,
        stages=['all_inherited_and_new_tests','complete_ordinary_traced_payment','qualified_text_only_C100_input_proof_reuse','fresh_original_network_terminal'],
        payment_diagnostic_wall_cap_s=60,payment_speed_ratio_floor=1.5,
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
        if (collection.returncode or len(ids)!=2583 or len(set(ids))!=2583
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
            paid=subprocess.run([sys.executable,str(EXP/'c102_token_payment_worker_v1.py')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        record['payment_exit']=paid.returncode
        payment=json.loads((RUN/'payment_result.json').read_text())
        print(json.dumps(dict(event='payment_diagnostic_returned',completed=payment['completed'],
            ratio=payment.get('diagnostic_speed_ratio'),failure=payment.get('failure'))),flush=True)
        if paid.returncode or not payment['completed'] or payment['source_drift']:
            raise ValueError('changed exact traversal has no qualified payment; no network launched')
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
