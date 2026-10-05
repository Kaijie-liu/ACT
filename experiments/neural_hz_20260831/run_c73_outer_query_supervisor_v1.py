"""Frozen full qualification and bounded offline phase/component/inverse run."""
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
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c73_outer_query_20260913_v1'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((EXP/'results/c72_inverse_phase_20260913_v2/preregistered.json').read_text())
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('complete inherited source drift')
    names=["C72_INVERSE_PHASE_PREREG_20260913.md","c72_inverse_phase_v1.py","c72_native_worker_v2.py","test_c72_inverse_phase_v1.py","run_c72_inverse_phase_supervisor_v2.py","C71_PHASE_INGESTION_PREREG_20260913.md","C71_PHASE_INGESTION_AUDIT_20260913.md","c71_phase_ingestion_worker_v1.py","run_c71_phase_ingestion_supervisor_v1.py","results/c71_phase_ingestion_20260913_v1/result.json","results/c71_phase_ingestion_20260913_v1/exit.json","results/c32_live_splice_20260911_v1/transfer_proof.json","results/c32_live_splice_20260911_v1/transfer_result.json","results/c70_native_proof_20260913_v1/exit.json","results/c70_native_proof_20260913_v1/tests.xml"]
    names += ['C72_WIRE_SCHEMA_CORRECTION_PREREG_20260913.md','c72_native_worker_v1.py','run_c72_inverse_phase_supervisor_v1.py','results/c72_inverse_phase_20260913_v1/preregistered.json','results/c72_inverse_phase_20260913_v1/exit.json','results/c72_inverse_phase_20260913_v1/phase/result.json']
    names += ['C73_OUTER_QUERY_PREREG_20260913.md','c73_outer_query_v1.py','c73_native_worker_v1.py','test_c73_outer_query_v1.py','run_c73_outer_query_supervisor_v1.py','C72_INVERSE_PHASE_AUDIT_20260913.md','results/c72_inverse_phase_20260913_v2/exit.json','results/c72_inverse_phase_20260913_v2/phase/result.json','results/c72_inverse_phase_20260913_v2/component/result.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=[*prior['tests'],'test_c73_outer_query_v1.py']
    if len(tests)!=80 or len(set(tests))!=80:raise ValueError('incomplete old/new test files')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('branch or production candidate drift')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    RUN.mkdir()
    for stage in ('phase','component','restore'):(RUN/stage).mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=1819,test_wall_cap_s=60,phase_cap_s=60,component_cap_s=240,restore_cap_s=60,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,
        entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        original_radix_caps=[16384,131072,16_000_000],
        stages=['complete_tests','phase','component','restore'],
        offline_proof_preparation_only=True,new_native_network_solver_or_default_authorized=False,
        complete_original_inputs_retained=True,historical_writes_authorized=False,formal_gain=0))
    start=time.monotonic();record=dict(formal_gain=0,qualification_passed=False,full_LIVE_admission=False)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        expected={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=1819 or len(set(ids))!=1819
                or {s.split('::',1)[0] for s in ids}!=expected):raise ValueError('complete exact qualification inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(count=len(ids),nodeids=ids,frozen_before_test_execution=True))
        left=60-(time.monotonic()-start)
        if left<=0:raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            tested=subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tested.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-start)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (tested.returncode or sorted(actual)!=sorted(ids)
                or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete qualification failed; no actual source work')
        record['qualification_passed']=True
        for stage,seconds in (('phase',60),('component',240),('restore',60)):
            if stage=='component':
                _atomic_exclusive_json(RUN/stage/'input_binding.json',dict(phase_result_sha256=_sha256(RUN/'phase/result.json')))
            elif stage=='restore':
                _atomic_exclusive_json(RUN/stage/'input_binding.json',dict(component_result_sha256=_sha256(RUN/'component/result.json')))
            record[stage+'_started']=True
            with (RUN/stage/'worker.log').open('x') as stream:
                job=subprocess.run([sys.executable,str(EXP/'c73_native_worker_v1.py'),stage],cwd=ROOT,
                    env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=seconds)
            record[stage+'_exit']=job.returncode
            if job.returncode:raise ValueError(stage+' gate rejected')
            result=json.loads((RUN/stage/'result.json').read_text())
            if not result['completed']:raise ValueError(stage+' lacks a completed result')
            print(json.dumps(dict(event=stage+'_passed',wall_s=result['wall_s'],diagnostic_work=result['diagnostic_work'])),flush=True)
        record['all_declared_stages_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-start,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record.get('all_declared_stages_passed') or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
