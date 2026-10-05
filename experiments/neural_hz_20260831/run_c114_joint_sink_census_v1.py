# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full inherited tests and exclusive bounded joint-factor observation batch."""
import faulthandler
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c88_inline_tile_v1 import actual_rows
from experiments.neural_hz_20260831.c114_joint_sink_census_v1 import analyse
from experiments.neural_hz_20260831.run_c111_bound_accounting_v1 import screen,INPUTS
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent
RUN=EXP/'results/c114_joint_sink_census_20260913_v1'
C112=EXP/'results/c112_factored_key_20260913_v1'
C113=EXP/'results/c113_failure_diagnostic_20260913_v1'
ORDINARY={
    'dense':('dense_complete_roots.pickle','0011ca30d8a5f1da522d7b10e3254ec96ca2360a51c53fe0dd32f45ed10eec35'),
    'masked':('masked_complete_roots.pickle','d13f24ddf709154df20519a15d2cd32ed6efe1409e9ba5246042257ba58e2d95')}


def summary(report):
    return {k:v for k,v in report.items() if k not in ('factors','group_certificates')}


def batch_stage(name,pool,held):
    reports={};views={};auth_started=time.monotonic()
    if name=='actual':
        source_bill=screen(pool,held)
        auth_s=time.monotonic()-auth_started
        base=held['all_aux'][0]['slot']
        for label,aliases in (('original',()),('sealed_root_aliases',held['aliases'])):
            reports[label],views[label]=analyse(held['all_aux'],held['all_outputs'],base,
                pool=pool,aliases=aliases,enabled=True)
        scope=dict(complete_source_chain=source_bill,
            original_real_HZ_numeric_roots_present=False,original_HZ_loaded_into_live_target=False)
    else:
        filename,expected=ORDINARY[name];path=C113/filename
        if _sha256(path)!=expected:raise ValueError('complete ordinary original-source archive drift')
        with path.open('rb') as stream:roots=pickle.load(stream)
        held['complete_ordinary_original_source_and_proofs']=roots
        auth_s=time.monotonic()-auth_started
        if name=='dense':
            packets=roots['new']['construction']['circuits']
            if len(packets)!=4 or len(roots['block_proofs'])!=4:raise ValueError('complete dense source scope differs')
            items=[(f'tile{i}_C113',p,p['new_factors']) for i,p in enumerate(packets)]
        else:
            packets=roots['all_packets'];records=roots['all_reports']
            if len(packets)!=8 or len(records)!=4:raise ValueError('complete masked source scope differs')
            items=[(f'tile{i//2}_{("C96","C113")[i%2]}',p,
                records[i//2]['implementations'][("C96","C113")[i%2]]['report']['new_factors'])
                for i,p in enumerate(packets)]
        for label,packet,count in items:
            if np.any(packet['ab_indptr']):raise ValueError('circuit defining equations must be binary-free')
            pool.charge('c114_complete_ordinary_packet_decode',
                4*sum(v.size for v in packet.values() if type(v) is np.ndarray)+4096)
            aux,outputs=actual_rows(dict(rows=len(packet['rhs']),new_factors=count),packet)
            reports[label],views[label]=analyse(aux,outputs,aux[0]['slot'],pool=pool,enabled=True)
            views[label]['complete_original_native_rows']=dict(aux=aux,outputs=outputs)
        scope=dict(complete_ordinary_source_archive=str(path.relative_to(EXP)),sha256=expected,
            original_ordinary_HZ_numeric_roots_present=True,original_HZ_loaded_into_live_target=False,
            all_original_source_graph_owners_maps_inverses_and_proofs_held=True)
    held['all_views']=views
    layout=numeric_layout(held,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete numeric entries exceed64M')
    with (RUN/(name+'_diagnostic_views.pickle')).open('xb') as stream:
        pickle.dump(views,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    data=dict(scope=scope,reports=reports,complete_numeric_bytes=layout.resident_bytes,
        complete_numeric_entries=layout.resident_entries,
        authentication_decode_and_inherited_screen_s=auth_s,
        numeric_work_not_all_CPU_or_hash_traffic=True,actual_eliminations=0,
        new_source_or_LIVE_admission=False,formal_gain=0)
    _atomic_exclusive_json(RUN/(name+'.json'),data)
    return dict(scope=scope,reports={k:summary(v) for k,v in reports.items()},
        complete_numeric_bytes=layout.resident_bytes,complete_numeric_entries=layout.resident_entries,
        authentication_decode_and_inherited_screen_s=auth_s)


def worker():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((RUN/'preregistered.json').read_text());pool=WorkPool(256_000_000)
    started=time.monotonic();record=dict(completed=False,stages={},formal_gain=0)
    fatal=(RUN/'fatal.log').open('x');faulthandler.enable(file=fatal,all_threads=True)
    try:
        if any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()):raise ValueError('complete source drift')
        for name in ('actual','dense','masked'):
            held={};begin=pool.used;record['active_stage']=name
            data,measurement=measured(lambda:batch_stage(name,pool,held),
                observe=lambda m:record.update(active_measurement=m))
            record['stages'][name]=dict(data=data,measurement=measurement,work=pool.used-begin)
            print(json.dumps(dict(event='c114_complete_stage',stage=name,work=pool.used-begin,
                wall_s=measurement['elapsed_s'],reports=data['reports'])),flush=True)
            del held
        record['completed']=True
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        faulthandler.disable();fatal.close()
        record.update(work=pool.used,work_parts=pool.parts,wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in freeze['source_sha256'].items()))
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='stages'}),flush=True)
    if not record['completed'] or record['source_drift']:raise SystemExit(1)


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((C112/'preregistered.json').read_text());done=json.loads((C112/'exit.json').read_text())
    if (_sha256(C112/'exit.json')!='8af05ad99161d7759985c9a9e62acf9eb5dc72cc8227bceedd817e48ca4c3d44'
            or not done['all_stages_passed'] or done['tests_count']!=3268):
        raise ValueError('last passed full3268-test prerequisite missing')
    if _sha256(C113/'exit.json')!='5c2cb5c6691814d45fda3d53f58c0c280c047bc87033365321da9d2cbaf50844':
        raise ValueError('unchanged full dense/masked diagnostic binding missing')
    hashes=dict(prior['source_sha256'])
    for previous in (C113,EXP/'results/c111_bound_accounting_20260913_v1'):
        for path,sha in json.loads((previous/'preregistered.json').read_text())['source_sha256'].items():
            if path in hashes and hashes[path]!=sha:raise ValueError('source manifest conflict')
            hashes[path]=sha
    for path,sha in INPUTS.values():hashes[path]=sha
    for filename,sha in ORDINARY.values():hashes[str((C113/filename).relative_to(EXP))]=sha
    names=['C114_JOINT_SINK_PREREG_20260913.md','c114_joint_sink_census_v1.py',
        'test_c114_joint_sink_census_v1.py','run_c114_joint_sink_census_v1.py',
        'C113_MAJOR_FACTOR_HANDOFF_20260913.md','run_c111_bound_accounting_v1.py',
        str((C112/'exit.json').relative_to(EXP)),str((C113/'exit.json').relative_to(EXP))]
    hashes.update({n:_sha256(EXP/n) for n in names})
    if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('frozen input/source drift')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('production provenance drift')
    tests=prior['tests']+['test_c114_joint_sink_census_v1.py']
    if len(tests)!=141 or len(set(tests))!=141:raise ValueError('inherited test-file inventory differs')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('ordinary assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=3280,complete_test_wall_cap_s=60,worker_wall_cap_s=240,
        CPU_threads=1,GPU=False,address_space_bytes=16*1024**3,transient_bytes=1024**3,
        unique_numeric_entries_cap=64_000_000,aggregate_diagnostic_work_cap=256_000_000,
        source_generation_cap_unchanged=[256_000_000,200_000_000],shared_radix_caps_unchanged=[16384,131072,16_000_000],
        failed_C113_qualification_not_reclassified=True,source_runtime_LIVE_admitted=False,
        new_solver_run_authorized=False,promotion_authorized=False,formal_gain=0))
    start=time.monotonic();record=dict(all_stages_passed=False,formal_gain=0)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collected=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collected.stdout)
        ids=[n for n in collected.stdout.splitlines() if n.startswith(('experiments/','act/')) and '::' in n]
        if (collected.returncode or len(ids)!=3280 or len(set(ids))!=3280
                or {n.split('::',1)[0] for n in ids}!={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}):
            raise ValueError('complete test-node inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(nodeids=ids,count=len(ids)))
        remaining=60-(time.monotonic()-start)
        if remaining<=0:raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            tested=subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
        record.update(tests_count=len(ids),tests_exit=tested.returncode,test_wall_s=time.monotonic()-start)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (tested.returncode or sorted(actual)!=sorted(ids)
                or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete mathematical qualification failed')
        print(json.dumps(dict(event='c114_tests_passed',tests=len(ids),wall_s=record['test_wall_s'])),flush=True)
        with (RUN/'worker.log').open('x') as stream:
            job=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
                env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=job.returncode
        result=json.loads((RUN/'result.json').read_text())
        if job.returncode or not result['completed'] or result['source_drift']:raise ValueError('complete census failed')
        record.update(all_stages_passed=True,work=result['work'],stages=list(result['stages']))
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-start,source_drift=any(_sha256(EXP/n)!=h for n,h in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(p.relative_to(RUN)):_sha256(p) for p in RUN.rglob('*') if p.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['all_stages_passed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':
    worker() if '--worker' in sys.argv else main()
