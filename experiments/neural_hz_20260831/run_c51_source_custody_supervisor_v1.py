"""One new source-custody qualification; all old closed versions stay closed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c51_source_custody_20260911_v1'
PRIOR=EXP/'results/c50_checked_cleanup_20260911_v1'
ANCHORS={'preregistered.json':'7aeb3f7b91f50eede18108ab112f44f14a679d0be584fdb074ad9d9b2120a3c0',
    'tests.log':'e98fb85478b3871727946e01e84540c9710b79e584b3faeb57ae2b26fa3ec411',
    'test_events.jsonl':'f5fca935a57892a1996f21083475a5ae15441e5f601cace5ea1ed09ef966738c',
    'test_result.json':'11e086295ded704cd55e805aca7a73ea8f27492aa68fc13c23141a80ae270b02',
    'exit.json':'5ba5fdecfd495746cc45710d26ceb8283001741105cec26baa62f12ba45e996d'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(PRIOR/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('closed C50 archive drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited complete source drift')
    tests=['test_c51_source_custody_v1.py',*prior['tests']]
    names=[Path(__file__).name,'c51_source_custody_v1.py','c51_custody_runner_v1.py','c51_compile_reuse_probe_v1.py',
        'c51_origin_collection_fixture_v1.py','c51_origin_definition_fixture_v1.py','c51_span_generator_worker_v1.py',
        'C51_SOURCE_CUSTODY_PREREG_20260911.md','C51_SOURCE_CUSTODY_CONTRACT_20260911.md',
        'C51_PREPARATION_DIAGNOSTICS_20260911.json','GOAL_COMMON_STRUCTURE_FOCUS_AMENDMENT_20260911.md',
        'GOAL_COMMON_STRUCTURE_FOCUS_AMENDMENT_20260911.sha256','C50_CHECKED_CLEANUP_AUDIT_20260911.md',
        'C50_VERIFICATION_HANDOFF_20260911.md',*tests,*(str(PRIOR/n) for n in ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names});provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('Python optimization is forbidden')
    if env.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'):raise ValueError('normal plugin configuration must stay')
    command=[sys.executable,str(EXP/'c51_span_generator_worker_v1.py'),str(DIRECTORY)]
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,
        original_tests=prior['original_tests'],original_ordered_test_count=2104,
        original_ordered_ids_sha256='143258636d9b6840970829d30810ae55649671a30f0e40a43a63cd7302797c0e',
        expected_test_count=2139,expected_phase_reports=6417,provenance=provenance,command=command,
        worker_wall_cap_s=240,test_wall_cap_s=60,address_space_bytes=16*1024**3,
        measured_transient_cap_bytes=1024**3,entry_cap=64_000_000,
        whole_generation_work_cap=256_000_000,nested_generation_branch_cap=200_000_000,
        offline_input_decode_work_cap=256_000_000,independent_report_work_cap=256_000_000,
        independent_report_branch_cap=200_000_000,unchanged_C31_full_source_proof_diagnostic_boundary=True,
        source_first_immutable_compile_custody=True,plain_assertion_mode_authorized=False,
        all_original_tests_assertions_and_candidate_executions_required=True,
        new_actual_generator_authorized_only_after_complete_audited_test_pass=True,
        new_native_or_solver_authorized=False,old_archive_or_production_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    try:
        with (DIRECTORY/'tests.log').open('x') as f:
            r=subprocess.run([sys.executable,str(EXP/'c51_custody_runner_v1.py'),str(DIRECTORY)],
                cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['tests_exit_code']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
        passed=json.loads((DIRECTORY/'test_result.json').read_text())
        if not passed['completed'] or passed['cases']!=2139 or passed['phase_reports']!=6417:
            raise ValueError('complete original population and every phase must pass')
        record['test_result_sha256']=_sha256(DIRECTORY/'test_result.json')
        with (DIRECTORY/'worker.log').open('x') as f:r=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit_code']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s']=exc.timeout;raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,old_C50_failure_unchanged=all(_sha256(PRIOR/n)==sha for n,sha in ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
