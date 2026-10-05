"""One new equivalent-cleanup qualification; C49 timeout stays immutable."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c50_checked_cleanup_20260911_v1'
PRIOR=EXP/'results/c49_span_generator_20260911_v1'
ANCHORS={'preregistered.json':'c18ea27cc19ccbfc99f338f3ef8db1f9cf4acb0de02fc7b823805c6df0399462',
    'tests.log':'2cccfbc549ae6b403906edd6d7caa899b0f39e53d5b91964790cb5a1179c456f',
    'exit.json':'e40e089919167dd8c7fbcac402bb00b5abd921d1ba03d72d64b7bc6429117c31'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(PRIOR/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('closed C49 qualification drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited complete source drift')
    tests=['test_c50_cleanup_compile_v1.py',*prior['tests']]
    names=[Path(__file__).name,'c50_assert_protocol_v1.py','c50_cleanup_compile_v1.py','c50_cleanup_probe_v1.py',
        'c50_cleanup_runner_v1.py','c50_cleanup_fixture_v1.py','c50_span_generator_worker_v1.py','C50_CHECKED_CLEANUP_PREREG_20260911.md',
        'C50_CLEANUP_COMPILER_CONTRACT_20260911.md','C50_CLEANUP_PROBE_20260911.json',
        'C49_SPAN_GENERATOR_AUDIT_20260911.md','C49_VERIFICATION_HANDOFF_20260911.md',*tests,
        '/data1/Kane/miniconda3/lib/python3.13/site-packages/_pytest/assertion/rewrite.py',
        '/data1/Kane/miniconda3/lib/python3.13/site-packages/_pytest/assertion/__init__.py',
        '/data1/Kane/miniconda3/lib/python3.13/site-packages/_pytest/fixtures.py',*(str(PRIOR/n) for n in ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names});provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    # Keep normal plugins and assertion rewriting. No plain mode or -O.
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('Python optimization is not allowed')
    if env.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'):raise ValueError('external plugin configuration may not be changed')
    command=[sys.executable,str(EXP/'c50_span_generator_worker_v1.py'),str(DIRECTORY)]
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,
        original_tests=prior['tests'],original_ordered_test_count=2104,
        original_ordered_ids_sha256='143258636d9b6840970829d30810ae55649671a30f0e40a43a63cd7302797c0e',
        provenance=provenance,command=command,worker_wall_cap_s=240,test_wall_cap_s=60,
        address_space_bytes=16*1024**3,measured_transient_cap_bytes=1024**3,entry_cap=64_000_000,
        whole_generation_work_cap=256_000_000,nested_generation_branch_cap=200_000_000,
        offline_input_decode_work_cap=256_000_000,independent_report_work_cap=256_000_000,
        independent_report_branch_cap=200_000_000,unchanged_C31_full_source_proof_diagnostic_boundary=True,
        only_proved_defined_function_local_rewriter_cleanup_changed=True,plain_assertion_mode_authorized=False,
        all_original_tests_assertions_and_failure_evaluations_required=True,
        new_actual_generator_authorized_only_after_complete_audited_test_pass=True,
        new_native_or_solver_authorized=False,old_archive_or_production_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    try:
        with (DIRECTORY/'tests.log').open('x') as f:
            r=subprocess.run([sys.executable,str(EXP/'c50_cleanup_runner_v1.py'),str(DIRECTORY)],
                cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['tests_exit_code']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
        passed=json.loads((DIRECTORY/'test_result.json').read_text())
        if not passed['completed'] or passed['cases']!=2120 or passed['phase_reports']!=6360:
            raise ValueError('complete all-case/all-phase test proof missing')
        record['test_result_sha256']=_sha256(DIRECTORY/'test_result.json')
        with (DIRECTORY/'worker.log').open('x') as f:r=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit_code']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s']=exc.timeout;raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,old_C49_timeout_unchanged=all(_sha256(PRIOR/n)==sha for n,sha in ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
