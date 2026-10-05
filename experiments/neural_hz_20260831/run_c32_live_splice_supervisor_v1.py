"""Freeze/test/offline-transfer/one native run/restore; exclusively save all exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256
from experiments.neural_hz_20260831.c32_transfer_proof_v1 import ANCHORS

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c32_live_splice_20260911_v1'
PRIOR=EXP/'results/c31_prepared_generator_20260911_v1'
PROOF_SHA='cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('complete independent proof input drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('complete inherited source drift')
    names=[Path(__file__).name,'c32_fresh_lineage_v1.py','c32_native_blocks_v1.py','c32_boundary_budget_v1.py',
        'c32_splice_binding_v1.py','c32_transfer_proof_v1.py','c32_live_splice_runtime_v1.py',
        'c32_native_slot_audit_v1.py','c32_live_splice_worker_v1.py','c32_archive_restore_guard_v1.py',
        'test_c32_live_splice_v1.py','C32_LIVE_SPLICE_PREREG_20260911.md',
        'C31_NATIVE_SPLICE_HANDOFF_20260911.md','CHECKPOINT_C31_PREPARED_GENERATOR_20260911_SHA256SUMS',*ANCHORS]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c32_live_splice_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=list(json.loads((EXP/'results/c9_live_relu_20260906_v1/preregistered.json').read_text())['command'])
    command[:3]=[sys.executable,str(EXP/'c32_live_splice_worker_v1.py'),str(DIRECTORY)]
    command[command.index('--output')+1]=str(DIRECTORY/'result.json')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,provenance=provenance,
        command=command,closed_proof_sha256=PROOF_SHA,wall_cap_s=240,test_wall_cap_s=60,
        transfer_wall_cap_s=60,restore_wall_cap_s=60,memory_gb=16,
        whole_work_cap=256_000_000,branch_work_cap=200_000_000,entry_cap=64_000_000,
        construction_cap_bytes=1024**3,radix_work_cap=16_000_000,
        native_payload_ledger='unchanged_C30_same_assembly_boundary_separate_and_paid',
        source_hash_authentication='separate_measured_diagnostic_not_free_or_full_CPU_work_claim',
        actual_native_first_write_authorized=True,archived_HZ_loaded_by_native_worker=False,
        old_predicate_padding_authorized=False,second_native_HZ_assembly_authorized=False,
        solver_authorized=False,full_replay_authorized=False,old_archive_writes_authorized=False,
        partial_acceptance_allowed=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    def run(args,log,cap):
        with (DIRECTORY/log).open('x') as stream:
            return subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=cap).returncode
    try:
        with (DIRECTORY/'closed_proof.json').open('xb') as f:
            f.write((PRIOR/'closed_proof.json').read_bytes());f.flush();os.fsync(f.fileno())
        record['tests_exit_code']=run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
            *(str(EXP/n) for n in tests)],'tests.log',60)
        if record['tests_exit_code']:return
        record['transfer_exit_code']=run([sys.executable,str(EXP/'c32_transfer_proof_v1.py'),str(DIRECTORY)],'transfer.log',60)
        if record['transfer_exit_code']:return
        transfer=json.loads((DIRECTORY/'transfer_result.json').read_text())
        if not transfer['completed'] or _sha256(DIRECTORY/'transfer_proof.json')!=transfer['transfer_proof_sha256']:
            raise ValueError('offline complete proof transfer failed')
        _atomic_exclusive_json(DIRECTORY/'live_inputs.json',dict(source_proof_sha256=PROOF_SHA,
            transfer_proof_sha256=transfer['transfer_proof_sha256'],transfer_result_sha256=_sha256(DIRECTORY/'transfer_result.json')))
        record['worker_exit_code']=run(command,'worker.log',240)
        if record['worker_exit_code']:return
        record['restore_exit_code']=run([sys.executable,str(EXP/'c32_archive_restore_guard_v1.py'),str(DIRECTORY)],'restore.log',60)
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
