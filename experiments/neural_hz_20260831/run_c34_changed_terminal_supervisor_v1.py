"""One changed-matrix ordinary terminal; immutable gates/observations on timeout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c34_final_proof_builder_v1 import ANCHORS
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c34_changed_terminal_20260911_v1'
PRIOR=EXP/'results/c33_live_splice_20260911_v1'
TEXT={
    'closed_proof.json':'cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5',
    'transfer_proof.json':'bab5946186e350159087a9a5d512d8e01759591e0d4309d9d2998e2936b87e86',
}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()) or any(_sha256(PRIOR/n)!=sha for n,sha in TEXT.items()):
        raise ValueError('actual native prerequisite/source proof changed')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('entire inherited dependency drift')
    names=[Path(__file__).name,'c34_terminal_binding_v1.py','c34_final_proof_builder_v1.py',
        'c34_witness_reconstruction_v1.py','c34_terminal_observer_v1.py','c34_changed_terminal_worker_v1.py',
        'c34_final_restore_guard_v1.py','test_c34_terminal_binding_v1.py','test_c34_terminal_observer_v1.py',
        'c10_terminal_observer_v1.py','test_c10_terminal_observer_v1.py','C34_CHANGED_TERMINAL_PREREG_20260911.md',
        'C33_CHANGED_TERMINAL_HANDOFF_20260911.md','CHECKPOINT_C33_NATIVE_SPLICE_20260911_SHA256SUMS',
        '../../act/front_end/specs.py',*ANCHORS]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=list(dict.fromkeys(['test_c34_terminal_binding_v1.py','test_c34_terminal_observer_v1.py',
        'test_c10_terminal_observer_v1.py',*prior['tests']]))
    provenance=_provenance(ROOT)
    command=list(prior['command']);command[:3]=[sys.executable,str(EXP/'c34_changed_terminal_worker_v1.py'),str(DIRECTORY)]
    command[command.index('--output')+1]=str(DIRECTORY/'result.json')
    at=command.index('--stop-after-layer');del command[at:at+2]
    command.extend(['--save-hz-checkpoint',str(DIRECTORY/'final_hz.pickle')])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,provenance=provenance,
        command=command,wall_cap_s=240,test_wall_cap_s=60,final_proof_wall_cap_s=60,restore_wall_cap_s=60,
        memory_gb=16,solver_cap_s=45,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        entry_cap=64_000_000,radix_work_cap=16_000_000,construction_cap_bytes=1024**3,
        witness_diagnostic_work_cap=256_000_000,witness_construction_cap_bytes=1024**3,
        same_actual_representation='c32_live_splice_runtime_v1.py',source_text_sha256=TEXT,
        terminal_authorized_only_after_changed_matrix_and_full_gates=True,base_feasibility_bypass=False,
        all_hash_authentication_and_native_payload_separate_paid_not_free=True,
        old_archive_or_production_writes_authorized=False,other_target_or_full_replay_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    def run(args,log,cap):
        with (DIRECTORY/log).open('x') as stream:
            return subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=cap).returncode
    try:
        for name in TEXT:
            with (DIRECTORY/name).open('xb') as f:f.write((PRIOR/name).read_bytes());f.flush();os.fsync(f.fileno())
        record['tests_exit_code']=run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
            *(str(EXP/n) for n in tests)],'tests.log',60)
        if record['tests_exit_code']:return
        record['proof_exit_code']=run([sys.executable,str(EXP/'c34_final_proof_builder_v1.py'),str(DIRECTORY)],'final_proof.log',60)
        if record['proof_exit_code']:return
        proof=json.loads((DIRECTORY/'final_proof_result.json').read_text())
        if not proof['completed'] or _sha256(DIRECTORY/'final_proof.json')!=proof['final_proof_sha256']:
            raise ValueError('independent final affine proof not completed')
        _atomic_exclusive_json(DIRECTORY/'live_inputs.json',dict(source_proof_sha256=TEXT['closed_proof.json'],
            transfer_proof_sha256=TEXT['transfer_proof.json'],final_proof_sha256=proof['final_proof_sha256']))
        try:record['worker_exit_code']=run(command,'worker.log',240)
        except subprocess.TimeoutExpired:record['worker_timeout_s']=240
        if (DIRECTORY/'terminal_gate.json').exists() and (DIRECTORY/'native_state.pickle').exists():
            record['restore_exit_code']=run([sys.executable,str(EXP/'c34_final_restore_guard_v1.py'),str(DIRECTORY)],'restore.log',60)
    except subprocess.TimeoutExpired as exc:record['stage_timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        observations=[]
        if (DIRECTORY/'events.jsonl').exists():
            for line in (DIRECTORY/'events.jsonl').read_text().splitlines():
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                if event['event'] in ('ordinary_terminal_start','c34_ordinary_milp_start','ordinary_milp_return',
                        'c34_exact_witness_reconstruction_passed','c34_exact_witness_reconstruction_rejected','c34_ordinary_terminal_returned'):
                    observations.append(event)
        result=json.loads((DIRECTORY/'result.json').read_text()) if (DIRECTORY/'result.json').exists() else None
        outcome=dict(formal_gain=0,worker_timed_out=record.get('worker_timeout_s')==240,
            terminal_gate_saved=(DIRECTORY/'terminal_gate.json').exists(),native_state_saved=(DIRECTORY/'native_state.pickle').exists(),
            ordinary_solver_observations=observations,concrete_result=None if result is None else result.get('verdict'),
            prefix_error=None if result is None else result.get('error'),
            invalid_concrete_witnesses=0 if result is None else sum(not v['valid'] for v in result.get('concrete_validations',[])))
        _atomic_exclusive_json(DIRECTORY/'outcome.json',outcome)
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
