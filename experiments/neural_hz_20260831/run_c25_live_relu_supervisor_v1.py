"""Exclusive fresh C25 actual native consumer qualification; never a solve."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as baseline
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c25_live_relu_20260911_v1'
PRIOR = EXP / 'results/c24_dense_closed_20260911_v1'
PROOF_SHA = '99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    anchors = {'result.json': '0b87f78d5dc73ab6655ac599dbccce8a59fde37a3b5e9e0781f0a59b399890fe',
        'exit.json': '7dbd873f7e34775b5576fb3f0e29ae0549387c22691aa992a5105c68024db8ea',
        'closed_proof.json': PROOF_SHA,
        'closed_hz.pickle': '8e72ae38cbdafe95f8d8a91928dc2b47a20f4fec5c2d4bd856a377ff267b1be0'}
    if any(_sha256(PRIOR / name) != sha for name, sha in anchors.items()):
        raise ValueError('completed C24 proof/storage/native artifacts drift')
    completed = json.loads((PRIOR / 'result.json').read_text())
    if not (completed['completed'] and completed['strict_complete_offline_reference_decrease']
            and completed['native_ingestion']['passed'] and completed['all_new_graph_fields_physically_retired']):
        raise ValueError('C24 complete proof/physical/native gate not passed')
    prior = json.loads((PRIOR / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / n) != sha for n, sha in hashes.items()):
        raise ValueError('inherited source/library drift')
    names = [Path(__file__).name, 'c25_live_binding_v1.py', 'c25_live_runtime_v1.py',
        'c25_live_relu_worker_v1.py', 'test_c25_live_closed_v1.py',
        'C25_LIVE_CLOSED_PREREG_20260911.md', 'C24_LIVE_HANDOFF_20260911.md',
        'C24_DENSE_CLOSED_AUDIT_20260911.md', 'CHECKPOINT_C24_CLOSED_20260911_SHA256SUMS',
        'c24_archive_restore_guard_v1.py', 'results/c24_archive_restore_guard_20260911_v1/result.json',
        *(str(PRIOR / name) for name in anchors)]
    hashes.update({n: _sha256(EXP / n) for n in names})
    tests = ['test_c25_live_closed_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = list(json.loads((EXP / 'results/c9_live_relu_20260906_v1/preregistered.json').read_text())['command'])
    command[:3] = [sys.executable, str(EXP / 'c25_live_relu_worker_v1.py'), str(DIRECTORY)]
    command[command.index('--output') + 1] = str(DIRECTORY / 'result.json')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes,
        'provenance': provenance, 'tests': tests, 'command': command, 'closed_proof_sha256': PROOF_SHA,
        'wall_cap_s': 240, 'memory_gb': 16, 'work_cap': 256_000_000,
        'entry_cap': 64_000_000, 'branch_work_cap': 200_000_000, 'construction_cap_bytes': 1024**3,
        'radix_work_cap': 16_000_000, 'extra_work_policy': 'coupled_whole_and_branch_pool',
        'independent_diagnostic_work_cap': 256_000_000,
        'actual_native_relu_authorized': True, 'fresh_closed_source_binding_authorized': True,
        'archived_HZ_substitution_authorized': False, 'unit_splice_authorized': False,
        'native_ingestion_authorized_after_complete_live_gate': True, 'solver_authorized': False,
        'partial_acceptance_allowed': False, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        # Copy only independently anchored textual proof, NEVER a completed HZ.
        with (DIRECTORY / 'closed_proof.json').open('xb') as handle:
            handle.write((PRIOR / 'closed_proof.json').read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        with (DIRECTORY / 'tests.log').open('x') as stream:
            run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / n) for n in tests)], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = run.returncode
        if run.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            run = subprocess.run(command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = run.returncode
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - started,
            source_drift=any(_sha256(EXP / n) != sha for n, sha in hashes.items()),
            provenance_drift=baseline._provenance(ROOT) != provenance,
            artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
