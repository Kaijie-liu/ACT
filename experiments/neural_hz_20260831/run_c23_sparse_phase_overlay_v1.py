"""Exclusive C23 freeze, full regression suite, bounded worker and archives."""

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
DIRECTORY = EXP / 'results/c23_sparse_phase_overlay_20260911_v1'
PRIOR = EXP / 'results/c22_uid_runs_20260911_v1'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    if (_sha256(PRIOR / 'exit.json') != '3abf8d58cfa26032c4d197137363f756e7e14e4b8400c5ffa6b89e6cecfd44d8'
            or _sha256(PRIOR / 'result.json') != 'b88d8556365684a0a02405943f3a2bbe70a682721795c8e089bde56bbf2a57c8'):
        raise ValueError('completed C22 artifacts drift')
    prior = json.loads((PRIOR / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / n) != sha for n, sha in hashes.items()):
        raise ValueError('inherited source/library drift')
    names = [Path(__file__).name, 'c23_sparse_phase_overlay_v1.py', 'c23_phase_overlay_audit_v1.py',
        'c23_sparse_phase_overlay_worker_v1.py', 'test_c23_sparse_phase_overlay_v1.py',
        'C23_SPARSE_PHASE_OVERLAY_PREREG_20260911.md', 'C22_SPARSE_PHASE_OVERLAY_DESIGN_20260911.md',
        'C22_UID_RUNS_AUDIT_20260911.md', 'CHECKPOINT_C22_UID_20260911_SHA256SUMS',
        str(PRIOR / 'exit.json'), str(PRIOR / 'result.json'),
        'results/c10_fused_emission_20260908_v1/fused_hz.pickle',
        'results/c10_fused_emission_20260908_v1/result.json',
        'results/c10_live_relu_20260908_v1/relu78.pickle',
        'results/c10_live_relu_20260908_v1/proof_binding.json',
        'results/c14_early_rejection_census_20260910_v1/single_use_factor_table.npz']
    hashes.update({n: _sha256(EXP / n) for n in names})
    tests = ['test_c23_sparse_phase_overlay_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(EXP / 'c23_sparse_phase_overlay_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'tests': tests, 'command': command, 'wall_cap_s': 240, 'memory_gb': 16,
        'whole_diagnostic_work_cap': 256_000_000, 'overlay_branch_work_cap': 200_000_000,
        'entry_cap': 64_000_000, 'construction_cap_bytes': 1024**3, 'uid_limit': 2**20,
        'solver_authorized': False, 'unit_splice_authorized': False, 'native_ingestion_authorized': False,
        'generation_authorized': False, 'sparse_append_overlay_diagnostic_authorized': True,
        'whole_live_path_authorized': False, 'partial_acceptance_allowed': False, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            completed = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / n) for n in tests)], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = completed.returncode
        if completed.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = completed.returncode
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
