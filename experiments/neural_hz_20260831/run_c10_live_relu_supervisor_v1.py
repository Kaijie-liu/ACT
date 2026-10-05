"""Freeze/export/test/run one actual fused native ReLU qualification."""

import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as baseline
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c10_live_relu_20260908_v1'
FINAL = EXP / 'results/c9_first_terminal_20260908_v1/final_hz.pickle'
LIVE = EXP / 'results/c9_live_relu_20260906_v1/relu78.pickle'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    prior_dir = EXP / 'results/c10_fused_emission_20260908_v1'
    if _sha256(prior_dir / 'result.json') != '315a152e3910b8340f5971be8346434a8d81e5106cd5f67321deea8def29fd5b':
        raise ValueError('sealed census result drift')
    prior = json.loads((prior_dir / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('inherited source drift')
    names = [Path(__file__).name, 'c10_live_runtime_v1.py', 'c10_portable_binding_v1.py',
        'c10_live_relu_worker_v1.py', 'test_c10_live_binding_v1.py',
        'C10_LIVE_RELU_PREREG_20260908.md', 'CHECKPOINT_C10_FUSED_20260908_SHA256SUMS',
        'results/c10_fused_emission_20260908_v1/result.json',
        'results/c10_fused_emission_20260908_v1/exit.json',
        'results/c10_fused_emission_20260908_v1/fused_hz.pickle']

    hashes.update({name: _sha256(EXP / name) for name in names})
    tests = ['test_c10_live_binding_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = list(json.loads((EXP / 'results/c9_live_relu_20260906_v1/preregistered.json').read_text())['command'])
    command[:3] = [sys.executable, str(EXP / 'c10_live_relu_worker_v1.py'), str(DIRECTORY)]
    command[command.index('--output') + 1] = str(DIRECTORY / 'result.json')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes,
        'provenance': provenance, 'tests': tests, 'command': command,
        'wall_cap_s': 240, 'memory_gb': 16, 'work_cap': 256_000_000,
        'entry_cap': 64_000_000, 'branch_work_cap': 200_000_000, 'construction_cap_bytes': 1024**3,
        'radix_work_cap': 16_000_000, 'extra_work_policy': 'coupled_whole_and_branch_pool',
        'transformation_authorized': True, 'native_ingestion_authorized_after_proof': True,
        'solver_authorized': False, 'live_integration_authorized': True, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'binding_export.log').open('x') as stream:
            exported = subprocess.run([sys.executable, '-m',
                'experiments.neural_hz_20260831.c10_portable_binding_v1',
                str(DIRECTORY / 'proof_binding.json')], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['binding_export_exit_code'] = exported.returncode
        if exported.returncode:
            return
        with (DIRECTORY / 'tests.log').open('x') as stream:
            run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / name) for name in tests)], cwd=ROOT, env=environment,
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
            source_drift=any(_sha256(EXP / name) != sha for name, sha in hashes.items()),
            provenance_drift=baseline._provenance(ROOT) != provenance,
            artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
