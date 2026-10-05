"""Exclusive fresh live-consumer qualification; automatic failure retention."""

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
DIRECTORY = EXP / 'results/c9_live_relu_20260906_v1'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    prerequisite = EXP / 'results/c9_checkpoint_qualification_20260905_v1'
    if _sha256(prerequisite / 'result.json') != 'f49458396d1862e4e0eccbb60bfa6393025b9fcbe8fe3c29a44ab483ab130c9c':
        raise ValueError('checkpoint prerequisite drift')
    proof, exited = (json.loads((prerequisite / name).read_text()) for name in ('result.json', 'exit.json'))
    if not proof['passed'] or proof['failed_transaction_v1_reclassified'] or exited['worker_exit_code'] != 0 or exited['source_drift'] or exited['provenance_drift']:
        raise ValueError('checkpoint prerequisite not qualified')
    prior = json.loads((prerequisite / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    c5 = json.loads((EXP / 'results/c5_first_terminal_20260905_v1/preregistered.json').read_text())
    for name, sha in c5['source_sha256'].items():
        if name in hashes and hashes[name] != sha:
            raise ValueError('incompatible prerequisite source manifests')
        hashes[name] = sha
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('prior source/library drift')
    names = [Path(__file__).name, 'c9_live_runtime_v1.py', 'c9_live_relu_audit_v1.py',
        'c9_live_relu_worker_v1.py', 'test_c9_live_runtime_v1.py', 'C9_LIVE_RELU_PREREG_20260906.md',
        'CHECKPOINT_C9_RADIX_20260906_SHA256SUMS', 'C9_COMPLETE_SUFFIX_AUDIT_20260906.md',
        'C9_LIVE_DEVELOPMENT_NOTE_20260908.md',
        'results/c9_checkpoint_qualification_20260905_v1/result.json',
        'results/c9_checkpoint_qualification_20260905_v1/exit.json']
    hashes.update({name: _sha256(EXP / name) for name in names})
    tests = ['test_c9_live_runtime_v1.py', *json.loads((EXP / 'results/c9_integrated_suffix_20260905_v1/preregistered.json').read_text())['tests']]
    command = list(c5['command'])
    command[:3] = [sys.executable, str(EXP / 'c9_live_relu_worker_v1.py'), str(DIRECTORY)]
    command[command.index('--output') + 1] = str(DIRECTORY / 'result.json')
    at = command.index('--save-hz-checkpoint')
    del command[at:at + 2]
    command.extend(['--stop-after-layer', '78'])
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'command': command, 'tests': tests, 'wall_cap_s': 240, 'memory_gb': 16, 'formal_gain': 0,
        'solve_authorized': False, 'native_ingestion_after_reference': True,
        'execution_date': '2026-09-08', 'preregistered_lineage_date': '2026-09-06'})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            tests_run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / name) for name in tests)], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = tests_run.returncode
        if tests_run.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = completed.returncode
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
