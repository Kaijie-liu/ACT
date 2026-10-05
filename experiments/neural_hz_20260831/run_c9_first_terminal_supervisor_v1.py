"""Freeze and retain the first ordinary terminal run after live qualification."""

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
DIRECTORY = EXP / 'results/c9_first_terminal_20260908_v1'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    prerequisite = EXP / 'results/c9_live_relu_20260906_v1'
    if _sha256(prerequisite / 'qualification.json') != '17a78cc29c59acb7eba60cd97a9e95766c1e90a9a86ed8d42cec2404484d6d9e':
        raise ValueError('live prerequisite drift')
    proof, exited = (json.loads((prerequisite / name).read_text()) for name in ('qualification.json', 'exit.json'))
    if not proof['passed'] or proof['terminal_solve_executed'] or exited['worker_exit_code'] != 0 or exited['source_drift'] or exited['provenance_drift']:
        raise ValueError('live prerequisite not qualified')
    prior = json.loads((prerequisite / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('prior source/library drift')
    names = [Path(__file__).name, 'c9_first_terminal_worker_v1.py', 'c9_terminal_binding_v1.py',
        'test_c9_terminal_binding_v1.py', 'C9_FIRST_TERMINAL_PREREG_20260908.md',
        'CHECKPOINT_C9_LIVE_RELU_20260908_SHA256SUMS', 'C9_LIVE_RELU_AUDIT_20260908.md',
        'results/c9_live_relu_20260906_v1/qualification.json', 'results/c9_live_relu_20260906_v1/exit.json']
    hashes.update({name: _sha256(EXP / name) for name in names})
    tests = ['test_c9_terminal_binding_v1.py', *prior['tests']]
    command = list(prior['command'])
    command[:3] = [sys.executable, str(EXP / 'c9_first_terminal_worker_v1.py'), str(DIRECTORY)]
    command[command.index('--output') + 1] = str(DIRECTORY / 'result.json')
    at = command.index('--stop-after-layer')
    del command[at:at + 2]
    command.extend(['--save-hz-checkpoint', str(DIRECTORY / 'final_hz.pickle')])
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'command': command, 'tests': tests, 'wall_cap_s': 240, 'memory_gb': 16, 'formal_gain': 0,
        'solve_authorized_only_after_gates': True, 'solver_cap_s': 45,
        'same_representation_runtime': 'c9_live_runtime_v1.py'})
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
