"""Freeze qualification and bounded actual-source consumer diagnostic outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json
EXP = Path(__file__).resolve().parent
RUN = EXP / 'results/c68_local_splice_20260913_v2'


def main():
    if RUN.exists():
        raise FileExistsError(RUN)
    prior = json.loads((EXP / 'results/c67_direct_csr_20260913_v1/preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / n) != sha for n, sha in hashes.items()):
        raise ValueError('frozen C67 dependency drift')
    names = ['C68_LOCAL_SPLICE_PREREG_20260913.md', 'C68_V2_SHARED_DEPENDENCY_REPAIR_20260913.md',
        'test_c68_local_splice_v1.py', 'c68_source_consumer_worker_v1.py', 'run_c68_local_splice_supervisor_v1.py',
        'results/c68_local_splice_20260913_v1/exit.json', 'results/c68_local_splice_20260913_v1/tests.log',
        'c68_local_splice_v1.py',
        'test_c68_local_splice_v2.py', 'c68_source_consumer_worker_v2.py', Path(__file__).name,
        'C67_NATIVE_LINEAGE_HANDOFF_20260913.md', 'C67_DIRECT_CSR_AUDIT_20260913.md',
        'results/c67_direct_csr_20260913_v1/actual/result.json',
        'results/c67_direct_csr_20260913_v1/preflight/result.json',
        'results/c67_direct_csr_20260913_v1/exit.json', 'c32_native_blocks_v1.py']
    hashes.update({n: _sha256(EXP / n) for n in names})
    tests = [*prior['tests'], 'test_c68_local_splice_v2.py']
    provenance = _provenance(ROOT)
    if provenance != prior['provenance']:
        raise ValueError('production candidate/branch changed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None, '', '0'):
        raise ValueError('normal assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN / 'preregistered.json', dict(source_sha256=hashes, provenance=provenance,
        tests=tests, required_test_count=1586, test_wall_cap_s=60, worker_cap_s=240,
        cpu_threads=1, gpu_enabled=False, address_space_bytes=16 * 1024**3,
        transient_cap_bytes=1024**3, entries_cap=64_000_000, whole_work_cap=256_000_000,
        branch_work_cap=200_000_000, native_or_LIVE_admission=False, formal_gain=0))
    started = time.monotonic()
    record = dict(formal_gain=0, qualification_passed=False, source_reader_started=False)
    try:
        command = [sys.executable, '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', *(str(EXP / n) for n in tests)]
        collection = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=60)
        with (RUN / 'collection.log').open('x') as stream:
            stream.write(collection.stdout)
        ids = [s for s in collection.stdout.splitlines() if s.startswith(('experiments/', 'act/')) and '::' in s]
        if collection.returncode or len(ids) != 1586 or len(set(ids)) != 1586:
            raise ValueError('complete old/new test inventory differs')
        _atomic_exclusive_json(RUN / 'inventory.json', dict(count=len(ids), nodeids=ids, frozen_before_test_execution=True))
        left = 60 - (time.monotonic() - started)
        if left <= 0:
            raise subprocess.TimeoutExpired(command, 60)
        with (RUN / 'tests.log').open('x') as stream:
            got = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=left)
        record.update(tests_exit=got.returncode, tests_count=len(ids), tests_wall_s=time.monotonic() - started)
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if got.returncode or sorted(actual) != sorted(ids) or any(c.find(n) is not None for c in cases for n in ('failure', 'error', 'skipped')):
            raise ValueError('qualification failed; actual source not started')
        record['qualification_passed'] = True
        record['source_reader_started'] = True
        with (RUN / 'source.log').open('x') as stream:
            actual = subprocess.run([sys.executable, str(EXP / 'c68_source_consumer_worker_v2.py')], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['source_exit'] = actual.returncode
        if actual.returncode:
            raise ValueError('actual source reader diagnostic failed')
        record['all_declared_diagnostics_completed'] = True
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic() - started,
            source_drift=any(_sha256(EXP / n) != sha for n, sha in hashes.items()),
            provenance_drift=_provenance(ROOT) != provenance,
            artifacts={str(f.relative_to(RUN)): _sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN / 'exit.json', record)
        print(json.dumps(record), flush=True)
    if not record.get('all_declared_diagnostics_completed') or record['source_drift'] or record['provenance_drift']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
