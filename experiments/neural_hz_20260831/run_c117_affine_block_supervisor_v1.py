# SPDX-License-Identifier: AGPL-3.0-or-later
"""Freeze, qualify all inherited tests, then run one fresh read-only census."""
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
RUN = EXP / 'results/c117_affine_block_census_20260922_v1'
EXPECTED_TESTS = 3341


def main():
    if RUN.exists():
        raise FileExistsError(RUN)
    previous = EXP / 'results/c116_row_composition_20260920_v1'
    prior = json.loads((previous / 'preregistered.json').read_text())
    done = json.loads((previous / 'exit.json').read_text())
    if (done['tests_exit'] or done['tests_count'] != 3319
        or done['source_drift'] or done['provenance_drift']
        or _sha256(previous / 'preregistered.json') !=
            '250cb161432eafc2fa1410d0fd54ba7dd805710d645091aba87afbca69f2d2b0'
        or _sha256(previous / 'result.json') !=
            '33172530dbe7b1b742eecfd08ef343e8db1d34875a240939c69e4688cd151c13'):
        raise ValueError('C116 frozen qualification/failure provenance differs')
    # C116 mathematical qualification passed; its physical admission DID NOT.
    # No inherited physical failure is converted into a successful prerequisite.
    hashes = dict(prior['source_sha256'])
    hashes.update({str((previous / n).relative_to(EXP)): h
        for n, h in done['artifacts'].items()})
    hashes[str((previous / 'exit.json').relative_to(EXP))] = _sha256(previous / 'exit.json')
    c107_path = EXP / 'results/c107_canonical_encoding_20260913_v1/preregistered.json'
    c107 = json.loads(c107_path.read_text())
    for name, digest in c107['source_sha256'].items():
        if name in hashes and hashes[name] != digest:
            raise ValueError('C107/C116 prerequisite hash disagreement')
        hashes[name] = digest
    hashes[str(c107_path.relative_to(EXP))] = _sha256(c107_path)
    names = ['C117_AFFINE_BLOCK_CENSUS_PREREG_20260922.md',
        'c117_affine_block_census_v1.py', 'test_c117_affine_block_census_v1.py',
        'c117_affine_block_worker_v1.py', 'run_c117_affine_block_supervisor_v1.py',
        'c102_complete_roots_v1.py', 'c24_dense_graph_v1.py', 'c9_live_runtime_v1.py',
        'c5_corrected_prefix_worker_v1.py', 'c5_runtime_materializer_v2.py']
    for name in names:
        digest = _sha256(EXP / name)
        if name in hashes and hashes[name] != digest:
            raise ValueError('new source list would overwrite frozen dependency')
        hashes[name] = digest
    if any(_sha256(EXP / n) != h for n, h in hashes.items()):
        raise ValueError('complete inherited source/artifact drift')
    tests = prior['tests'] + ['test_c117_affine_block_census_v1.py']
    if len(tests) != 144 or len(set(tests)) != 144:
        raise ValueError('complete inherited/new test-file inventory differs')
    provenance = _provenance(ROOT)
    if provenance != prior['provenance'] or provenance != c107['provenance']:
        raise ValueError('production provenance drift')
    command = list(c107['command'])
    command[1:3] = [str(EXP / 'c117_affine_block_worker_v1.py'), str(RUN)]
    command[command.index('--output') + 1] = str(RUN / 'pipeline_result.json')
    bench, iid = command[3], int(command[4])
    bench_root = Path(command[command.index('--bench-root') + 1]) / bench
    instances = bench_root / 'instances.csv'
    rows = [line.split(',') for line in instances.read_text().splitlines() if line.strip()]
    model = bench_root / rows[iid][0].replace('./', '')
    spec = Path(command[command.index('--vnnlib-root') + 1]) / bench / rows[iid][1].replace('./', '')
    inputs = {str(p.resolve()): _sha256(p) for p in (instances, model, spec)}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None, '', '0'):
        raise ValueError('ordinary assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN / 'preregistered.json', dict(source_sha256=hashes,
        input_sha256=inputs, provenance=provenance, tests=tests,
        required_test_count=EXPECTED_TESTS, complete_test_wall_cap_s=60,
        stage_worker_wall_cap_s=240, fresh_pipeline_command=command,
        cpu_threads=1, gpu_enabled=False, address_space_bytes=16 * 1024**3,
        transient_bytes=1024**3, entries_cap=64_000_000,
        aggregate_diagnostic_work_cap=256_000_000, graph_support_reservation=96_000_000,
        whole_work_cap=256_000_000, branch_work_cap=200_000_000,
        shared_radix_caps=[16384, 131072, 16_000_000],
        numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False,
        current_scope='complete_fresh_source_program_and_consumer_census_only',
        fresh_runtime_authorized=True, solver_run_authorized=False,
        source_runtime_LIVE_admitted=False, promotion_authorized=False, formal_gain=0))
    started = time.monotonic()
    record = dict(all_stages_passed=False, formal_gain=0)
    try:
        test_command = [sys.executable, '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
            *(str(EXP / n) for n in tests)]
        collection = subprocess.run([*test_command, '--collect-only'], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        with (RUN / 'collection.log').open('x') as stream:
            stream.write(collection.stdout)
        ids = [s for s in collection.stdout.splitlines()
            if s.startswith(('experiments/', 'act/')) and '::' in s]
        files = {str((EXP / n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids) != EXPECTED_TESTS or len(set(ids)) != EXPECTED_TESTS
            or {n.split('::', 1)[0] for n in ids} != files):
            raise ValueError('complete exact test node inventory differs')
        _atomic_exclusive_json(RUN / 'inventory.json', dict(nodeids=ids, count=len(ids)))
        left = 60 - (time.monotonic() - started)
        if left <= 0:
            raise subprocess.TimeoutExpired(test_command, 60)
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*test_command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=left)
        record.update(tests_exit=tested.returncode, tests_count=len(ids),
            test_wall_s=time.monotonic() - started)
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
        if (tested.returncode or sorted(actual) != sorted(ids)
            or any(c.find(n) is not None for c in cases for n in ('failure', 'error', 'skipped'))):
            raise ValueError('complete inherited/new mathematical qualification failed')
        print(json.dumps(dict(event='all_tests_passed', tests=len(ids), wall_s=record['test_wall_s'])), flush=True)
        if any(_sha256(EXP / n) != h for n, h in hashes.items()):
            raise ValueError('frozen source changed after qualification')
        with (RUN / 'worker.log').open('x') as stream:
            worker = subprocess.run([sys.executable, str(EXP / 'c117_affine_block_worker_v1.py')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit'] = worker.returncode
        result = json.loads((RUN / 'result.json').read_text())
        if (worker.returncode or not result['completed'] or result.get('failure')
            or result['solver_calls'] or result['source_drift'] or result['input_drift']
            or result['provenance_drift']):
            raise ValueError('complete fresh read-only census failed')
        record.update(all_stages_passed=True, work=result['work'],
            graph_node_count=result['graph_node_count'],
            repeated_complete_program_groups=len(result['census']['repeated_complete_programs']))
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic() - started,
            source_drift=any(_sha256(EXP / n) != h for n, h in hashes.items()),
            input_drift=any(_sha256(Path(n)) != h for n, h in inputs.items()),
            provenance_drift=_provenance(ROOT) != provenance,
            artifacts={str(f.relative_to(RUN)): _sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN / 'exit.json', record)
        print(json.dumps(record), flush=True)
    if (not record['all_stages_passed'] or record['source_drift']
        or record['input_drift'] or record['provenance_drift']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
