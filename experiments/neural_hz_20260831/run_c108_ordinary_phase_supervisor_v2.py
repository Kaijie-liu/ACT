# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reuse unchanged C107 qualification, prove observer, run one bounded diagnosis."""
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
RUN = EXP/'results/c108_ordinary_phase_20260913_v2'
PREVIOUS = EXP/'results/c107_canonical_encoding_20260913_v1'
NAMES = ['C108_ORDINARY_PHASE_PREREG_20260913.md', 'C108_ORDINARY_PHASE_V2_20260913.md', 'c108_ordinary_phase_v2.py',
         'test_c108_ordinary_phase_v2.py', 'c108_ordinary_phase_worker_v2.py',
         'run_c108_ordinary_phase_supervisor_v2.py']


def main():
    """Save every stage and exception automatically, including parent timeout."""
    if RUN.exists():
        raise FileExistsError(RUN)
    prior = json.loads((PREVIOUS/'preregistered.json').read_text())
    closed = json.loads((PREVIOUS/'exit.json').read_text())
    if (_sha256(PREVIOUS/'exit.json') != '4132b99d30698cca49aab824340894dd02b96216a747b8dfe4e82d9f0e3a2914'
            or closed['tests_count'] != 3186 or closed['tests_exit']
            or closed.get('timeout_s') != 240 or closed['source_drift']
            or closed['provenance_drift'] or closed['payment_exit']):
        raise ValueError('complete C107 qualification and honest closure missing')
    if any(_sha256(PREVIOUS/n) != sha for n,sha in closed['artifacts'].items()):
        raise ValueError('C107 artifact drift')
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP/n) != sha for n,sha in hashes.items()):
        raise ValueError('inherited complete source/configuration/ABI drift')
    provenance = _provenance(ROOT)
    if provenance != prior['provenance']:
        raise ValueError('production provenance drift')
    from experiments.neural_hz_20260831.c108_ordinary_phase_v2 import SOURCE_HASHES, _WRAPPER
    hashes.update(SOURCE_HASHES)
    hashes.update({n:_sha256(EXP/n) for n in NAMES})
    hashes[_WRAPPER._h.__file__] = _sha256(Path(_WRAPPER._h.__file__))
    hashes[sys.executable] = _sha256(Path(sys.executable))
    for n in ('preregistered.json','exit.json','inventory.json','tests.xml','payment_result.json','terminal_gate.json'):
        path = PREVIOUS/n
        hashes[str(path.relative_to(EXP))] = _sha256(path)
    if any(_sha256(EXP/n) != sha for n,sha in hashes.items()):
        raise ValueError('new observation dependencies differ')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None, '', '0'):
        raise ValueError('normal assertions required')
    command = list(prior['command'])
    command[:3] = [sys.executable, str(EXP/'c108_ordinary_phase_worker_v2.py'), str(RUN)]
    command[command.index('--output')+1] = str(RUN/'result.json')
    if '--stop-after-layer' in command or command[command.index('--solver-timeout')+1] != '45':
        raise ValueError('ordinary full terminal command differs')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json', dict(source_sha256=hashes,
        provenance=provenance, command=command, diagnostic_only=True, formal_gain=0,
        reused_tests=prior['tests'], reused_test_count=3186,
        reuse_basis='approved S4: unchanged source/configuration/environment and complete C107 receipts',
        new_fixture='test_c108_ordinary_phase_v2.py', fixture_wall_cap_s=60,
        stage_worker_wall_cap_s=240, solver_seconds=45, cpu_threads=1, gpu_enabled=False,
        address_space_bytes=16*1024**3, transient_bytes=1024**3,
        whole_work_cap=256_000_000, branch_work_cap=200_000_000, entries_cap=64_000_000,
        shared_radix_caps=[16384,131072,16_000_000], observer_reserved_work=262144,
        observer_uses_existing_runtime_diagnostic_pool=True, callbacks_cap=512,
        scalar_events_cap=32, serialized_event_bytes_cap=65536,
        observer_allocation_allowance_bytes=262144, full_numeric_hashes_unchanged=True,
        numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False,
        native_payload_separate_C32_boundary=True, fresh_numeric_source_required=True,
        archived_numeric_HZ_runtime_input=False, base_feasibility_required=True,
        library_or_native_objects_replaced=False, promotion_authorized=False))
    record = dict(diagnostic_only=True, formal_gain=0, reused_test_count=3186,
                  fixture_passed=False, diagnostic_boundary_observed=False)
    started = time.monotonic()
    try:
        fixture_command = [sys.executable, '-m', 'pytest', '-q', '-s', '--tb=short',
            '-p', 'no:cacheprovider', str(EXP/'test_c108_ordinary_phase_v2.py'),
            '--junitxml='+str(RUN/'fixture.xml')]
        with (RUN/'fixture.log').open('x') as stream:
            fixture = subprocess.run(fixture_command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60)
        record['fixture_exit'] = fixture.returncode
        cases = ET.parse(RUN/'fixture.xml').findall('.//testcase')
        if (fixture.returncode or len(cases) != 1
                or cases[0].get('name') != 'test_ordinary_nonconvex_phase_observation'
                or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('ordinary observation fixture failed; no target launched')
        record['fixture_passed'] = True
        print(json.dumps(dict(event='c108_fixture_passed', elapsed_s=time.monotonic()-started)), flush=True)
        if any(_sha256(EXP/n) != sha for n,sha in hashes.items()):
            raise ValueError('source drift after observation fixture')
        record['fresh_terminal_started'] = True
        launch = time.monotonic()
        _atomic_exclusive_json(RUN/'worker_launch.json', dict(monotonic_s=launch,
            worker_wall_cap_s=240, utc_unix_s=time.time(), command=command))
        with (RUN/'worker.log').open('x') as stream:
            try:
                worker = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=240)
                record['worker_exit'] = worker.returncode
            except subprocess.TimeoutExpired:
                record['timeout_s'] = 240
                record['timeout_observed_monotonic_s'] = time.monotonic()
                record['worker_elapsed_wall_s'] = record['timeout_observed_monotonic_s']-launch
        events_path = RUN/'events.jsonl'
        phases = []
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                event = json.loads(line)
                if event.get('event') == 'c108_ordinary_phase':
                    phases.append(event)
        record['ordinary_phases'] = phases
        record['diagnostic_boundary_observed'] = any(p['phase'] == 'run_start' for p in phases)
        record['ordinary_run_return_seen'] = any(p['phase'] == 'run_return' for p in phases)
        record['pipeline_result_present'] = (RUN/'result.json').exists()
    except subprocess.TimeoutExpired as exc:
        record['preterminal_timeout_s'] = exc.timeout
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n) != sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT) != provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json', record)
        print(json.dumps(record), flush=True)
    if (not record['fixture_passed'] or not record['diagnostic_boundary_observed']
            or record['source_drift'] or record['provenance_drift'] or record.get('failure')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

