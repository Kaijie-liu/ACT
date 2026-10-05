"""One full qualification run; no target/network job or background test loop."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import (
    _sha256, _atomic_exclusive_json)

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c86_complete_tile_20260913_v1'


def main():
    """Freeze exact test/source inventory and retain success or failure."""
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    if RUN.exists():
        raise FileExistsError(RUN)
    previous = json.loads((EXP/'results/c85_exact_conv_20260913_v1/preregistered.json').read_text())
    previous_end = json.loads((EXP/'results/c85_exact_conv_20260913_v1/exit.json').read_text())
    full = json.loads((EXP/'results/c84_affine_recipe_20260913_v1/preregistered.json').read_text())
    hashes, provenance = dict(previous['source_sha256']), _provenance(ROOT)
    if (not previous_end['all_declared_stages_passed'] or previous_end['source_drift']
            or previous_end['provenance_drift'] or provenance != previous['provenance']
            or any(_sha256(EXP/n) != h for n, h in hashes.items())):
        raise ValueError('complete unchanged C85 source/front gate required')
    names = ['C86_COMPLETE_TILE_HZ_PREREG_20260913.md', 'c86_complete_tile_hz_v1.py',
             'test_c86_complete_tile_hz_v1.py', Path(__file__).name,
             'results/c85_exact_conv_20260913_v1/preregistered.json',
             'results/c85_exact_conv_20260913_v1/exit.json',
             'results/c85_exact_conv_20260913_v1/result.json']
    hashes.update({n: _sha256(EXP/n) for n in names})
    tests = full['tests']+['test_c85_exact_conv_v1.py', 'test_c86_complete_tile_hz_v1.py']
    expected_count = 2039
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None, '', '0'):
        raise ValueError('normal assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json', dict(
        source_sha256=hashes, provenance=provenance, tests=tests, required_test_count=expected_count,
        cpu_threads=1, gpu_enabled=False, address_space_bytes=16*1024**3,
        test_wall_cap_s=60, per_build_work_cap=256_000_000, full_inherited_suite_rerun=True,
        fresh_original_network_native_authorized=False, terminal_solve_or_promotion_authorized=False,
        full_physical_resource_admission_claimed=False, formal_gain=0))
    record = dict(qualification_passed=False, formal_gain=0)
    started = time.monotonic()
    try:
        cmd = [sys.executable, '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
               '-o', 'junit_family=legacy', *(str(EXP/n) for n in tests)]
        collection = subprocess.run([*cmd, '--collect-only'], cwd=ROOT, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        with (RUN/'collection.log').open('x') as stream:
            stream.write(collection.stdout)
        ids = [s for s in collection.stdout.splitlines()
               if s.startswith(('experiments/', 'act/')) and '::' in s]
        expected_files = {str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids) != expected_count or len(set(ids)) != expected_count
                or {s.split('::', 1)[0] for s in ids} != expected_files):
            raise ValueError('exact complete qualification inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json', dict(count=len(ids), nodeids=ids,
                                                        frozen_before_test_execution=True))
        left = 60-(time.monotonic()-started)
        if left <= 0:
            raise subprocess.TimeoutExpired(cmd, 60)
        with (RUN/'tests.log').open('x') as stream:
            tested = subprocess.run([*cmd, '--junitxml='+str(RUN/'tests.xml')], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=left)
        record.update(tests_exit=tested.returncode, tests_count=len(ids), test_files=len(tests),
                      test_wall_s=time.monotonic()-started)
        cases = ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual = [c.get('classname', '').replace('.', '/')+'.py::'+c.get('name', '') for c in cases]
        if (tested.returncode or sorted(actual) != sorted(ids)
                or any(c.find(tag) is not None for c in cases for tag in ('failure', 'error', 'skipped'))):
            raise ValueError('full mathematical qualification failed')
        fixtures = [dict(name=c.get('name'), properties={p.get('name'): p.get('value')
                    for p in c.findall('./properties/property')}) for c in cases
                    if 'test_complete_all_coordinate_projection_and_actual_extension' in c.get('name', '')]
        _atomic_exclusive_json(RUN/'result.json', dict(completed=True, full_suite_pass=True,
            full_tile_equation_fixtures=fixtures, original_source_HZ_bound=False,
            whole_HZ_physical_gate_pass=False, solver_executed=False, formal_gain=0))
        record.update(qualification_passed=True, all_declared_stages_passed=True)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n) != h for n, h in hashes.items()),
            provenance_drift=_provenance(ROOT) != provenance,
            artifacts={str(f.relative_to(RUN)): _sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json', record)
        print(json.dumps(record), flush=True)
    if not record.get('all_declared_stages_passed') or record['source_drift'] or record['provenance_drift']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
