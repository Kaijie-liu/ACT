"""Exclusive C120 qualification and ordinary complete-source supervision."""
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
RUN = EXP/'results/c120_denominator_f4_20260922_v1'
EXPECTED_TESTS = 3405


def main():
    if RUN.exists():
        raise FileExistsError(RUN)
    previous = EXP/'results/c119_denominator_f4_20260922_v1'
    prior = json.loads((previous/'preregistered.json').read_text())
    done = json.loads((previous/'exit.json').read_text())
    if (_sha256(previous/'exit.json') !=
            'adf8b07777e65b1022a1939870e109de66187d6e168d272b680ed9e1fbdbfdcd'
        or done['all_stages_passed'] or done['worker_exit'] != 1 or done['tests_exit']
        or done['tests_count'] != 3384 or done['source_drift'] or done['provenance_drift']):
        raise ValueError('complete failed C119 checkpoint identity differs')
    hashes = dict(prior['source_sha256'])
    hashes.update({str((previous/n).relative_to(EXP)): h for n,h in done['artifacts'].items()})
    hashes[str((previous/'exit.json').relative_to(EXP))] = _sha256(previous/'exit.json')
    seal = EXP/'CHECKPOINT_C119_DENOMINATOR_F4_20260922_SHA256SUMS'
    if _sha256(seal) != 'a86304296d4c7363fe1c61ad1c8c7bac7ac04e44366dc6afc5771edb4a29f728':
        raise ValueError('complete failed C119 checkpoint seal differs')
    for line in seal.read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = str((ROOT/name).relative_to(EXP))
        if relative in hashes and hashes[relative] != digest:
            raise ValueError('inherited sealed source conflict')
        hashes[relative] = digest
    hashes[seal.name] = _sha256(seal)
    new_names = ['C120_DENOMINATOR_F4_PREREG_20260922.md',
        'c120_word_f4_v1.py',
        'test_c120_word_f4_v1.py', 'c120_complete_source_fixture_v1.py',
        'c120_denominator_f4_worker_v1.py', 'run_c120_denominator_f4_supervisor_v1.py']
    hashes.update({n:_sha256(EXP/n) for n in new_names})
    if any(_sha256(EXP/n) != h for n,h in hashes.items()):
        raise ValueError('complete frozen dependency or artifact drift')
    tests = prior['tests'] + ['test_c120_word_f4_v1.py']
    if len(tests) != 147 or len(set(tests)) != 147:
        raise ValueError('complete inherited/new test file population differs')
    provenance = _provenance(ROOT)
    if provenance != prior['provenance']:
        raise ValueError('production provenance changed')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None, '', '0'):
        raise ValueError('ordinary assertions required')
    # JSON-only complete work bound, before tests or new numerical work.
    old_result = json.loads((previous/'result.json').read_text())
    old_cases = [previous/(mode+'_complete_source.json') for mode in ('dense','masked')]
    case_json_bytes = sum(path.stat().st_size for path in old_cases)
    c62_upper = sum(v for k,v in old_result['work_parts'].items() if k.startswith('c62_'))
    kernel_saving = (20736-2768)*1024
    complete_work_upper = 253_209_396-kernel_saving+c62_upper+24*(case_json_bytes+65536)+4096+131072
    if (case_json_bytes != 24171 or c62_upper != 13_321_560
        or complete_work_upper != 250_419_860 or complete_work_upper > 256_000_000):
        raise ValueError('complete paid integer-F4 preflight bound differs')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json', dict(source_sha256=hashes,
        provenance=provenance, tests=tests, required_test_count=EXPECTED_TESTS,
        complete_test_wall_cap_s=60, stage_worker_wall_cap_s=240,
        cpu_threads=1, gpu_enabled=False, address_space_bytes=16*1024**3,
        transient_bytes=1024**3, entries_cap=64_000_000,
        aggregate_diagnostic_work_cap=256_000_000,
        complete_work_upper=complete_work_upper, constructor_kernel_saving=kernel_saving,
        report_auth_bytes_upper=89707, complete_C62_work_upper=c62_upper,
        whole_work_cap=256_000_000, branch_work_cap=200_000_000,
        shared_radix_caps=[16384,131072,16_000_000],
        fixed_complete_source_modes=['dense','masked'],
        fixed_source_geometry=dict(C=16,K=32,input=[6,6],output=[4,4]),
        child_source_reserved_work_per_case=32_000_000,
        numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False,
        scope='ordinary_complete_HZ_direct_F2_and_denominator_F4_comparison',
        archived_HZ_restore_authorized=False, original_network_run_authorized=False,
        solver_authorized=False, actual_network_source_or_LIVE_admitted=False,
        promotion_authorized=False, formal_gain=0))
    started = time.monotonic()
    record = dict(all_stages_passed=False, formal_gain=0)
    try:
        command = [sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
            *(str(EXP/n) for n in tests)]
        collection = subprocess.run([*command,'--collect-only'], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        with (RUN/'collection.log').open('x') as stream:
            stream.write(collection.stdout)
        ids = [s for s in collection.stdout.splitlines()
            if s.startswith(('experiments/','act/')) and '::' in s]
        files = {str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids) != EXPECTED_TESTS or len(set(ids)) != EXPECTED_TESTS
            or {n.split('::',1)[0] for n in ids} != files):
            raise ValueError('complete exact test inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json', dict(nodeids=ids,count=len(ids)))
        left = 60-(time.monotonic()-started)
        if left <= 0:
            raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            tested = subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tested.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases = ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual = [c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (tested.returncode or sorted(actual) != sorted(ids)
            or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete inherited/new qualification failed')
        print(json.dumps(dict(event='all_tests_passed',tests=len(ids),wall_s=record['test_wall_s'])),flush=True)
        if any(_sha256(EXP/n) != h for n,h in hashes.items()):
            raise ValueError('source drift after qualification')
        with (RUN/'worker.log').open('x') as stream:
            worker = subprocess.run([sys.executable,str(EXP/'c120_denominator_f4_worker_v1.py')],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit'] = worker.returncode
        result = json.loads((RUN/'result.json').read_text())
        if (worker.returncode or not result['completed'] or result['source_drift']
            or result['provenance_drift'] or result['data']['complete_source_count'] != 2):
            raise ValueError('complete ordinary source comparison failed')
        record.update(all_stages_passed=True, work=result['work'], complete_source_count=2)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n) != h for n,h in hashes.items()),
            provenance_drift=_provenance(ROOT) != provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record)
        print(json.dumps(record),flush=True)
    if not record['all_stages_passed'] or record['source_drift'] or record['provenance_drift']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

