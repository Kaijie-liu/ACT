# SPDX-License-Identifier: AGPL-3.0-or-later
"""Freeze all inherited qualification and run the complete column population."""
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
RUN = EXP / 'results/c118_column_orbits_20260922_v1'
EXPECTED_TESTS = 3363


def main():
    if RUN.exists():
        raise FileExistsError(RUN)
    previous = EXP / 'results/c117_affine_block_census_20260922_v1'
    prior = json.loads((previous / 'preregistered.json').read_text())
    done = json.loads((previous / 'exit.json').read_text())
    if (_sha256(previous / 'exit.json') !=
            '381e5448b7b092e47a558ddaee824323953130c13342511712caad0a4c225da7'
        or done['tests_exit'] or done['tests_count'] != 3341
        or done['source_drift'] or done['input_drift'] or done['provenance_drift']
        or done['all_stages_passed']):
        raise ValueError('C117 complete mathematical qualification/failure identity changed')
    hashes = dict(prior['source_sha256'])
    hashes.update({str((previous / n).relative_to(EXP)): h for n,h in done['artifacts'].items()})
    hashes[str((previous / 'exit.json').relative_to(EXP))] = _sha256(previous / 'exit.json')
    seal = EXP / 'CHECKPOINT_C117_AFFINE_BLOCK_20260922_SHA256SUMS'
    if _sha256(seal) != '4761e8da4c43244378d789b58058f298fe01fe76e3b2e873caa32202016cd3b7':
        raise ValueError('C117 complete checkpoint seal differs')
    for line in seal.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = ROOT / name
        relative = str(path.relative_to(EXP))
        if relative in hashes and hashes[relative] != digest:
            raise ValueError('inherited sealed source conflict')
        hashes[relative] = digest
    hashes[seal.name] = _sha256(seal)
    new_names = ['C118_COLUMN_ORBITS_PREREG_20260922.md', 'c118_column_orbits_v1.py',
        'test_c118_column_orbits_v1.py', 'c118_column_orbit_worker_v1.py',
        'run_c118_column_orbit_supervisor_v1.py']
    hashes.update({n:_sha256(EXP / n) for n in new_names})
    if any(_sha256(EXP / n) != h for n,h in hashes.items()):
        raise ValueError('complete frozen dependency/artifact drift')
    tests = prior['tests'] + ['test_c118_column_orbits_v1.py']
    if len(tests) != 145 or len(set(tests)) != 145:
        raise ValueError('complete inherited/new test-file population differs')
    provenance = _provenance(ROOT)
    if provenance != prior['provenance']:
        raise ValueError('production provenance changed')
    # JSON-only population/work preflight, before any numerical NPZ load.
    graph = json.loads((previous / 'complete_census.json').read_text())['report']['nodes']
    selected = [n for n in graph if n['operator_geometry'] is not None
        and n['operator_geometry']['type'] == 'csr']
    preflight = []
    for node in selected:
        geometry = node['operator_geometry']
        m,n = geometry['shape']; z = geometry['stored_nnz']
        load = 16 * (2*z + m + 1)
        census = load + 16*(2*z+n+1) + 16*(2*z+n) + 256*n + 16*n
        preflight.append(dict(node=node['node'], shape=[m,n], nnz=z,
            archive_load_work=load, complete_census_work=census))
    if (len(selected) != 13 or sum(r['archive_load_work'] for r in preflight) != 48_075_984
        or sum(r['complete_census_work'] for r in preflight) != 192_189_600):
        raise ValueError('complete fixed operator fee preflight changed')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):
        raise ValueError('ordinary assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN / 'preregistered.json', dict(source_sha256=hashes,
        provenance=provenance, tests=tests, required_test_count=EXPECTED_TESTS,
        complete_test_wall_cap_s=60, stage_worker_wall_cap_s=240,
        cpu_threads=1, gpu_enabled=False, address_space_bytes=16*1024**3,
        transient_bytes=1024**3, entries_cap=64_000_000,
        complete_operator_preflight=preflight, aggregate_diagnostic_work_cap=256_000_000,
        whole_work_cap=256_000_000, branch_work_cap=200_000_000,
        shared_radix_caps=[16384,131072,16_000_000],
        numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False,
        scope='all_archived_explicit_CSR_columns_exact_orbit_census',
        archived_HZ_restore_authorized=False, fresh_original_network_run_authorized=False,
        solver_authorized=False, source_or_LIVE_admitted=False,
        promotion_authorized=False, formal_gain=0))
    started = time.monotonic()
    record = dict(all_stages_passed=False, formal_gain=0)
    try:
        command = [sys.executable, '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
            *(str(EXP / n) for n in tests)]
        collection = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        with (RUN / 'collection.log').open('x') as stream:
            stream.write(collection.stdout)
        ids = [s for s in collection.stdout.splitlines()
            if s.startswith(('experiments/','act/')) and '::' in s]
        files = {str((EXP / n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids) != EXPECTED_TESTS or len(set(ids)) != EXPECTED_TESTS
            or {n.split('::',1)[0] for n in ids} != files):
            raise ValueError('complete exact test inventory differs')
        _atomic_exclusive_json(RUN / 'inventory.json', dict(nodeids=ids,count=len(ids)))
        left = 60-(time.monotonic()-started)
        if left <= 0:
            raise subprocess.TimeoutExpired(command,60)
        with (RUN / 'tests.log').open('x') as stream:
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
            worker = subprocess.run([sys.executable,str(EXP/'c118_column_orbit_worker_v1.py')],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit'] = worker.returncode
        result = json.loads((RUN/'result.json').read_text())
        if (worker.returncode or not result['completed'] or result['source_drift']
            or result['provenance_drift'] or result['data']['complete_operator_count'] != 13):
            raise ValueError('complete explicit-operator census failed')
        reports = result['data']['reports'].values()
        record.update(all_stages_passed=True, work=result['work'],
            complete_operator_count=13,
            strict_nnz_positive_groups=sum(r['strict_nnz_positive_group_count'] for r in reports),
            conditional_nnz_saving=sum(r['candidate_conditional_nnz_saving'] for r in reports))
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
