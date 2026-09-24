"""Retain one finite synthetic control attempt; never launches real requests."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time

from scoped_proof.io import ROOT, PYTHON, load, save, sha

MODULES = ['bounded_evidence.tests', 'residual_proof.tests', 'shared_route_residual.tests',
           'shared_route_residual.supervision_tests', 'checked_route_frontier.tests',
           'checked_route_frontier.supervision_tests', 'scoped_source.tests',
           'source_enclosure.tests', 'source_enclosure.portable_tests',
           'full_source.tests', 'scoped_proof.tests']
EXPECTED_TESTS = 137
PROTOCOL = ROOT / 'docs/bounded_evidence_protocol_20260925_r1.md'


def sources():
    cfg = load(ROOT / 'configs/backend_controls/residual_proof_compare_r1.json')
    old = cfg['common']['sources']
    for path, digest in old.items():
        if sha(ROOT / path) != digest: raise ValueError('sealed source changed: ' + path)
    new = [PROTOCOL, *sorted((ROOT / 'bounded_evidence').glob('*.py'))]
    return {**old, **{str(p.relative_to(ROOT)): sha(p) for p in new}}


def run(root):
    from scoped_proof.supervisor import execute
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT / 'data/moe/results'):
        raise ValueError('control artifacts must be under local ignored results')
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    bound = sources()
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               BOUNDED_CONTROL_EVIDENCE_ROOT=str(root / 'pipeline_controls'))
    save(root / 'launch.json', {'schema': 'BOUNDED_PUBLICATION_CONTROLS_R1', 'sources': bound,
         'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'protocol_sha256': sha(PROTOCOL), 'measured_sizes': [2**18, 2**21],
         'calls': ['small_legacy', 'small_stream', 'large_stream', 'large_legacy'],
         'real_requests': 0, 'no_retries': True})
    command = [PYTHON, '-m', 'unittest', *MODULES, '-v']
    test_start = time.monotonic()
    with (root / 'tests.log').open('xb') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
    output = (root / 'tests.log').read_text()
    match = re.search(r'Ran (\d+) tests', output)
    count = int(match[1]) if match else 0
    tests = {'status': 'PASS' if result.returncode == 0 and count == EXPECTED_TESTS else 'FAIL',
             'returncode': result.returncode, 'tests': count, 'seconds': time.monotonic() - test_start,
             'command': command, 'log_sha256': sha(root / 'tests.log')}
    save(root / 'tests.json', tests)
    rows = []
    if tests['status'] == 'PASS':
        for name, size, mode in [('small_legacy', 2**18, 'legacy'), ('small_stream', 2**18, 'stream'),
                                 ('large_stream', 2**21, 'stream'), ('large_legacy', 2**21, 'legacy')]:
            call_start = time.monotonic()
            folder = root / name; folder.mkdir()
            end = call_start + 30
            command = [PYTHON, '-S', '-m', 'bounded_evidence.probe', 'measure', str(folder),
                       '--deadline', str(end - 2), '--mode', mode, '--size', str(size)]
            stage = execute(command, folder / 'worker.log', end - 2, env, 8 * 2**30)
            row = {'id': name, 'mode': mode, 'size': size, 'stage': stage,
                   'budget_seconds': 30, 'status': stage['status'], 'probe': None}
            if stage['status'] == 'COMPLETED':
                row['probe'] = load(folder / 'probe.json')
                record = row['probe']['record']
                if (record['sha256'] != sha(folder / 'payload.json') or
                        record['bytes'] != (folder / 'payload.json').stat().st_size):
                    row['status'] = 'ERROR'
            row['seconds_before_ledger'] = time.monotonic() - call_start
            row['overhead_seconds'] = row['seconds_before_ledger'] - stage['seconds']
            row['includes'] = 'launch/import/generate/serialize/hash/fsync/report/cleanup/reception/hash-check'
            row['excludes'] = 'own final ledger write, timed by enclosing control wall; overrun invalidates'
            if row['seconds_before_ledger'] >= 30: row['status'] = 'TIMEOUT'
            save(folder / 'cost.json', row)
            returned = time.monotonic() - call_start
            if returned >= 30:
                save(folder / 'publication_timeout.json', {'seconds': returned, 'status': 'TIMEOUT'})
                row['status'] = 'TIMEOUT'
            rows.append({'id': name, 'cost_sha256': sha(folder / 'cost.json'),
                         'returned_seconds': returned, 'effective_status': row['status']})
    unchanged = sources() == bound
    report = {'schema': 'BOUNDED_PUBLICATION_CONTROL_RESULT_R1', 'tests': tests, 'rows': rows,
              'sources_unchanged': unchanged, 'real_requests': 0, 'new_real_certificates': 0,
              'seconds_before_summary': time.monotonic() - start}
    save(root / 'summary.json', report)
    print({'tests': tests['status'], 'count': count, 'probes': len(rows), 'root': str(root)}, flush=True)
    if tests['status'] != 'PASS' or not unchanged: raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    run(p.parse_args().root)
