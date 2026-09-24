"""Explicit controls/freeze/execute for finite SYNTHETIC work only."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time

from scoped_proof.io import ROOT, PYTHON, load, save, sha

DOC = ROOT / 'docs/batched_evidence_protocol_20260925_r1.md'
GATE = ROOT / 'docs/batched_evidence_controls_20260925_r1.json'
CONFIG = ROOT / 'configs/backend_controls/batched_evidence_study_r1.json'
DEST = ROOT / 'data/moe/results/batched_evidence_study_20260925_r1'
ENV = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
           CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')


def sources():
    prior = load(ROOT / 'docs/bounded_evidence_controls_20260925_r1.json')['control_sources']
    for name, digest in prior.items():
        if sha(ROOT / name) != digest: raise ValueError('sealed binding changed: ' + name)
    files = [DOC, *sorted((ROOT / 'batched_evidence').glob('*.py'))]
    return {**prior, **{str(p.relative_to(ROOT)): sha(p) for p in files}}


def admitted():
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return available >= 10 * 2**30 and os.getloadavg()[0] / os.cpu_count() <= .5


def clean():
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'feat/moe-route-verification':
        raise ValueError('research branch required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT): raise ValueError('clean checkout required')


def controls(root):
    from bounded_evidence.controls import MODULES
    if not admitted(): raise ValueError('resource gate closed')
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT / 'data/moe/results') or GATE.exists():
        raise ValueError('new local controls root and gate required')
    root.mkdir(parents=True, exist_ok=False); bound = sources(); start = time.monotonic()
    save(root / 'launch.json', {'sources': bound, 'real_requests': 0})
    suites = []
    for name, modules, expected in [('new', ['batched_evidence.tests'], 21), ('previous', MODULES, 137)]:
        begin = time.monotonic(); command = [PYTHON, '-m', 'unittest', *modules, '-v']
        env = dict(ENV, BOUNDED_CONTROL_EVIDENCE_ROOT=str(root / (name + '_pipeline')))
        error = None; code = None
        with (root / (name + '.log')).open('xb') as log:
            try:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240).returncode
            except subprocess.TimeoutExpired as exc: error = str(exc)
        output = (root / (name + '.log')).read_text(); found = re.search(r'Ran (\d+) tests', output)
        count = int(found[1]) if found else 0
        row = {'name': name, 'status': 'PASS' if code == 0 and count == expected else 'FAIL',
               'tests': count, 'expected': expected, 'returncode': code, 'error': error,
               'seconds': time.monotonic() - begin, 'log_sha256': sha(root / (name + '.log'))}
        save(root / (name + '.json'), row); suites.append(row)
    report = {'status': 'PASS' if all(s['status'] == 'PASS' for s in suites) and bound == sources() else 'FAIL',
              'tests': sum(s['tests'] for s in suites), 'suites': suites, 'root': str(root),
              'source_hashes': bound, 'seconds': time.monotonic() - start,
              'real_requests': 0, 'new_real_certificates': 0}
    save(GATE, report); print({k: report[k] for k in ('status', 'tests', 'seconds', 'real_requests')}, flush=True)
    if report['status'] != 'PASS': raise SystemExit(1)


def roster():
    calls = []
    for traced in (False, True):
        for size in (2**18, 2**21):
            for repeat in range(3):
                modes = ['legacy', 'stream', 'batch']
                modes = modes[repeat:] + modes[:repeat]
                for mode in modes:
                    calls.append({'id': f'probe_{int(traced)}_{size}_{repeat}_{mode}', 'kind': 'probe',
                                  'traced': traced, 'size': size, 'repeat': repeat, 'mode': mode, 'budget': 30})
    for fixture in ('small', 'medium'):
        calls.append({'id': 'profile_' + fixture, 'kind': 'profile', 'fixture': fixture, 'budget': 300})
    return calls


def freeze():
    clean(); gate = load(GATE)
    if gate['status'] != 'PASS' or gate['tests'] != 158 or gate['source_hashes'] != sources():
        raise ValueError('test/source gate failed')
    if CONFIG.exists() or DEST.exists(): raise FileExistsError('new freeze/output required')
    save(CONFIG, {'schema': 'BATCHED_EVIDENCE_SYNTHETIC_R1', 'sources': sources(),
        'gate_sha256': sha(GATE), 'calls': roster(), 'output': str(DEST), 'real_requests': 0,
        'rss_limit': 8 * 2**30, 'threads': 2, 'terminal_reserve': 2,
        'batch_items': 128, 'batch_bytes': 16384, 'block_bytes': 65536,
        'no_retry_or_expansion': True,
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    print('FROZEN ONLY: 36 synthetic publication calls + 2 separate unchanged source profiles; zero real requests.')


def execute():
    from scoped_proof.supervisor import execute as watched
    clean(); cfg = load(CONFIG)
    if (cfg['sources'] != sources() or cfg['gate_sha256'] != sha(GATE) or cfg['calls'] != roster()
            or cfg['output'] != str(DEST) or DEST.exists() or not admitted()):
        raise ValueError('frozen source/config/resource gate failed')
    DEST.mkdir()
    save(DEST / 'launch.json', {'config_sha256': sha(CONFIG), 'config': cfg,
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    rows = []
    for call in cfg['calls']:
        begin = time.monotonic(); folder = DEST / call['id']; folder.mkdir()
        if not admitted():
            row = {'id': call['id'], 'status': 'NOT_STARTED_RESOURCE', 'seconds': None, 'launched': False}
        else:
            end = begin + call['budget']; work_end = end - 2
            if call['kind'] == 'probe':
                cmd = [PYTHON, '-S', '-m', 'batched_evidence.probe', 'measure', str(folder),
                       '--deadline', str(work_end), '--mode', call['mode'], '--size', str(call['size'])]
                if call['traced']: cmd.append('--traced')
            else:
                cmd = [PYTHON, '-S', '-m', 'batched_evidence.profile', str(folder),
                       '--deadline', str(work_end), '--fixture', call['fixture']]
            stage = watched(cmd, folder / 'worker.log', work_end, ENV, cfg['rss_limit'])
            cost = {'call': call, 'stage': stage, 'seconds_before_ledger': time.monotonic() - begin,
                    'budget_seconds': call['budget'], 'started_monotonic': begin, 'deadline_monotonic': end,
                    'work_deadline_monotonic': work_end, 'launched': True,
                    'includes': 'imports/generation/construction/checks/publication/owned cleanup',
                    'excludes': 'final ledger write (included in outer return); later audit'}
            cost['overhead_seconds'] = cost['seconds_before_ledger'] - stage['seconds']
            save(folder / 'cost.json', cost)
            seconds = time.monotonic() - begin
            row = {'id': call['id'], 'status': 'TIMEOUT' if seconds >= call['budget'] else stage['status'],
                   'seconds': seconds, 'cost_sha256': sha(folder / 'cost.json'), 'launched': True}
        save(DEST / (call['id'] + '_terminal.json'), row); rows.append(row)
        print({'id': row['id'], 'status': row['status'], 'seconds': row['seconds']}, flush=True)
    save(DEST / 'execution.json', {'rows': rows, 'config_sha256': sha(CONFIG), 'real_requests': 0, 'no_retries': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('controls', 'freeze', 'execute'))
    p.add_argument('--root', type=Path); a = p.parse_args()
    if a.action == 'controls':
        if a.root is None: p.error('--root required')
        controls(a.root)
    elif a.action == 'freeze': freeze()
    else: execute()
