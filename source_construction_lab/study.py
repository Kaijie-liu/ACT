"""Bounded synthetic timing, not a real request or a change to sealed R1."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.supervisor import execute
from source_enclosure.format import identity

CONFIG = ROOT / 'configs/backend_controls/source_construction_parse_r1.json'
OUTPUT = ROOT / 'data/moe/results/source_construction_parse_synthetic_20260924_r1'


def run_one(root, fixture, mode, *, budget=30., commands=None):
    root = Path(root)
    if root.exists(): raise FileExistsError(root)
    if type(budget) not in (float, int) or not math.isfinite(budget) or not 0 < budget <= 300:
        raise ValueError('bounded total clock required')
    started = time.monotonic(); deadline = started + budget
    work_deadline = deadline - min(2., budget/10)
    root.mkdir(parents=True, exist_ok=False)
    plan = {'fixture': fixture, 'mode': mode, 'budget_seconds': budget,
        'started_monotonic': started, 'work_deadline': work_deadline, 'deadline': deadline}
    plan_record = save(root / 'plan.json', plan)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    status = 'ERROR'; error = None; phases = []
    try:
        for phase in ('build', 'check'):
            begin = time.monotonic()-started
            command = ([PYTHON, '-S', '-m', 'source_construction_lab.worker', phase,
                str(root), '--deadline', str(work_deadline)] if commands is None else commands(phase, root, work_deadline))
            row = execute(command, root / (phase+'.log'), work_deadline, env, 8*2**30)
            row.update(phase=phase, start_seconds=begin, end_seconds=time.monotonic()-started)
            phases.append(row); save(root / (phase+'_stage.json'), row)
            if row['status'] != 'COMPLETED': status = row['status']; break
        else:
            b = load(root / 'build_receipt.json'); c = load(root / 'check_receipt.json')
            e, k = fixture['experts'], fixture['classes']
            if (c['source_sha256'] != b['source']['sha256'] or
                c['construction_sha256'] != b['construction']['sha256'] or
                c['source_sha256'] != sha(root/'source.json') or
                c['construction_sha256'] != sha(root/'construction.json') or
                c['result']['output_obligations'] != e*(e-1)//2*(k-1) or
                c['result']['complete_output_positive_proof'] is not False):
                raise ValueError('missing/changed complete checked construction')
            status = 'COMPLETED_CONSTRUCTION_CHECK_ONLY'
    except Exception as exc:
        status = 'ERROR'; error = repr(exc)
    if time.monotonic() >= work_deadline: status = 'TIMEOUT'
    terminal = {'status': status, 'plan_sha256': plan_record['sha256'], 'phases': phases, 'error': error,
        'seconds_before_terminal': time.monotonic()-started, 'positive_certificate': False,
        'source_hash': sha(root/'source.json') if (root/'source.json').exists() else None,
        'construction_hash': sha(root/'construction.json') if (root/'construction.json').exists() else None}
    terminal_record = save(root/'terminal.json', terminal)
    elapsed = time.monotonic()-started
    if elapsed >= budget: status = 'TIMEOUT'
    cost = {'status': status, 'terminal_sha256': terminal_record['sha256'], 'end_to_end_seconds': elapsed,
        'stage_seconds': sum(r['seconds'] for r in phases),
        'overhead_seconds': elapsed-sum(r['seconds'] for r in phases),
        'sampled_peak_rss': max((r['sampled_peak_rss'] for r in phases), default=0),
        'budget_seconds': budget, 'excludes': 'final cost ledger write and later saved-only review',
        'includes': 'synthetic generation, source validation, construction, serialization, independent check, imports, owned cleanup, terminal'}
    save(root/'cost.json', cost)
    if time.monotonic() >= deadline:
        save(root/'publication_timeout.json', {'seconds': time.monotonic()-started, 'status': 'TIMEOUT'})
        status = 'TIMEOUT'
    return {'status': status, 'seconds': time.monotonic()-started, 'root': str(root)}


def run():
    cfg = load(CONFIG)
    gate = load(ROOT/'docs/source_construction_parse_controls_20260924_r1.json', cfg['control_sha256'])
    if gate['status'] != 'PASS' or gate['tests_run'] != 17 or gate['sources'] != cfg['sources']:
        raise ValueError('control gate mismatch')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'feat/moe-route-verification':
        raise ValueError('registered branch only')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise ValueError('clean implementation freeze required')
    for name, digest in cfg['sources'].items():
        if sha(ROOT/name) != digest: raise ValueError('frozen source changed: '+name)
    if OUTPUT.exists() or cfg['output'] != str(OUTPUT): raise ValueError('new fixed output required')
    mem = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable'] < 10*2**30 or os.getloadavg()[0]/os.cpu_count() > .5:
        raise ValueError('resource admission closed')
    OUTPUT.mkdir()
    save(OUTPUT/'launch.json', {'head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config_sha256': sha(CONFIG), 'config': cfg})
    rows = []
    for item in cfg['calls']:
        value = run_one(OUTPUT/item['id'], cfg['fixtures'][item['fixture']], item['mode'], budget=cfg['budget_seconds'])
        rows.append({**item, **value})
        print(json.dumps(rows[-1]), flush=True)
    save(OUTPUT/'execution.json', {'rows': rows, 'new_real_requests': 0, 'new_solver_calls': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--execute-frozen-synthetic', action='store_true'); a = p.parse_args()
    if not a.execute_frozen_synthetic: p.error('explicit synthetic execution flag required')
    run()
