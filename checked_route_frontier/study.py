"""Synthetic complete-cost study only. No interface accepts a real checkpoint.

One <=300s budget includes producer, exact checking, serialization and cleanup.
Partial files survive but cannot turn a timeout/error into a completed proof.
"""
import argparse
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.supervisor import execute
from source_enclosure.format import identity

OUTPUT = ROOT / 'data/moe/results/checked_route_frontier_synthetic_20260924_r1'
CONFIG = ROOT / 'configs/backend_controls/checked_route_frontier_synthetic_r1.json'


def accept(root, plan):
    doc = load(root/'source.json')
    bundle = load(root/'construction.json')
    result = load(root/'check.json')
    count = doc['request']['experts']*(doc['request']['experts']-1)//2*(doc['request']['classes']-1)
    if (identity(doc) != plan['expected_source_sha256'] or
            result['source_sha256'] != identity(doc) or result['bundle_sha256'] != identity(bundle) or
            result['invocation'] != plan['invocation'] or result['required'] != count or
            len(result['rows']) != count or result['native_float_proof'] is not False or
            result['route_changing_established'] is not False):
        raise ValueError('incomplete/mismatched checker receipt')
    allowed = {'POSITIVE_BOUND', 'NONPOSITIVE_BOUND', 'MISSING', 'NO_CANDIDATE'}
    if plan['mode'] == 'frontier':
        allowed.add('DISCHARGED_BY_CHECKED_ROUTE_EXCLUSION')
    if any(row['index'] != i or row['status'] not in allowed for i, row in enumerate(result['rows'])):
        raise ValueError('checker row inventory/status')
    closed = result['proposal_complete'] is True and all(
        row['status'] in ('POSITIVE_BOUND', 'DISCHARGED_BY_CHECKED_ROUTE_EXCLUSION') for row in result['rows'])
    if result['complete_output_positive_proof'] != closed or (result['status'] == 'CHECKED_DECLARED_REAL_GRAPH_REQUEST') != closed:
        raise ValueError('inconsistent checker aggregation')
    return result


def run_one(root, fixture, mode, *, budget=30., commands=None, expected_source_sha256=None):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    if (fixture not in ('prunable', 'tied') or mode not in ('exhaustive', 'frontier') or
            type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 300):
        raise ValueError('fixed synthetic request policy')
    started = time.monotonic()
    deadline = started + budget
    work_deadline = deadline - min(2., budget/10)
    if expected_source_sha256 is None:
        # Controls derive the fixed anchor inside the charged clock. The study
        # supplies its previously frozen anchor; workers still pay generation.
        from checked_route_frontier.fixtures import timing
        expected_source_sha256 = identity(timing(fixture))
    root.mkdir(parents=True, exist_ok=False)
    plan = {'fixture': fixture, 'mode': mode, 'invocation': uuid.uuid4().hex,
            'budget_seconds': budget, 'started_monotonic': started, 'deadline': deadline,
            'work_deadline': work_deadline, 'real_request': False,
            'expected_source_sha256': expected_source_sha256}
    plan_info = save(root/'plan.json', plan)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    phases, checked = [], None
    status, error = 'ERROR', None
    try:
        for phase in ('build', 'check'):
            command = ([PYTHON, '-S', '-m', 'checked_route_frontier.worker', phase, str(root),
                        '--deadline', str(work_deadline)] if commands is None else commands(phase, root, work_deadline))
            begin = time.monotonic()-started
            row = execute(command, root/(phase+'.log'), work_deadline, env, 8*2**30)
            row.update(phase=phase, start_seconds=begin, end_seconds=time.monotonic()-started)
            phases.append(row)
            save(root/(phase+'_stage.json'), row)
            if row['status'] != 'COMPLETED':
                status = row['status']
                break
        else:
            checked = accept(root, plan)
            status = 'COMPLETED_SYNTHETIC_CHECK'  # positive flag is separate
    except Exception as exc:
        status, error, checked = 'ERROR', repr(exc), None
    if time.monotonic() >= work_deadline:
        status, checked = 'TIMEOUT', None
    terminal = {'status': status, 'plan_sha256': plan_info['sha256'], 'phases': phases,
        'error': error, 'seconds_before_publication': time.monotonic()-started,
        'synthetic_positive': checked is not None and checked['complete_output_positive_proof'],
        'new_real_positive': False, 'required': 252,
        'check_sha256': sha(root/'check.json') if (root/'check.json').exists() else None}
    terminal_info = save(root/'terminal.json', terminal)
    elapsed = time.monotonic()-started
    if elapsed >= budget:
        status = 'TIMEOUT'
    stage = sum(r['seconds'] for r in phases)
    cost = {'status': status, 'terminal_sha256': terminal_info['sha256'], 'end_to_end_seconds': elapsed,
        'stage_seconds': stage, 'overhead_seconds': elapsed-stage, 'budget_seconds': budget,
        'sampled_peak_rss': max((r['sampled_peak_rss'] for r in phases), default=0),
        'includes': 'fixture generation, validation, router proofs, propagation, LP construction, exact checks, serialization, imports, cleanup, terminal',
        'excludes': 'final ledger write and later audit; ledger overrun invalidates acceptance'}
    save(root/'cost.json', cost)
    if time.monotonic() >= deadline:
        save(root/'publication_timeout.json', {'status': 'TIMEOUT', 'seconds': time.monotonic()-started})
        status = 'TIMEOUT'
    return {'status': status, 'seconds': time.monotonic()-started, 'root': str(root)}


def run():
    config = load(CONFIG)
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'feat/moe-route-verification':
        raise ValueError('branch')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise ValueError('clean committed freeze required')
    for name, digest in config['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('frozen implementation changed: '+name)
    gate = load(ROOT/config['controls']['path'], config['controls']['sha256'])
    if gate['status'] != 'PASS':
        raise ValueError('controls failed')
    if OUTPUT.exists() or config['output'] != str(OUTPUT):
        raise ValueError('new fixed output only')
    mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable'] < 10*2**30 or os.getloadavg()[0]/os.cpu_count() > .5:
        raise ValueError('resource admission closed')
    OUTPUT.mkdir()
    save(OUTPUT/'launch.json', {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                              'config_sha256': sha(CONFIG), 'config': config})
    rows = []
    for call in config['calls']:
        row = {**call, **run_one(OUTPUT/call['id'], call['fixture'], call['mode'], budget=config['budget_seconds'],
                               expected_source_sha256=config['fixture_sha256'][call['fixture']])}
        rows.append(row)
        print(row, flush=True)
    save(OUTPUT/'execution.json', {'rows': rows, 'new_real_requests': 0, 'new_solver_calls': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--execute-frozen-synthetic', action='store_true')
    args = p.parse_args()
    if not args.execute_frozen_synthetic:
        p.error('explicit synthetic execution required')
    run()
