"""Frozen six-run saved-checker comparison, only after the controls receipt.

No network, checkpoint, dataset, proposal or solver search. Each run is a fresh
process and starts with an empty request cache. Original terminals stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
from statistics import median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'docs/general_evidence_execution_v1_results.json'
PARENT_SHA = '67cf5602723221df1b6ac1d10a04363de8b71bd385722c8a87f272353b9a21fe'
CONTROLS = ROOT/'docs/exact_matrix_cache_v1_controls.json'
OUT = ROOT/'data/moe/results/exact_matrix_cache_benchmark_20260916_v1'
ORDER = ('reference', 'uncached', 'cached', 'cached', 'uncached', 'reference')


def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save_new(p, value):
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')


def verify():
    controls = read(CONTROLS)
    if controls['status'] != 'PASS' or controls['failures'] or controls['errors'] or controls['skipped']:
        raise ValueError('passing controls required before timing')
    for name, digest in controls['sources'].items():
        if sha(ROOT/name) != digest: raise ValueError('controlled source changed: '+name)
    if sha(PARENT) != PARENT_SHA: raise ValueError('parent archive identity differs')
    return controls


def child(position):
    entered = time.monotonic(); verify(); mode = ORDER[position]
    parent = read(PARENT); row = parent['evidence_rows'][0]
    if row['dataset_index'] != 114 or row['saved_precheck'] is None:
        raise ValueError('fixed first archived precheck differs')
    raw = ROOT/parent['raw_root']/'rank0_evidence'
    binding = next(r for r in parent['terminal_bindings'] if r['job_id'] == 'rank0_evidence')
    if sha(raw/'terminal.json') != binding['terminal_sha256'] or sha(raw/'request.json') != binding['request_sha256']:
        raise ValueError('original request/terminal changed')
    for name in ('manifest.json', 'independent.json'):
        if sha(raw/name) != row['saved_source_sha256'][name]: raise ValueError('bound input changed')
    manifest = read(raw/'manifest.json'); req = read(raw/'request.json')['evidence_request']
    expected = read(raw/'independent.json')
    # Import both paths in every process; do not preload or parse proof matrices.
    from moe_evidence.checker import check_manifest as original
    from moe_evidence.storage import loader
    from exact_matrix_cache.checker import check_manifest as candidate
    def tick():
        if time.monotonic()-entered > 280: raise TimeoutError('offline checker work cap')
    load = loader(raw, tick); start = time.monotonic()
    if mode == 'reference':
        result = original(manifest, req, load, tick=tick); cache = None
    else:
        checked = candidate(manifest, req, load, enabled=mode == 'cached', tick=tick)
        result, cache = checked['result'], checked['cache']
    seconds = time.monotonic()-start
    if result != expected: raise ValueError('exact result differs from frozen saved precheck')
    summary = {'position': position, 'mode': mode, 'status': 'PASS', 'dataset_index': 114,
               'result_equal': True, 'checked_status': result['status'],
               'positive_obligations': result['positive_obligations'], 'nonpositive_obligations': result['nonpositive_obligations'],
               'result_sha256': hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
               'checker_seconds': seconds, 'child_seconds': time.monotonic()-entered,
               'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'cache': cache, 'new_solver_queries': 0, 'original_terminal': 'TIMEOUT', 'request_promotion': False}
    save_new(OUT/f'{position}_{mode}.json', summary)


def run():
    verify()
    from scripts.optional_evidence_dev_contract import git, save
    from evidence_cohort.run import environment, resource as resources, resource_ok, wait_owned
    if git('branch', '--show-current') != 'feat/moe-route-verification' or git('status', '--porcelain'):
        raise ValueError('clean published feature checkout required for timing')
    head = git('rev-parse', 'HEAD')
    remote = git('ls-remote', 'origin', 'refs/heads/feat/moe-route-verification').split()[0]
    if remote != head: raise ValueError('publish implementation/controls/protocol before timing')
    if not resource_ok(resources()): raise RuntimeError('offline timing resource gate unavailable; no run started')
    OUT.mkdir(exist_ok=False); rows = []; error = None; started = time.monotonic()
    runtime = {'state': 'RUNNING', 'head': head, 'remote_before_launch': remote, 'order': list(ORDER),
               'controls_sha256': sha(CONTROLS), 'parent_archive_sha256': PARENT_SHA,
               'new_solver_queries': 0, 'rows': rows, 'timing_scope': 'offline checker only, not a verifier rerun'}
    save(OUT/'runtime.json', runtime)
    for i, mode in enumerate(ORDER):
        try:
            verify()
            current = resources()
            if not resource_ok(current): raise RuntimeError('resource gate lost; no timing retry')
            start = time.monotonic()
            with (OUT/f'{i}_{mode}.log').open('x') as log:
                p = subprocess.Popen([sys.executable, '-m', 'exact_matrix_cache.benchmark', '--child', str(i)],
                                     cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                outcome = wait_owned(p, start+300)
            process_seconds = time.monotonic()-start
            if outcome['killed'] or outcome['return_code'] != 0:
                raise RuntimeError('checker timeout/failure: '+str(outcome))
            row = read(OUT/f'{i}_{mode}.json')
            if row['status'] != 'PASS' or row['position'] != i or row['mode'] != mode:
                raise ValueError('worker result mismatch')
            rows.append({**row, 'process_seconds': process_seconds, 'resources': current})
            save(OUT/'runtime.json', runtime)
            print(f'{i+1}/6 {mode} PASS {row["checker_seconds"]:.3f}s', flush=True)
        except Exception as exc:
            error = repr(exc); break
    runtime.update(state='COMPLETED' if error is None else 'FAILED', error=error,
                   unattempted_positions=list(range(len(rows)+(error is not None), len(ORDER))),
                   failed_position=len(rows) if error else None, seconds=time.monotonic()-started)
    save(OUT/'runtime.json', runtime)
    if error: raise RuntimeError(error)
    if len({r['result_sha256'] for r in rows}) != 1: raise ValueError('mode/round result disagreement')
    stats = {}
    for mode in ('reference', 'uncached', 'cached'):
        subset = [r for r in rows if r['mode'] == mode]
        stats[mode] = {'runs': len(subset), 'median_checker_seconds': median(r['checker_seconds'] for r in subset),
                       'median_process_seconds': median(r['process_seconds'] for r in subset),
                       'max_process_rss_kib': max(r['peak_process_rss_kib'] for r in subset)}
    result = {**runtime, 'status': 'PASS', 'summary': stats,
              'reference_over_cached_median_ratio': stats['reference']['median_checker_seconds']/stats['cached']['median_checker_seconds'],
              'uncached_over_cached_median_ratio': stats['uncached']['median_checker_seconds']/stats['cached']['median_checker_seconds'],
              'limitations': ['one stored observed request, two runs/mode; descriptive only',
                             'no cProfile; wall time still subject to shared-server variation',
                             'no complete-request speedup or new SAFE; original TIMEOUT unchanged',
                             'cache limits bound retained representation counts, not total process RSS']}
    save_new(OUT/'summary.json', result)
    save_new(ROOT/'docs/exact_matrix_cache_v1_timing.json', result)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--child', type=int, choices=range(6))
    a = p.parse_args(); run() if a.child is None else child(a.child)
