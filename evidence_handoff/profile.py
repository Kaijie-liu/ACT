"""One bounded saved-evidence profile, no checkpoint, data, proposal or solver.

Selection is the first saved precheck in the already archived cohort (rank0,
input114), not chosen by a positive bound. Profile timing is instrumented and
outside all original request budgets; it cannot rescue the original TIMEOUT.
"""
import argparse
from collections import Counter
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'docs/general_evidence_execution_v1_results.json'
PARENT_SHA = '67cf5602723221df1b6ac1d10a04363de8b71bd385722c8a87f272353b9a21fe'
OUT = ROOT/'data/moe/results/evidence_handoff_profile_20260916_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')


def child():
    from moe_evidence.checker import check_manifest
    from moe_evidence.storage import loader
    if sha(PARENT) != PARENT_SHA:
        raise ValueError('parent archive changed')
    parent = json.loads(PARENT.read_bytes()); row = parent['evidence_rows'][0]
    if row['dataset_index'] != 114 or row['saved_precheck'] is None:
        raise ValueError('fixed first precheck changed')
    raw = ROOT/parent['raw_root']/'rank0_evidence'
    binding = next(r for r in parent['terminal_bindings'] if r['job_id'] == 'rank0_evidence')
    if sha(raw/'terminal.json') != binding['terminal_sha256'] or sha(raw/'request.json') != binding['request_sha256']:
        raise ValueError('original request/terminal changed')
    for name, digest in row['saved_source_sha256'].items():
        if sha(raw/name) != digest:
            raise ValueError('saved source changed: '+name)
    manifest = json.loads((raw/'manifest.json').read_bytes())
    req = json.loads((raw/'request.json').read_bytes())['evidence_request']
    expected = json.loads((raw/'independent.json').read_bytes())
    started = time.monotonic(); loads = Counter(); sizes = {}; load_seconds = [0.]
    def tick():
        if time.monotonic()-started > 280:
            raise TimeoutError('offline profiling cap; no request promotion')
    load = loader(raw, tick)
    def observed(ref):
        t = time.monotonic(); result = load(ref); load_seconds[0] += time.monotonic()-t
        loads[ref['file']] += 1; sizes[ref['file']] = (raw/ref['file']).stat().st_size
        return result
    profiler = cProfile.Profile(); error = None; result = None
    try:
        profiler.enable()
        result = check_manifest(manifest, req, observed, tick=tick)
        if result != expected:
            raise ValueError('saved independent result not reproduced')
    except Exception as exc:
        error = repr(exc)
    finally:
        profiler.disable()
    elapsed = time.monotonic()-started
    stats = pstats.Stats(profiler); functions = []
    for (file, line, function), (primitive, calls, exclusive, cumulative, callers) in stats.stats.items():
        try: file = str(Path(file).relative_to(ROOT))
        except ValueError: pass
        functions.append({'file': file, 'line': line, 'function': function, 'calls': calls,
                          'primitive_calls': primitive, 'exclusive_seconds': exclusive, 'cumulative_seconds': cumulative})
    top = sorted(functions, key=lambda r: r['cumulative_seconds'], reverse=True)[:40]
    relevant = [r for r in functions if (r['file'].startswith(('act/', 'moe_evidence/', 'portable_proof/'))
                and r['function'] in ('identity', 'rational', '_entries', 'check', 'evaluate', 'check_export',
                                     'check_construction', 'check_manifest', 'load', 'strict_json', 'original_bytes'))]
    modules = Counter()
    for r in functions: modules[r['file']] += r['exclusive_seconds']
    result = {'schema': 'SAVED_EVIDENCE_PROFILE_V1', 'status': 'PASS' if error is None else 'FAIL',
              'error': error, 'parent_archive_sha256': PARENT_SHA, 'dataset_index': 114,
              'selection_rule': 'first archived evidence row with saved precheck; fixed before profiling',
              'profile_source_sha256': sha(Path(__file__)), 'original_terminal_sha256': binding['terminal_sha256'],
              'original_terminal': 'TIMEOUT', 'saved_result_reproduced': error is None,
              'checked_status': expected['status'] if error is None else None,
              'positive_obligations': expected['positive_obligations'] if error is None else None,
              'nonpositive_obligations': expected['nonpositive_obligations'] if error is None else None,
              'new_solver_queries': 0, 'request_promotion': False, 'instrumented_wall_seconds': elapsed,
              'timing_scope': 'one cProfile-instrumented saved checker, not an algorithm speed comparison',
              'loader_calls': sum(loads.values()), 'distinct_loaded_files': len(loads),
              'loaded_logical_bytes': sum(loads[n]*sizes[n] for n in loads),
              'distinct_file_bytes': sum(sizes.values()), 'loader_inclusive_seconds': load_seconds[0],
              'loaded_file_counts': dict(loads), 'top_cumulative_functions': top,
              'relevant_functions': relevant,
              'module_exclusive_seconds': dict(modules.most_common(20)),
              'exclusive_time_sum_seconds': sum(modules.values()),
              'limitations': ['cumulative functions overlap; do not sum them',
                             'one old observed request; not cohort or verification performance',
                             'upstream HZ/source/guard/route exclusions remain trusted']}
    save_new(OUT/'profile.json', result)
    if error is not None:
        raise RuntimeError(error)


def run():
    from evidence_cohort.run import environment, resource, resource_ok, wait_owned
    if not resource_ok(resource()):
        raise RuntimeError('profiling resource gate unavailable')
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    with (OUT/'profile.log').open('x') as log:
        p = subprocess.Popen([sys.executable, '-m', 'evidence_handoff.profile', '--child'],
                             cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        outcome = wait_owned(p, started+300)
    ok = outcome['return_code'] == 0 and not outcome['killed'] and (OUT/'profile.json').exists()
    execution = {'status': 'PASS' if ok else 'FAIL', 'outer': outcome,
                 'seconds': time.monotonic()-started, 'new_solver_queries': 0}
    save_new(OUT/'execution.json', execution)
    if not ok:
        raise RuntimeError('saved-evidence profile incomplete; no retry')
    result = json.loads((OUT/'profile.json').read_bytes())
    if result['status'] != 'PASS':
        raise ValueError('profile failed')
    save_new(ROOT/'docs/evidence_handoff_v1_profile.json', {**result, 'execution': execution})
    print(json.dumps({k: result[k] for k in ('status', 'checked_status', 'instrumented_wall_seconds',
                                            'loader_calls', 'distinct_loaded_files', 'loader_inclusive_seconds')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--child', action='store_true')
    child() if p.parse_args().child else run()
