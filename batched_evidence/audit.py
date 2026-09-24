"""Saved-only study audit and cost decomposition. Never calls a proposer."""
import argparse
import copy
import math
from pathlib import Path
import statistics
import sys
import time

from scoped_proof.io import ROOT, load, sha, save

CONFIG = ROOT / 'configs/backend_controls/batched_evidence_study_r1.json'


def cost_check(cost, terminal):
    s = cost['stage']; call = cost['call']; budget = call['budget']
    vals = [cost['seconds_before_ledger'], s['seconds'], cost['overhead_seconds'], terminal['seconds']]
    if (not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in vals)
            or abs(vals[0] - vals[1] - vals[2]) > 1e-8 or vals[3] < vals[0]
            or cost['budget_seconds'] != budget or budget not in (30, 300)
            or s['cleanup_included'] is not True): raise ValueError('full cost does not close')
    if (abs(cost['deadline_monotonic'] - cost['started_monotonic'] - budget) > 1e-8
            or abs(cost['deadline_monotonic'] - cost['work_deadline_monotonic'] - 2) > 1e-8
            or s['deadline_monotonic'] != cost['work_deadline_monotonic']): raise ValueError('deadline changed')
    expected = 'TIMEOUT' if terminal['seconds'] >= budget else s['status']
    if terminal['status'] != expected: raise ValueError('terminal changed')
    if s['status'] == 'COMPLETED' and vals[1] >= budget - 2: raise ValueError('late completed worker')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def bytes_equal(a, b):
    with a.open('rb') as x, b.open('rb') as y:
        while True:
            p, q = x.read(65536), y.read(65536)
            if p != q: return False
            if not p: return True


def review():
    started = time.monotonic(); cfg = load(CONFIG); root = Path(cfg['output'])
    for name, digest in cfg['sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('source drift: ' + name)
    gate = load(ROOT / 'docs/batched_evidence_controls_20260925_r1.json', cfg['gate_sha256'])
    if gate['status'] != 'PASS' or gate['tests'] != 158 or gate['source_hashes'] != cfg['sources']:
        raise ValueError('controls identity/gate')
    launch = load(root / 'launch.json'); execution = load(root / 'execution.json')
    if (launch['config'] != cfg or launch['config_sha256'] != sha(CONFIG)
            or execution['config_sha256'] != sha(CONFIG) or not execution['no_retries']):
        raise ValueError('frozen config mismatch')
    calls, terminals = cfg['calls'], execution['rows']
    if len(calls) != 38 or [c['id'] for c in calls] != [r['id'] for r in terminals]:
        raise ValueError('missing/reordered denominator')
    probes = []; profiles = []; failures = []; cost_rejections = 0
    for call, terminal in zip(calls, terminals):
        if load(root / (call['id'] + '_terminal.json')) != terminal: raise ValueError('terminal changed')
        folder = root / call['id']
        if not terminal['launched']:
            if terminal['status'] != 'NOT_STARTED_RESOURCE' or terminal['seconds'] is not None:
                raise ValueError('unlaunched cost invented')
            failures.append(terminal); continue
        c = load(folder / 'cost.json', terminal['cost_sha256'])
        if c['call'] != call: raise ValueError('method/config changed')
        cost_check(c, terminal)
        for key in ('overhead_seconds', 'seconds_before_ledger', 'budget_seconds'):
            bad = copy.deepcopy(c); bad[key] = -1
            try: cost_check(bad, terminal)
            except ValueError: cost_rejections += 1
            else: raise AssertionError('cost mutation accepted')
        if terminal['status'] != 'COMPLETED': failures.append(terminal); continue
        if call['kind'] == 'probe':
            p = load(folder / 'probe.json')
            if (p['mode'] != call['mode'] or p['synthetic_size'] != call['size'] or p['traced'] != call['traced']
                    or p['record'] != {'sha256': sha(folder / 'payload.json'), 'bytes': (folder / 'payload.json').stat().st_size}
                    or any(p[k] != 0 for k in ('real_requests', 'native_solver_calls', 'complete_real_certificates'))):
                raise ValueError('probe identity/payload/scope')
            times = [p[k] for k in ('generation_seconds', 'serialization_seconds', 'worker_before_report_seconds')]
            if not all(finite(t) for t in times) or times[0] + times[1] > times[2] + 1e-8 or times[2] > c['stage']['seconds']:
                raise ValueError('probe hidden cost')
            if p['traced']:
                if type(p['serialization_traced_peak_bytes']) is not int or p['serialization_traced_peak_bytes'] < 0:
                    raise ValueError('memory measurement missing')
                if p['mode'] != 'legacy' and p['serialization_traced_peak_bytes'] > 2**20:
                    raise ValueError('registered additional-memory gate failed')
            elif p['serialization_traced_peak_bytes'] is not None: raise ValueError('tracing strata mixed')
            if p['mode'] != 'legacy':
                m = p['metrics']
                if (not m['published'] or not m['returned'] or m['bytes_written'] != p['record']['bytes']
                        or not 0 < m['max_block_bytes'] <= 65536 or m['max_fragment_bytes'] > 24576
                        or m['seconds_including_fsync_and_publication'] > p['serialization_seconds']):
                    raise ValueError('streaming policy violated')
                if p['mode'] == 'batch' and (m['max_batch_items'] > 128 or m['max_batch_bytes'] > 16384
                                           or m['batched_scalars'] <= 0): raise ValueError('batch policy drift')
            probes.append({'call': call, 'result': p, 'whole_seconds': terminal['seconds'], 'cost': c})
        else:
            p = load(folder / 'profile.json'); doc = load(folder / 'source.json', p['files']['source']['sha256'])
            bundle = load(folder / 'construction.json', p['files']['construction']['sha256'])
            from residual_proof.check import check
            result = check(doc, bundle, invocation='profile', expected_source_sha256=p['source_sha256'], deadline=time.monotonic() + 300)
            if result != load(folder / 'source_check.json') or result['output_obligations'] != p['retained_duties']:
                raise ValueError('source check mismatch')
            if (p['fixture'] != call['fixture'] or p['original_duties'] != p['excluded_duties'] + p['retained_duties']
                    or p['complete_output_positive_proof'] is not False or p['native_solver_calls'] != 0
                    or p['output_bounds_generated'] != 0 or p['real_requests'] != 0): raise ValueError('profile scope')
            for phase in p['phases']:
                measured = sum(v['seconds'] for v in p['operations'] if v['phase'] == phase['name'])
                if (not all(finite(phase[k]) for k in ('seconds', 'component_seconds', 'other_seconds'))
                        or abs(measured - phase['component_seconds']) > 1e-8
                        or abs(phase['seconds'] - measured - phase['other_seconds']) > 1e-8):
                    raise ValueError('profile component accounting')
            if (not all(finite(v['seconds']) and v['status'] == 'COMPLETED' for v in p['operations'])
                    or sum(v['seconds'] for v in p['phases']) > p['worker_before_report_seconds']
                    or p['worker_before_report_seconds'] > c['stage']['seconds']): raise ValueError('profile full cost')
            totals = {}
            for op in p['operations']:
                key = op['phase'] + ':' + op['role']; totals[key] = totals.get(key, 0.) + op['seconds']
            profiles.append({'call': call, 'whole_seconds': terminal['seconds'], 'cost': c,
                             'result': p, 'components': totals, 'fresh_source_recheck': 'PASS'})
    groups = []; byte_checks = 0
    for size in (2**18, 2**21):
        group = [p for p in probes if p['call']['size'] == size]
        if group:
            reference = root / group[0]['call']['id'] / 'payload.json'
            for p in group:
                if not bytes_equal(reference, root / p['call']['id'] / 'payload.json'): raise ValueError('byte difference')
                byte_checks += 1
        for traced in (False, True):
            for mode in ('legacy', 'stream', 'batch'):
                xs = [p for p in group if p['call']['traced'] == traced and p['call']['mode'] == mode]
                secs = [p['result']['serialization_seconds'] for p in xs]
                groups.append({'size': size, 'traced': traced, 'mode': mode, 'completed': len(xs), 'required': 3,
                    'median_serialization_seconds': statistics.median(secs) if secs else None,
                    'min_seconds': min(secs) if secs else None, 'max_seconds': max(secs) if secs else None,
                    'median_whole_seconds': statistics.median(p['whole_seconds'] for p in xs) if xs else None,
                    'max_traced_peak_bytes': max(p['result']['serialization_traced_peak_bytes'] for p in xs) if xs and traced else None})
    files = {str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(root.rglob('*')) if p.is_file()}
    return {'audit': 'PASS', 'issues': 0, 'config_sha256': sha(CONFIG), 'sources_rechecked': len(cfg['sources']),
            'all_calls': 38, 'probe_completions': len(probes), 'profile_completions': len(profiles),
            'failures': failures, 'byte_differentials': byte_checks, 'cost_mutations_rejected': cost_rejections,
            'groups': groups, 'probes': probes, 'profiles': profiles, 'files': files,
            'real_requests': 0, 'new_real_certificates': 0, 'native_solver_queries': 0,
            'separate_audit_seconds': time.monotonic() - started}


if __name__ == '__main__':
    if not sys.flags.no_site: raise ValueError('python -S required')
    def forbid(event, args):
        if event == 'import' and (args[0].split('.')[0] in ('torch', 'numpy', 'scipy', 'act', 'highspy')
                or args[0] in ('batched_evidence.profile', 'checked_route_frontier.build', 'residual_proof.build',
                               'shared_route_residual.propose', 'full_source.obligations')):
            raise ImportError('no model/solver/producer in saved audit')
        if event.startswith(('subprocess.', 'socket.')) or event in ('os.system', 'os.fork', 'os.exec'):
            raise PermissionError('saved-only audit')
    sys.addaudithook(forbid)
    p = argparse.ArgumentParser(); p.add_argument('--report', type=Path); a = p.parse_args()
    result = review()
    if a.report: save(a.report, result)
    print({k: result[k] for k in ('audit', 'issues', 'all_calls', 'probe_completions', 'profile_completions', 'real_requests', 'separate_audit_seconds')})
