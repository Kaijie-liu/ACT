"""Saved-only independent control audit. No producer, model or solver execution."""
import argparse
import math
from pathlib import Path
import sys
import time

from scoped_proof.io import ROOT, load, sha


def cost(row, returned):
    stage = row['stage']
    values = [row['seconds_before_ledger'], stage['seconds'], row['overhead_seconds'], returned]
    if (not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values)
            or abs(values[0] - values[1] - values[2]) > 1e-8
            or returned < values[0] or row['budget_seconds'] != 30
            or stage['cleanup_included'] is not True):
        raise ValueError('full probe cost/cleanup does not close')
    if row['status'] != 'COMPLETED' or stage['status'] != 'COMPLETED' or returned >= 30:
        raise ValueError('incomplete or late probe; no passed control')
    probe = row['probe']
    if (probe['mode'] != row['mode'] or probe['synthetic_size'] != row['size'] or
            probe['native_solver_calls'] != 0 or probe['complete_output_certificates'] != 0):
        raise ValueError('probe identity/scope')
    timing = [probe[k] for k in ('generation_seconds', 'serialization_seconds', 'worker_before_report_seconds')]
    if (not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in timing)
            or timing[0] + timing[1] > timing[2] + 1e-8 or timing[2] > stage['seconds']):
        raise ValueError('unaccounted generation/serialization work')
    if type(probe['serialization_traced_peak_bytes']) is not int or probe['serialization_traced_peak_bytes'] < 0:
        raise ValueError('invalid traced memory')
    if row['mode'] == 'stream':
        s = probe['metrics']
        if (not s['published'] or not s['returned'] or s['bytes_written'] != probe['record']['bytes']
                or not 0 < s['max_block_bytes'] <= 65536 or s['max_fragment_bytes'] > 24576
                or s['seconds_including_fsync_and_publication'] > probe['serialization_seconds']
                or probe['serialization_traced_peak_bytes'] > 2**20):
            raise ValueError('bounded serialization control failed')


def equal_bytes(left, right):
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(65536), b.read(65536)
            if x != y: return False
            if not x: return True


def review(root):
    started = time.monotonic()
    root = Path(root)
    launch = load(root / 'launch.json'); summary = load(root / 'summary.json')
    for path, digest in launch['sources'].items():
        if sha(ROOT / path) != digest: raise ValueError('control source drift: ' + path)
    tests = load(root / 'tests.json')
    if (tests != summary['tests'] or tests['status'] != 'PASS' or tests['tests'] != 137
            or tests['returncode'] != 0 or sha(root / 'tests.log') != tests['log_sha256']
            or summary['real_requests'] != 0 or summary['new_real_certificates'] != 0
            or summary['sources_unchanged'] is not True):
        raise ValueError('control gate/identity/scope')
    order = ['small_legacy', 'small_stream', 'large_stream', 'large_legacy']
    if launch['calls'] != order or [r['id'] for r in summary['rows']] != order:
        raise ValueError('fixed finite roster incomplete/reordered')
    rows = []
    for item in summary['rows']:
        folder = root / item['id']
        row = load(folder / 'cost.json', item['cost_sha256'])
        cost(row, item['returned_seconds'])
        if (item['effective_status'] != row['status'] or (folder / 'publication_timeout.json').exists()
                or load(folder / 'probe.json') != row['probe']):
            raise ValueError('terminal changed/late')
        record = row['probe']['record']
        if record != {'sha256': sha(folder / 'payload.json'), 'bytes': (folder / 'payload.json').stat().st_size}:
            raise ValueError('payload changed')
        rows.append(row)
    for size in ('small', 'large'):
        if not equal_bytes(root / (size + '_legacy/payload.json'), root / (size + '_stream/payload.json')):
            raise ValueError('canonical byte differential failed')
    # Real source/route/all-output checks on the synthetic complete pipeline;
    # these are unchanged mathematical checks, not a JSON-only acceptance.
    from residual_proof.audit import audit
    pipeline = audit(root / 'pipeline_controls/good')
    if not pipeline['complete_output_positive_proof']:
        raise ValueError('synthetic proof not complete')
    failed = root / 'pipeline_controls/test_watchdog_partial_and_exception_charged_without_acceptance'
    failures = [audit(failed / name) for name in ('hang', 'write_error')]
    if [r['effective_status'] for r in failures] != ['TIMEOUT', 'ERROR']:
        raise ValueError('failed synthetic requests relabelled')
    files = {str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(root.rglob('*')) if p.is_file()}
    return {'status': 'PASS', 'issues': 0, 'tests_passed': 137, 'new_tests': 16,
            'unchanged_tests': 121, 'frozen_source_files': len(launch['sources']),
            'control_sources': launch['sources'], 'raw_root': str(root), 'files': files,
            'probe_results': rows, 'synthetic_pipeline': pipeline, 'failure_controls': failures,
            'real_requests': 0, 'new_real_certificates': 0,
            'separate_audit_seconds': time.monotonic() - started}


if __name__ == '__main__':
    if not sys.flags.no_site: raise ValueError('python -S required')
    def guard(event, args):
        if event == 'import' and args[0].split('.')[0] in ('numpy', 'scipy', 'torch', 'act', 'highspy'):
            raise ImportError('no model or solver in saved audit')
        if event.startswith(('subprocess.', 'socket.')) or event in ('os.system', 'os.fork', 'os.exec'):
            raise PermissionError('saved-only audit')
    sys.addaudithook(guard)
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('--report', type=Path)
    args = p.parse_args(); report = review(args.root)
    if args.report:
        from scoped_proof.io import save
        save(args.report, report)
    print({k: report[k] for k in ('status', 'issues', 'tests_passed', 'frozen_source_files', 'real_requests', 'separate_audit_seconds')})
