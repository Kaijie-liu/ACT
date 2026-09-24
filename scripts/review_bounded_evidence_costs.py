"""Fresh saved-only replay + negative cost controls; no new queries/probes."""
import argparse
import copy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k != 'separate_audit_seconds'}
    if isinstance(value, list): return [stable(v) for v in value]
    return value


def run():
    if not sys.flags.no_site: raise ValueError('python -S required')
    def guard(event, args):
        if event == 'import' and args[0].split('.')[0] in ('torch', 'act', 'numpy', 'scipy', 'highspy'):
            raise ImportError('saved-only replay')
        if event.startswith(('subprocess.', 'socket.')) or event in ('os.system', 'os.fork', 'os.exec'):
            raise PermissionError('no new work')
    sys.addaudithook(guard)
    from bounded_evidence.audit import review, cost
    from scoped_proof.io import load, sha
    started = time.monotonic()
    path = ROOT / 'docs/bounded_evidence_controls_20260925_r1.json'
    frozen = load(path)
    fresh = review(frozen['raw_root'])
    if stable(fresh) != stable(frozen): raise ValueError('fresh archive mismatch')
    rejected = 0
    for row in fresh['probe_results']:
        returned = row['seconds_before_ledger'] + .0001
        cost(row, returned)
        for name in ('budget', 'negative', 'missing_cleanup', 'late', 'nan', 'elapsed',
                     'wrong_mode', 'hidden_generation', 'new_solve'):
            bad = copy.deepcopy(row); end = returned
            if name == 'budget': bad['budget_seconds'] = 301
            if name == 'negative': bad['overhead_seconds'] = -1
            if name == 'missing_cleanup': bad['stage']['cleanup_included'] = False
            if name == 'late': end = 31
            if name == 'nan': bad['seconds_before_ledger'] = float('nan')
            if name == 'elapsed': bad['seconds_before_ledger'] += 1
            if name == 'wrong_mode': bad['probe']['mode'] = 'other'
            if name == 'hidden_generation': bad['probe']['generation_seconds'] = 100
            if name == 'new_solve': bad['probe']['native_solver_calls'] = 1
            try: cost(bad, end)
            except ValueError: rejected += 1
            else: raise AssertionError('cost corruption accepted: ' + name)
    return {'status': 'PASS', 'issues': 0, 'archive_sha256': sha(path),
            'replay_source_sha256': sha(__file__), 'all_source_bindings_rechecked': len(fresh['control_sources']),
            'raw_files_rehashed': len(fresh['files']), 'corrupted_costs_rejected': rejected,
            'complete_synthetic_request_rechecked': True, 'new_real_requests': 0,
            'new_native_solver_calls': 0, 'new_complete_real_certificates': 0,
            'separate_replay_seconds': time.monotonic() - started,
            'scope': 'software evidence replay, not human review or proof of deployed floating-point execution'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--report', type=Path); args = p.parse_args()
    result = run()
    if args.report:
        from scoped_proof.io import save
        save(args.report, result)
    print(result)
