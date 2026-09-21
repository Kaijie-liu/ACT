"""Retain bounded no-solver revision validation, including fresh saved-only audit."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'docs/review_response_validation_20260921_r2.json'
CHECKS = (
    ('scripts/test_review_handoff.py', []),
    ('scripts/test_moe_main_source.py', []),
    ('scripts/test_manuscript_source_contract.py', []),
    ('scripts/test_rebuild_moe_main_tables.py', []),
    ('scripts/test_moe_input_composition.py', []),
    ('scripts/test_review_revision.py', []),
    ('scripts/rebuild_moe_main_tables.py', ['--check']),
    ('scripts/build_review_revision_inventory.py', ['--check']),
    ('scripts/audit_moe_input_composition.py', ['--check']),
    ('scripts/audit_moe_main_source.py', ['--check']),
)


def main():
    if OUTPUT.exists(): raise ValueError('validation receipt already exists')
    records = []
    for name, args in CHECKS:
        # Only the two saved-tensor audits need installed torch to decode tensors.
        if name in ('scripts/audit_moe_main_source.py', 'scripts/audit_moe_input_composition.py'):
            flags = []
        elif name == 'scripts/test_rebuild_moe_main_tables.py':
            # The unchanged legacy test imports its sibling. Keep -S, but allow
            # its script directory; table reconstruction itself still uses -I -S.
            flags = ['-S']
        else:
            flags = ['-I', '-S']
        command = [sys.executable, '-B', *flags, str(ROOT/name), *args]
        started = time.monotonic()
        try:
            p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            record = {'command': command, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr, 'timeout': False}
        except subprocess.TimeoutExpired as exc:
            record = {'command': command, 'returncode': None, 'stdout': str(exc.stdout or ''), 'stderr': str(exc.stderr or ''), 'timeout': True}
        record.update(seconds=time.monotonic()-started, source_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
        records.append(record)
        print(name, record['returncode'], flush=True)
        if record['returncode'] != 0: break
    passed = len(records) == len(CHECKS) and all(r['returncode'] == 0 for r in records)
    out = {'schema': 'REVIEW_RESPONSE_VALIDATION_V1', 'status': 'PASS' if passed else 'FAILED_RETAINED',
           'checks': records, 'new_solver_calls': 0, 'new_model_forwards': 0, 'new_source_propagations': 0,
           'third_party_mathematical_review': False, 'source_complete_positive_proof': False,
           'meaning': 'Arithmetic, identities, document consistency, fresh saved-only rereads; NOT independent network reproof.',
           'prior_attempt': 'docs/review_response_validation_20260921.json',
           'revision_inventory_sha256': hashlib.sha256((ROOT/'docs/review_revision_inventory_20260921_r2.json').read_bytes()).hexdigest()}
    with OUTPUT.open('x') as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write('\n')
    if not passed: raise SystemExit(1)


if __name__ == '__main__':
    main()
