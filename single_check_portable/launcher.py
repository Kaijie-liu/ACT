"""Pinned single-check V3 launcher: python -I -S verify.py; finite absolute deadline."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ENTERED = time.monotonic()
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
STDLIB = tuple(Path(p).resolve() for p in sys.path if p)
DEADLINE = ENTERED  # set before any bundle import


def tick():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError('shared request/check deadline exhausted')


def guard(event, args):
    if event == 'open' and not isinstance(args[0], int):
        tick(); path = Path(os.fsdecode(args[0])).resolve(); mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            raise PermissionError('checker read-only')
        if not path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in STDLIB):
            raise PermissionError('outside bundle/stdlib')
    if event.startswith(('subprocess.', 'socket.')) or event in ('os.system', 'os.exec', 'os.fork'):
        raise PermissionError('external execution/network forbidden')
    if event == 'import' and args[0].split('.')[0] in ('torch', 'numpy', 'scipy', 'highspy', 'gurobipy'):
        raise ImportError('solver/model library forbidden')


def main():
    global DEADLINE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-hash', required=True); p.add_argument('--statement-hash', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--deadline-monotonic', type=float)
    group.add_argument('--timeout-seconds', type=float)
    a = p.parse_args()
    if not sys.flags.no_site or not sys.flags.isolated:
        raise ValueError('requires python -I -S')
    DEADLINE = a.deadline_monotonic if a.deadline_monotonic is not None else ENTERED+a.timeout_seconds
    if not math.isfinite(DEADLINE) or DEADLINE > ENTERED+300:
        raise ValueError('finite deadline at most 300 seconds required')
    tick(); sys.addaudithook(guard)
    # Check every executable byte before importing code from the supplied bundle.
    raw = (ROOT/'bundle.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.bundle_hash:
        raise ValueError('bootstrap bundle identity mismatch')
    meta = json.loads(raw)
    required = {'verify.py', 'code/runtime.py', 'code/transport.py',
                'code/exact_matrix_cache/cache.py', 'code/exact_matrix_cache/checker.py',
                'code/moe_evidence/checker.py', 'code/moe_evidence/schema.py'}
    actual = {str(path.relative_to(ROOT)) for path in (ROOT/'code').rglob('*') if path.is_file()} | {'verify.py'}
    if not required <= actual or not actual <= set(meta['files']):
        raise ValueError('unbound/missing checker source')
    for name in actual:
        tick(); path = ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != meta['files'][name]:
            raise ValueError('bootstrap checker source mismatch')
    sys.path.insert(0, str(ROOT/'code'))
    from runtime import verify
    checked = verify(ROOT, a.bundle_hash, a.statement_hash, tick=tick)
    tick()
    response = {'result': checked['result'], 'cache': checked['cache'],
                'scope': checked['scope'], 'bundle_sha256': a.bundle_hash,
                'decision': 'PACK_ONLY_THEN_ISOLATED_FULL_CHECK',
                'statement_sha256': a.statement_hash, 'deadline_monotonic': DEADLINE,
                'check_seconds': time.monotonic()-ENTERED, 'isolated': True,
                'site_disabled': True, 'solver_imported': False}
    encoded = json.dumps(response, sort_keys=True, allow_nan=False)
    tick(); print(encoded, flush=True); tick()


if __name__ == '__main__':
    try: main()
    except TimeoutError:
        # A prior stdout result, if any, is invalidated by nonzero process exit.
        print('{"status":"TIMEOUT","complete_independent_check":false}', file=sys.stderr)
        sys.exit(3)
