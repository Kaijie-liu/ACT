"""Run from any directory: python -I -S verify.py --bundle-hash ... --statement-hash ..."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
# -I -S supplies interpreter paths only; never add the ACT checkout.
STDLIB = tuple(Path(p).resolve() for p in sys.path if p)
sys.path.insert(0, str(ROOT / 'code'))


def guard(event, args):
    if event == 'open' and not isinstance(args[0], int):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            raise PermissionError('checker is read-only')
        if not path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in STDLIB):
            raise PermissionError('checker cannot read outside bundle/stdlib')
    if event.startswith(('subprocess.', 'socket.')) or event in ('os.system', 'os.exec', 'os.fork'):
        raise PermissionError('external execution/network forbidden')
    if event == 'import' and args[0].split('.')[0] in ('torch', 'numpy', 'scipy', 'highspy', 'gurobipy'):
        raise ImportError('solver/model libraries forbidden')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-hash', required=True)
    parser.add_argument('--statement-hash', required=True)
    args = parser.parse_args()
    if not sys.flags.no_site or not sys.flags.isolated:
        raise ValueError('invoke using python -I -S')
    sys.addaudithook(guard)
    from runtime import verify
    start = time.monotonic()
    result = verify(ROOT, args.bundle_hash, args.statement_hash)
    print(json.dumps({'result': result, 'check_seconds': time.monotonic() - start,
                      'isolated': True, 'site_disabled': True, 'solver_imported': False}, sort_keys=True))


if __name__ == '__main__':
    main()
