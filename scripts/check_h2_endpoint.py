"""Recompute the exact synthetic H2 control; never run a native solver/model.

--write creates a new compact archive, --check independently recomputes it.
This is not a real-request supervisor or an evidence generator benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoped_source.endpoint_controls import report

SOURCES = [
    'configs/h2_endpoint_algebra_20260930.json',
    'scoped_source/endpoint_build.py', 'scoped_source/endpoint_check.py',
    'scoped_source/endpoint_controls.py', 'scoped_source/endpoint_tests.py',
    'scoped_source/sparse_check.py', 'scoped_source/sparse_ir.py',
    'scoped_source/graph.py', 'source_enclosure/format.py',
    'upstream_source/checker.py', 'scripts/check_h2_endpoint.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write', type=Path)
    group.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = report()
    result['implementation_files'] = {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    if args.check:
        if json.loads(args.check.read_text()) != result:
            raise SystemExit('FAIL: recorded H2 result or implementation differs')
        print('PASS: exact synthetic result, all duties and implementation identities unchanged')
    elif args.write:
        destination = args.write.resolve()
        if not destination.is_relative_to(ROOT/'docs'):
            raise SystemExit('archive must be a new file inside repository docs')
        with destination.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(destination)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__': main()
