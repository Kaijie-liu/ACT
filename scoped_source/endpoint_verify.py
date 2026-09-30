"""Relocatable H2 verifier: python -B -I -S verify.py --manifest-sha ... --source-sha ...

Only stdlib is imported before identity verification. This read restriction is
an audit policy, not a security sandbox against malicious interpreter code.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
STDLIB = tuple(Path(p).resolve() for p in sys.path if p)
CODE = ('scoped_source/graph.py', 'scoped_source/sparse_ir.py',
        'scoped_source/sparse_check.py', 'source_enclosure/format.py',
        'upstream_source/checker.py', 'router_source/checker.py',
        'scoped_source/endpoint_check.py', 'scoped_source/endpoint_source_check.py')
FILES = {'source.json', 'proof.json', 'verify.py'} | {'code/'+p for p in CODE}


def guard(event, args):
    if event == 'open' and not isinstance(args[0], int):
        path = Path(os.fsdecode(args[0])).resolve(); mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT):
            raise PermissionError('read-only checker')
        stdlib = not {'site-packages','dist-packages'}.intersection(path.parts) and any(path.is_relative_to(p) for p in STDLIB)
        if not path.is_relative_to(ROOT) and not stdlib:
            raise PermissionError('outside checker bundle/stdlib')
    if event.startswith(('subprocess.', 'socket.', 'ctypes.')) or event in ('os.system', 'os.exec', 'os.fork', 'os.posix_spawn', 'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link', 'os.symlink'):
        raise PermissionError('external execution or mutation forbidden')
    if event == 'import' and (args[0].split('.')[0] in ('act','torch','numpy','scipy','highspy','gurobipy','ctypes') or
                             args[0] in ('scoped_source.sparse_build','scoped_source.sparse_controls','local_check',
                                         'scoped_source.endpoint_build','scoped_source.endpoint_source_build',
                                         'scoped_source.endpoint_source_controls','scoped_source.endpoint_controls')):
        raise ImportError('producer/model/solver import forbidden')


def read(path):
    if path.is_symlink() or path.stat().st_size > 64*2**20:
        raise ValueError('symlink or oversized bundle member')
    return path.read_bytes()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def verify(root, expected_manifest, expected_source, expected_mode, deadline):
    def tick():
        if time.monotonic() >= deadline: raise TimeoutError('checker deadline')
    tick(); raw = read(root/'manifest.json')
    if hashlib.sha256(raw).hexdigest() != expected_manifest: raise ValueError('external manifest binding')
    manifest = decode(raw)
    if (manifest['schema'] != 'H2_PORTABLE_DECLARED_SOURCE_V1' or
            manifest['source_sha256'] != expected_source or set(manifest['files']) != FILES or
            expected_mode not in ('endpoints','mccormick') or manifest['mode'] != expected_mode):
        raise ValueError('complete manifest/source binding')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink(): raise ValueError('symlink anywhere in bundle')
        if path.is_file(): actual.add(str(path.relative_to(root)))
    if actual != FILES | {'manifest.json'}: raise ValueError('extra/missing bundle file')
    payload = {}
    for name in sorted(FILES):
        raw = read(root/name); tick()
        if hashlib.sha256(raw).hexdigest() != manifest['files'][name]: raise ValueError('bundle member identity: '+name)
        if name.endswith('.json'): payload[name] = decode(raw)
    # No package code executes until all code, proof and source identities pass.
    sys.path.insert(0, str(root/'code'))
    from scoped_source.endpoint_source_check import check
    result = check(payload['source.json'], payload['proof.json'],
                   expected_source_sha256=expected_source, expected_mode=expected_mode, deadline=deadline)
    tick()
    return {'schema': 'H2_PORTABLE_CHECK_RESULT_V1', 'manifest_sha256': expected_manifest,
            'source_sha256': expected_source, 'invocation': manifest['invocation'], 'mode': expected_mode,
            'result': result, 'isolated': True, 'site_disabled': True,
            'solver_imported': False, 'producer_imported': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--mode', required=True, choices=('endpoints','mccormick'))
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site: raise ValueError('python -I -S required')
    start = time.monotonic(); deadline = args.deadline if args.deadline is not None else start+300
    if not math.isfinite(deadline) or deadline > start+300: raise ValueError('at-most-300-second budget')
    sys.addaudithook(guard)
    result = verify(ROOT, args.manifest_sha, args.source_sha, args.mode, deadline)
    result['check_seconds'] = time.monotonic()-start
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
