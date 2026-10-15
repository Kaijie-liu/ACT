"""V2 portable check with externally anchored producer/consumer selection."""
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY, compact, decode, digest, identity, require, sha, Work
from scoped_source.factored_io import read
from scoped_source.check_direct_source import CheckedDirectForest
from scoped_source.scaled_endpoints import contract
from scoped_source.direct_candidates import Driver, MAX_TILE, PREFIX
from scoped_source.wire_package import inventory
from scoped_source.direct_package import CODE_FILES as V1_CODE
from scoped_source.direct_method import validate

MANIFEST = 'DIRECT_LIFECYCLE_MANIFEST_V1'
BUNDLE = 'DIRECT_LIFECYCLE_BUNDLE_V1'
CODE_FILES = V1_CODE + ('scoped_source/direct_method.py', 'scoped_source/direct_lifecycle_package.py')


def check_manifest(root, raw, *, expected, expected_method, invocation, work, enabled=False):
    require(enabled is True, 'lifecycle portable is opt-in')
    method = validate(expected_method)
    sha(invocation)
    work.tick()
    require(type(raw) is bytes and len(raw) <= 2**20 and digest(raw) == sha(expected), 'manifest identity')
    m = decode(raw)
    require(type(m) is dict and compact(m) == raw and set(m) == {
        'schema', 'source', 'forest', 'contract', 'method', 'invocation', 'tiles', 'code'}
        and m['schema'] == MANIFEST and m['invocation'] == invocation
        and identity(m['method']) == identity(method), 'frozen manifest method/invocation')

    def member(ref, cap):
        require(type(ref) is dict and set(ref) == {'file', 'bytes', 'sha256'}
                and type(ref['bytes']) is int and 0 < ref['bytes'] <= cap, 'bounded member')
        work.tick()
        value = read(root, ref['file'], cap, sha(ref['sha256']))
        work.tick()
        require(len(value) == ref['bytes'], 'member length')
        return value

    require(type(m['code']) is list and [r['file'] for r in m['code']] == ['code/' + n for n in CODE_FILES],
            'complete checker code inventory')
    for ref in m['code']:
        member(ref, 64 * 2**10)
    require(type(m['source']) is dict and set(m['source']) == {'directory', 'sha256'}
            and m['source']['directory'] == 'source', 'local source')
    require(m['forest']['file'] == 'forest.bin' and m['contract']['file'] == 'contract.json', 'fixed source members')
    f = CheckedDirectForest(member(m['forest'], POLICY['total_bytes']), Path(root) / 'source', m['source']['sha256'],
                            expected=m['forest']['sha256'], work=work, enabled=True)
    cr = member(m['contract'], 8192)
    c = decode(cr)
    require(compact(c) == cr and c == contract(f), 'complete property contract')
    d = Driver(f, c, expected=identity(c), work=work, pair_scoped=method['replay']['kind'] == 'pair_scoped')
    try:
        require(type(m['tiles']) is list and len(m['tiles']) == d.session.tile_count
                and [r['file'] for r in m['tiles']] == [f'tiles/{i:04d}.bin' for i in range(d.session.tile_count)],
                'all unique ordered tiles')
        for ref in m['tiles']:
            _, _, token = d.prepare()
            raw = member(ref, MAX_TILE)
            require(len(raw) >= PREFIX.size and PREFIX.unpack_from(raw)[-2] == method['producer']['profile'],
                    'actual candidate scalar profile')
            d.receive(token, raw, lambda i, b: (digest(b), len(b)))
        result = d.finish()
        return dict(result=result, owner=dict(d.owner.stats), driver_cost=dict(d.costs), candidate_bytes=d.total_bytes,
                    consumer=method['replay'])
    finally:
        if not d.closed:
            d.abort()


def check_package(root, *, expected, expected_method, invocation, work, enabled=False):
    require(enabled is True, 'lifecycle package is opt-in')
    start = time.monotonic()
    refs, total = inventory(root, expected, work, schema=BUNDLE)
    ref = refs['manifest.json']
    raw = read(root, 'manifest.json', 2**20, ref['sha256'])
    result = check_manifest(root, raw, expected=ref['sha256'], expected_method=expected_method,
                            invocation=invocation, work=work, enabled=True)
    inventory(root, expected, work, schema=BUNDLE)
    work.tick()
    return dict(**result, package_sha256=expected, package_bytes=total, method=validate(expected_method),
                invocation=invocation, work=dict(work.counts), seconds=time.monotonic() - start)


def main():
    import argparse
    from scoped_source.direct_method import method
    p = argparse.ArgumentParser()
    p.add_argument('root'); p.add_argument('expected'); p.add_argument('invocation')
    p.add_argument('--arm', required=True, choices=('dense', 'pair_scoped'))
    p.add_argument('--profile', type=int, choices=(0, 1), required=True)
    p.add_argument('--enable-lifecycle', action='store_true')
    p.add_argument('--seconds', type=float, default=300)
    a = p.parse_args()
    require(0 < a.seconds <= 300, 'unchanged deadline')
    print(compact(check_package(a.root, expected=a.expected, invocation=a.invocation,
                  expected_method=method(a.arm, a.profile), work=Work(time.monotonic() + a.seconds),
                  enabled=a.enable_lifecycle)).decode())


if __name__ == '__main__':
    main()
