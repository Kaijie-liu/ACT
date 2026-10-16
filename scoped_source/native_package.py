"""Portable native-path evidence check. Standard library, no optimizer import."""
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY, require, sha, compact, decode, digest, identity
from scoped_source.factored_io import read
from scoped_source.wire_package import inventory
from scoped_source.direct_lifecycle_package import CODE_FILES as BASE_CODE, SHARED_CODE, PAGED_CODE
from scoped_source.native_method import validate
from scoped_source.native_result import Comparator
from scoped_source.direct_candidates import Driver, MAX_TILE, PREFIX
from scoped_source.scaled_endpoints import contract
from scoped_source.shared_format import load
from scoped_source.check_shared_source import CheckedSharedForest

MANIFEST = 'NATIVE_SHARED_MANIFEST_V1'
BUNDLE = 'NATIVE_SHARED_BUNDLE_V1'
CODE_FILES = tuple(sorted(set(BASE_CODE + SHARED_CODE + PAGED_CODE +
    ('scoped_source/native_method.py', 'scoped_source/native_result.py', 'scoped_source/native_package.py',
     'scoped_source/native_portable.py'))))


def manifest(root, expected, expected_method, invocation, work, expected_code, expected_source):
    mth = validate(expected_method); sha(invocation)
    refs, total = inventory(root, expected, work, schema=BUNDLE)
    raw = read(root, 'manifest.json', 2**20, refs['manifest.json']['sha256'])
    m = decode(raw)
    require(compact(m) == raw and type(m) is dict and set(m) ==
        {'schema','source','forest','contract','tiles','code','method','invocation','result'}
        and m['schema'] == MANIFEST and m['invocation'] == invocation
        and identity(m['method']) == identity(mth), 'native manifest identity/schema')
    require(type(m['code']) is list and [r['file'] for r in m['code']] == ['code/'+n for n in CODE_FILES],
            'complete independent checker code')
    require(type(expected_code) is dict and set(expected_code) == set(CODE_FILES),
            'externally bound checker implementation')
    for ref in m['code']:
        require(ref == refs.get(ref['file']), 'checker code inventory binding')
        require(ref['sha256'] == sha(expected_code[ref['file'].removeprefix('code/')]),
                'checker implementation external identity')
        member(root, ref, 64*2**10, work)
    require(type(m['source']) is dict and set(m['source']) == {'directory','sha256'}
            and m['source']['directory'] == 'source', 'local declared source binding')
    require(sha(m['source']['sha256']) == sha(expected_source), 'externally frozen request source')
    require(m['forest']['file'] == 'forest/index.json' and m['contract']['file'] == 'contract.json',
            'fixed source and property members')
    for ref in [m['forest'], m['contract'], m['result'], *m['tiles']]:
        require(ref == refs.get(ref['file']), 'native manifest member inventory')
    return m, total


def member(root, ref, cap, work):
    require(type(ref) is dict and set(ref) == {'file','bytes','sha256'}
            and type(ref['bytes']) is int and 0 < ref['bytes'] <= cap, 'native bounded member')
    work.tick(); raw = read(root, ref['file'], cap, sha(ref['sha256'])); work.tick()
    require(len(raw) == ref['bytes'], 'native member length')
    return raw


def check_package(root, *, expected, expected_method, invocation, expected_code, expected_source, work, enabled=False):
    require(enabled is True, 'native-path independent check opt-in')
    started = time.monotonic(); mth = validate(expected_method)
    m, total = manifest(root, expected, mth, invocation, work, expected_code, expected_source)
    packet = load(Path(root)/'forest', member(root,m['forest'],POLICY['member_bytes'],work), m['forest']['sha256'], work)
    f = CheckedSharedForest(packet, Path(root)/'source', m['source']['sha256'],
        expected=packet.identity, work=work, enabled=True)
    packet = None
    raw = member(root, m['contract'], 8192, work); c = decode(raw)
    require(compact(c) == raw and c == contract(f), 'independent complete output properties')
    d = Driver(f,c,expected=identity(c),work=work,pair_scoped=True)
    try:
        require(type(m['tiles']) is list and len(m['tiles']) == d.session.tile_count
                and [r['file'] for r in m['tiles']] == [f'tiles/{i:04d}.bin' for i in range(d.session.tile_count)],
                'native complete ordered unique tiles')
        for ref in m['tiles']:
            _, _, token = d.prepare(); raw = member(root,ref,MAX_TILE,work)
            require(len(raw) >= PREFIX.size and PREFIX.unpack_from(raw)[-2] == 0, 'native binary64 profile')
            d.receive(token,raw,lambda i,b:(digest(b),len(b)))
        sink = Comparator(root,m['result'],mth,invocation,m['forest']['sha256'],m['tiles'],work)
        result = d.finish(sink=sink)
        inventory(root,expected,work,schema=BUNDLE); work.tick()
        return dict(result=result,result_ref=m['result'],context=dict(sink.reader.context),
            package_sha256=expected,package_bytes=total,source=m['source']['sha256'],
            request={k:f.declaration[k] for k in ('experts','classes','label')},
            owner=dict(d.owner.stats),driver_cost=dict(d.costs),
            candidate_bytes=d.total_bytes,work=dict(work.counts),seconds=time.monotonic()-started)
    finally:
        if not d.closed: d.abort()
