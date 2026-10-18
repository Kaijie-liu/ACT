"""Relocatable V2 binding + unchanged independent mathematical recheck.

The expected capsule AND bundle digests come from the external frozen execution
record, not from the moved directory itself. Offline success never upgrades an
online timeout. Only the declared source is proved, not checkpoint/CUDA identity.
The managed package / relocated private copy must have no concurrent external
writer during checking. The old reader reopens files after preflight: seals
detect changes, but do not turn this into an atomic parser-input binding against
an adversarial co-writer or prove an arbitrary source's complete heap bound.
"""
import hashlib
from pathlib import Path

from scoped_source.hz_request_store import require, sha
from scoped_source.native_resources import Resources, bounded_read, parse, measure
from scoped_source.native_resource_protocol import common_contract, profile, layout_preflight
from scoped_source.native_package import check_package, CODE_FILES

AUXILIARY_CODE = ('scoped_source/native_protocol.py',
    'scoped_source/native_resources.py','scoped_source/native_resource_protocol.py',
    'scoped_source/native_resource_portable.py')


def check_bound_package(root, *, expected_bundle, expected_capsule, expected_envelope,
                        expected_invocation, resources, work, enabled=False):
    require(enabled is True, 'bound portable V2 opt-in')
    require(type(resources) is Resources, 'reviewed resource owner, no arbitrary callback')
    require(work.deadline == resources.deadline or
            work.deadline == resources.deadline-1, 'one cumulative checking deadline')
    resources.active('phase')
    with resources.hold('metadata') as slot:
        raw, _ = bounded_read(Path(root)/'bindings.json',min(2**20,slot['encoded']),
                              work.deadline,sha(expected_capsule))
        b, _, _ = parse(raw,resources,'metadata')
        require(type(b) is dict and set(b) == {'schema','envelope','invocation','bindings',
            'invocation_hash','auxiliary','source_scope'} and b['schema'] == 'NATIVE_RESOURCE_BINDING_V2', 'portable binding schema')
        e = b['envelope']; i = b['invocation']; refs = b['bindings']
        er = measure(e,work.deadline,2**20,slot['depth'])
        ir = measure(i,work.deadline,2**20,slot['depth'])
        require(er['sha256'] == sha(expected_envelope) == refs['envelope']
                and ir['sha256'] == sha(expected_invocation) == b['invocation_hash'], 'external portable anchors')
        resources.verify_plan(e['resources'],expected_envelope)
        require(e['resources']['formula'] == e['common']['ownership_formula'], 'portable formula binding')
        common_contract(e['common']); profile(e['profile'])
        require(measure(e['common'],work.deadline,2**20,slot['depth'])['sha256'] == refs['common']
                and measure(e['layout'],work.deadline,2**20,slot['depth'])['sha256'] == refs['layout']
                and e['layout']['common'] == refs['common'], 'portable common/layout binding')
        sha(refs['registry'])
        require(i['schema'] == 'NATIVE_RESOURCE_INVOCATION_V2' and i['bindings'] == refs
                and i['profile'] == e['profile'] and i['envelope'] == refs['envelope']
                and i['root'] == e['root'], 'original execution binding, not relocated pathname')
        require([r['file'] for r in b['auxiliary']] == ['code/'+n for n in AUXILIARY_CODE], 'portable resource checker roster')
        for r,n in zip(b['auxiliary'],AUXILIARY_CODE,strict=True):
            require(r['sha256'] == e['profile']['implementation'][n], 'bound auxiliary code')
            raw_code, _ = bounded_read(Path(root)/r['file'],65536,work.deadline,r['sha256'])
            require(len(raw_code) == r['bytes'], 'auxiliary code bytes')
        layout_preflight(Path(root)/'source',e['layout'],e['profile']['source']['source'],resources,work)
        result = check_package(root,expected=expected_bundle,expected_method=e['profile']['method'],
            invocation=expected_invocation,expected_code={n:e['profile']['implementation'][n] for n in CODE_FILES},
            expected_source=e['profile']['source']['source'],work=work,enabled=True)
        return dict(result=result,bindings=refs,capsule=expected_capsule,
                    offline_only=True,online_upgrade=False,
                    guarantee='DECLARED_SOURCE_NOT_CHECKPOINT_OR_NATIVE_FLOAT_PROOF')
