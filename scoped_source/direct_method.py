"""Closed control-only method registry. Metadata is not proof of an optimizer."""
from scoped_source.hz_request_store import identity, require

PROTOCOL = '6f53abbaf465f78fe586eb66c7e08ea0a8623dc1a2809e9793148898e0a2826f'
ARMS = ('dense', 'pair_scoped')
SHARED_PROTOCOL = '5adaa408bab129089a31935706b4ce518ad5f88b6f949af4dcddd14c74c294dc'


def method(arm, profile=0, *, shared=False):
    require(type(arm) is str and arm in ARMS and type(profile) is int and profile in (0, 1),
            'registered fixed-stub method')
    require(type(shared) is bool, 'explicit shared mode')
    consumer = dict(kind=arm, check_width=2 if arm == 'pair_scoped' else 8)
    return dict(schema='SHARED_LIFECYCLE_METHOD_V1' if shared else 'DIRECT_LIFECYCLE_METHOD_V1',
                protocol=SHARED_PROTOCOL if shared else PROTOCOL,
                producer=dict(kind='FIXED_SCALAR_STUB_V1', profile=profile),
                candidate_width=8, production=consumer, replay=dict(consumer))


def validate(value):
    require(type(value) is dict, 'method object')
    try:
        expected = method(value['production']['kind'], value['producer']['profile'],
                          shared=value.get('schema') == 'SHARED_LIFECYCLE_METHOD_V1')
    except (KeyError, TypeError):
        raise ValueError('method fields') from None
    require(identity(value) == identity(expected), 'frozen producer/production/replay method')
    return expected
