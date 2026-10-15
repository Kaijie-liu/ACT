"""Closed control-only method registry. Metadata is not proof of an optimizer."""
from scoped_source.hz_request_store import identity, require

PROTOCOL = '6f53abbaf465f78fe586eb66c7e08ea0a8623dc1a2809e9793148898e0a2826f'
ARMS = ('dense', 'pair_scoped')


def method(arm, profile=0):
    require(type(arm) is str and arm in ARMS and type(profile) is int and profile in (0, 1),
            'registered fixed-stub method')
    consumer = dict(kind=arm, check_width=2 if arm == 'pair_scoped' else 8)
    return dict(schema='DIRECT_LIFECYCLE_METHOD_V1', protocol=PROTOCOL,
                producer=dict(kind='FIXED_SCALAR_STUB_V1', profile=profile),
                candidate_width=8, production=consumer, replay=dict(consumer))


def validate(value):
    require(type(value) is dict, 'method object')
    try:
        expected = method(value['production']['kind'], value['producer']['profile'])
    except (KeyError, TypeError):
        raise ValueError('method fields') from None
    require(identity(value) == identity(expected), 'frozen producer/production/replay method')
    return expected
