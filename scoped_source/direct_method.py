"""Closed control-only method registry. Metadata is not proof of an optimizer."""
from scoped_source.hz_request_store import identity, require

PROTOCOL = '6f53abbaf465f78fe586eb66c7e08ea0a8623dc1a2809e9793148898e0a2826f'
ARMS = ('dense', 'pair_scoped')
SHARED_PROTOCOL = '5adaa408bab129089a31935706b4ce518ad5f88b6f949af4dcddd14c74c294dc'
PAGED_PROTOCOL = '80d7de69285e7f8044bc777ac4c0d495a8cabb97b0682bbfca35eb80e2b2d535'


def paged(value): return value['schema'] == 'PAGED_LIFECYCLE_METHOD_V1'
def is_shared(value): return value['schema'] in ('SHARED_LIFECYCLE_METHOD_V1', 'PAGED_LIFECYCLE_METHOD_V1')


def method(arm, profile=0, *, shared=False, paged_results=False):
    require(type(arm) is str and arm in ARMS and type(profile) is int and profile in (0, 1),
            'registered fixed-stub method')
    require(type(shared) is bool, 'explicit shared mode')
    require(type(paged_results) is bool and (not paged_results or shared), 'paged results require shared source')
    consumer = dict(kind=arm, check_width=2 if arm == 'pair_scoped' else 8)
    value = dict(schema='PAGED_LIFECYCLE_METHOD_V1' if paged_results else
                 'SHARED_LIFECYCLE_METHOD_V1' if shared else 'DIRECT_LIFECYCLE_METHOD_V1',
                protocol=PAGED_PROTOCOL if paged_results else SHARED_PROTOCOL if shared else PROTOCOL,
                producer=dict(kind='FIXED_SCALAR_STUB_V1', profile=profile),
                candidate_width=8, production=consumer, replay=dict(consumer))
    if paged_results:
        value['results'] = dict(schema='SHARED_RESULT_STREAM_V1', page_rows=8, page_bytes=65536)
    return value


def validate(value):
    require(type(value) is dict, 'method object')
    try:
        expected = method(value['production']['kind'], value['producer']['profile'],
                          shared=value.get('schema') in ('SHARED_LIFECYCLE_METHOD_V1', 'PAGED_LIFECYCLE_METHOD_V1'),
                          paged_results=value.get('schema') == 'PAGED_LIFECYCLE_METHOD_V1')
    except (KeyError, TypeError):
        raise ValueError('method fields') from None
    require(identity(value) == identity(expected), 'frozen producer/production/replay method')
    return expected
