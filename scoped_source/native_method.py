"""Closed, default-off native method identity; not an execution authorization."""
from scoped_source.hz_request_store import identity, require

ARMS = ('native_coo_reference', 'native_source_row_blocks')
ALGORITHM = 'CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3'
SCOPE = 'HZ_NATIVE_SOURCE_INTEGRATION_20261017_V1'


def method(arm):
    require(type(arm) is str and arm in ARMS, 'registered native arm')
    return dict(schema='NATIVE_SHARED_METHOD_V1', scope=SCOPE,
        producer=dict(kind=ALGORITHM, representation=arm, device='cpu', dtype='float64',
                      threads=1, profile=0, updates=128, evaluations=129),
        candidate_width=8, production=dict(kind='pair_scoped', check_width=2),
        replay=dict(kind='pair_scoped', check_width=2),
        results=dict(schema='SHARED_RESULT_STREAM_V1', page_rows=8, page_bytes=65536))


def validate(value):
    require(type(value) is dict, 'native method object')
    try:
        expected = method(value['producer']['representation'])
    except (KeyError, TypeError):
        raise ValueError('native method fields') from None
    require(identity(value) == identity(expected), 'complete native method binding')
    return expected
