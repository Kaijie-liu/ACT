"""Closed, default-off native method identity; not an execution authorization."""
from scoped_source.hz_request_store import identity, require

ARMS = ('native_coo_reference', 'native_source_row_blocks')
ALGORITHM = 'CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3'
SCOPE = 'HZ_NATIVE_SOURCE_INTEGRATION_20261017_V1'
MIDPOINT_ALGORITHM = 'CPU_SCALED_REFERENCE_ENDPOINT_MIDPOINT_LAST_V1'
MIDPOINT_SCOPE = 'HZ_ENDPOINT_DUAL_EXCHANGE_20261019_V1'
MIDPOINT_SCHEMA = 'NATIVE_SHARED_ENDPOINT_MIDPOINT_METHOD_V1'


def method(arm, *, endpoint_midpoint=False):
    require(type(arm) is str and arm in ARMS, 'registered native arm')
    require(type(endpoint_midpoint) is bool, 'explicit endpoint midpoint option')
    result = dict(schema='NATIVE_SHARED_METHOD_V1', scope=SCOPE,
        producer=dict(kind=ALGORITHM, representation=arm, device='cpu', dtype='float64',
                      threads=1, profile=0, updates=128, evaluations=129),
        candidate_width=8, production=dict(kind='pair_scoped', check_width=2),
        replay=dict(kind='pair_scoped', check_width=2),
        results=dict(schema='SHARED_RESULT_STREAM_V1', page_rows=8, page_bytes=65536))
    if endpoint_midpoint:
        result.update(schema=MIDPOINT_SCHEMA, scope=MIDPOINT_SCOPE)
        result['producer'].update(kind=MIDPOINT_ALGORITHM,
            endpoint_exchange=dict(kind='same_property_best_midpoint', after_evaluation=127,
                gradient_updates_two_endpoints=127, midpoint_updates_two_endpoints=1,
                gradient_updates_single_endpoint=128, preserve_forward_calls=128))
    return result


def validate(value):
    require(type(value) is dict, 'native method object')
    require(value.get('schema') in ('NATIVE_SHARED_METHOD_V1', MIDPOINT_SCHEMA),
            'closed native method version')
    try:
        expected = method(value['producer']['representation'],
                          endpoint_midpoint=value['schema'] == MIDPOINT_SCHEMA)
    except (KeyError, TypeError):
        raise ValueError('native method fields') from None
    require(identity(value) == identity(expected), 'complete native method binding')
    return expected
