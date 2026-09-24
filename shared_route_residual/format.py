"""Serialization/binding only. No source or mathematical acceptance logic."""
from source_enclosure.format import identity

SCHEMA='SHARED_ROUTER_RESIDUAL_CERTIFICATE_V1'


def binding(doc, prefix, invocation):
    if type(invocation) is not str or not invocation or len(invocation)>128:
        raise ValueError('explicit request invocation required')
    state=prefix['router']['steps'][-1]['state']
    return {'source_sha256':identity(doc),'request_sha256':identity(doc['request']),
        'prefix_sha256':identity(prefix),'router_sha256':identity(state),
        'factor_order_sha256':identity({'continuous':state['continuous_ids'],'binary':state['binary_ids']}),
        'invocation':invocation,'factor_domain':'CONTINUOUS_BOX_MINUS1_PLUS1_WITH_BINARY_RELAXATION'}
