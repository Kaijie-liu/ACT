"""Stdlib-only binding of construction receipts, not cached mathematical facts."""
from scoped_proof.io import load, sha
from source_enclosure.format import identity

SCHEMA = 'SCOPED_PARSE_PROOF_V1'
MODES = ('uncached', 'cached')
LIMITS = {'entries': 64, 'bytes': 64 * 1024**2, 'cells': 2_000_000}


def policy(spec):
    value = spec['construction_policy']
    if (set(value) != {'schema', 'mode', 'cache_limits', 'checker_cache'} or
        value['schema'] != SCHEMA or value['mode'] not in MODES or
        value['cache_limits'] != LIMITS or value['checker_cache'] is not False):
        raise ValueError('explicit frozen construction policy required')
    return value


def validate_receipt(root):
    invocation = load(root/'invocation.json')
    spec = load(root/'spec.json', invocation['spec_file_sha256'])
    p = policy(spec); receipt = load(root/'construction_receipt.json')
    if (receipt['invocation'] != invocation['invocation'] or
        receipt['spec_sha256'] != invocation['spec_file_sha256'] or
        receipt['policy_sha256'] != identity(p)):
        raise ValueError('construction receipt wrong execution/policy')
    for name in ('source', 'construction'):
        file = root/(name+'.json'); record = receipt[name]
        if record != {'sha256': sha(file), 'bytes': file.stat().st_size}:
            raise ValueError('construction receipt wrong artifact')
    report = receipt['report']; parser = report['parser']
    if (report['mode'] != p['mode'] or parser['enabled'] is not (p['mode'] == 'cached') or
        parser['policy'] != 'SOURCE_CONSTRUCTION_CSR_PARSE_V1' or
        parser['scope'] != receipt['source']['sha256'] or parser['limits'] != LIMITS or
        any(parser[k] != 0 for k in ('live_entries', 'live_cells', 'live_bytes')) or
        any(report[k] is not False for k in ('algebra_changed', 'checks_elided',
            'source_validation_cached', 'bound_or_verdict_cached')) or
        report['not_a_proof_verdict'] is not True):
        raise ValueError('unregistered parsing/checking behavior')
    counts=('lookups','hits','parses','evictions','oversized','peak_entries','peak_bytes','peak_cells')
    if (any(type(parser[k]) is not int or parser[k]<0 for k in counts) or
        parser['lookups'] != parser['hits'] + parser['parses'] or
        any(parser['peak_'+k]>LIMITS[k] for k in ('entries','bytes','cells'))):
        raise ValueError('parser accounting')
    if p['mode'] == 'uncached' and (parser['hits'] or parser['peak_entries']):
        raise ValueError('disabled arm cached parses')
    return {'invocation': invocation['invocation'], 'spec_sha256': invocation['spec_file_sha256'],
        'construction_receipt_sha256': sha(root/'construction_receipt.json'),
        'mode': p['mode'], 'provenance_only': True}
