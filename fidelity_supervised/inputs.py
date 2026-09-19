"""Hash-bound input decoding inside the caller's clock; no historical answers."""
from pathlib import Path
from portable_proof.runtime import digest
from lp_sandwich.check import strict_json, identity, rational, validate_statement
from sparse_basis.engine import dimensions


def load(spec, tick):
    tick()
    kinds = [key for key in ('input', 'export') if key in spec]
    if len(kinds) != 1:
        raise ValueError('exactly one input representation required')
    kind = kinds[0]
    ref = spec[kind]
    raw = Path(ref['path']).read_bytes()
    tick()
    if digest(raw) != ref['sha256']:
        raise ValueError('supplied input bytes changed')
    v = strict_json(raw)
    s = spec['statement']
    if kind == 'input':
        if set(v) != {'lp', 'statement'} or v['statement'] != s:
            raise ValueError('LP/statement binding')
    else:
        if (ref['sha256'] != s['export_sha256'] or
                identity(v['source']) != v['source_sha256'] or v['source_sha256'] != s['source_sha256'] or
                list(map(rational, v['q'])) != list(map(rational, s['property']['q'])) or
                rational(v['offset']) != rational(s['property']['constant'])):
            raise ValueError('export/source/property binding')
        v = {'lp': v['lp'], 'statement': s}
    validate_statement(s, v['lp'], spec['statement_sha256'], tick)
    dimensions(v['lp'])
    tick()
    return v
