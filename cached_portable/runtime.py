"""Stdlib portable V2: transport and math rechecked, only exact parsing reused."""
from pathlib import Path
from transport import Store, compact, digest, original_bytes, strict_json


def verify(directory, expected_bundle, expected_statement, *, tick):
    tick(); root = Path(directory).resolve(); raw = (root/'bundle.json').read_bytes()
    if digest(raw) != expected_bundle:
        raise ValueError('bundle hash mismatch')
    meta = strict_json(raw)
    if meta['schema'] != 'PORTABLE_WEIGHTED_TOP2_CACHE_V2':
        raise ValueError('not a V2 cache bundle')
    parser = meta['parser']
    if (type(parser['enabled']) is not bool or parser != {
            'policy': 'EXACT_CSR_PARSE_V1', 'enabled': parser['enabled'],
            'lifetime': 'ONE_CHECK', 'cached_verdicts': False}):
        raise ValueError('unsupported cache contract')
    if digest(compact(meta['statement'])) != expected_statement:
        raise ValueError('statement mismatch')
    for name, sha in meta['files'].items():
        tick(); p = root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or digest(p.read_bytes()) != sha:
            raise ValueError('bundle file/path mismatch')
    store = Store(root/'evidence.zip')
    try:
        def load(ref):
            tick()
            if Path(ref['file']).name != ref['file']:
                raise ValueError('nonlocal evidence')
            item = meta['logical_files'][ref['file']]
            obj = store.decode(store.get(item['root']))
            if digest(original_bytes(obj)) != ref['sha256'] or item['original_sha256'] != ref['sha256']:
                raise ValueError('source identity mismatch')
            tick(); return obj
        m = load(meta['manifest'])
        if {k: m[k] for k in ('request', 'routes', 'common_facts', 'contexts')} != meta['statement']:
            raise ValueError('statement source/route mismatch')
        from exact_matrix_cache.checker import check_manifest
        checked = check_manifest(m, meta['statement']['request'], load,
                                 enabled=parser['enabled'], tick=tick)
        if checked['result'] != meta['expected_result']:
            raise ValueError('precheck differs from isolated result')
        tick(); return checked
    finally:
        store.zip.close(); store.cache.clear()
