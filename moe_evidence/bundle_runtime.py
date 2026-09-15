"""Portable general-request verifier; invokes only independently checked math."""
from pathlib import Path
from transport import Store,compact,digest,original_bytes,strict_json


def verify(directory,expected_bundle,expected_statement):
    root=Path(directory).resolve();raw=(root/'bundle.json').read_bytes()
    if digest(raw)!=expected_bundle:raise ValueError('bundle hash mismatch')
    meta=strict_json(raw)
    if meta['schema']!='PORTABLE_WEIGHTED_TOP2_V1':raise ValueError('bundle schema')
    if digest(compact(meta['statement']))!=expected_statement:raise ValueError('statement mismatch')
    for name,sha in meta['files'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or digest(p.read_bytes())!=sha:
            raise ValueError('bundle file/path mismatch')
    store=Store(root/'evidence.zip')
    def load(ref):
        if Path(ref['file']).name!=ref['file']:raise ValueError('nonlocal evidence')
        item=meta['logical_files'][ref['file']]
        obj=store.decode(store.get(item['root']))
        if digest(original_bytes(obj))!=ref['sha256'] or item['original_sha256']!=ref['sha256']:
            raise ValueError('source identity mismatch')
        return obj
    m=load(meta['manifest'])
    if {k:m[k] for k in ('request','routes','common_facts','contexts')}!=meta['statement']:
        raise ValueError('statement source/route binding mismatch')
    from moe_evidence.checker import check_manifest
    result=check_manifest(m,meta['statement']['request'],load)
    if result!=meta['expected_result']:raise ValueError('candidate result differs from independent result')
    return result
