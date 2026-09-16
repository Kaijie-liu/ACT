"""New portable identity; old bundles/builders and proof mathematics stay fixed."""
from pathlib import Path
import time

from portable_proof.runtime import strict_json, original_bytes, compact, digest
from moe_evidence.bundle import pack as old_pack

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'PORTABLE_WEIGHTED_TOP2_CACHE_V2'


def pack(source, destination, result, *, enabled=False, tick=lambda: None):
    if type(enabled) is not bool:
        raise ValueError('explicit boolean cache option required')
    started = time.monotonic(); tick()
    d = Path(destination).resolve()
    if not d.is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('write outside authorized workspace')
    # Build only in a fresh destination. V1 metadata is never published as V2.
    old = old_pack(source, d, result, tick=tick)
    meta = strict_json((d/'bundle.json').read_bytes())
    for src, target in (
        ('exact_matrix_cache/cache.py', 'code/exact_matrix_cache/cache.py'),
        ('exact_matrix_cache/checker.py', 'code/exact_matrix_cache/checker.py'),
        ('cached_portable/runtime.py', 'code/runtime.py'),
        ('cached_portable/launcher.py', 'verify.py')):
        tick(); raw = (ROOT/src).read_bytes(); path = d/target
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        meta['checker_sources'][src] = digest(raw)
    (d/'code/exact_matrix_cache/__init__.py').write_bytes(b'')
    meta['schema'] = SCHEMA
    meta['checker_sources']['cached_portable/pack.py'] = digest(Path(__file__).read_bytes())
    meta['parser'] = {'policy': 'EXACT_CSR_PARSE_V1', 'enabled': enabled,
                      'lifetime': 'ONE_CHECK', 'cached_verdicts': False}
    # Metadata pins the exact executable bytes, including stripped solver modules.
    meta['files'] = {}
    for p in d.rglob('*'):
        if p.is_file() and p != d/'bundle.json':
            tick(); meta['files'][str(p.relative_to(d))] = digest(p.read_bytes())
    tick(); (d/'bundle.json').write_bytes(original_bytes(meta)); tick()
    return {**old, 'schema': SCHEMA, 'cache_enabled': enabled,
            'bundle_sha256': digest((d/'bundle.json').read_bytes()),
            'statement_sha256': digest(compact(meta['statement'])),
            'pack_seconds': time.monotonic()-started,
            'bundle_bytes': sum(p.stat().st_size for p in d.rglob('*') if p.is_file())}
