"""Pack unverified evidence. No result argument and no mathematical precheck."""
from pathlib import Path
import time

from portable_proof.runtime import strict_json, original_bytes, compact, digest
from cached_portable.pack import pack as transport_pack

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'PORTABLE_WEIGHTED_TOP2_SINGLE_CHECK_V3'
DECISION = 'PACK_ONLY_THEN_ISOLATED_FULL_CHECK'


def pack(source, request, destination, *, enabled=False, tick=lambda: None):
    started = time.monotonic(); tick()
    source = Path(source); d = Path(destination)
    raw = (source/'manifest.json').read_bytes(); manifest = strict_json(raw)
    if manifest['request'] != request:
        raise ValueError('requested theorem/source mismatch')
    source_sha = digest(raw)
    # The old builder only serializes/hashes; None is discarded before V3 publication.
    # It does not run a checker even when the supplied evidence is invalid.
    info = transport_pack(source, d, None, enabled=enabled, tick=tick)
    meta = strict_json((d/'bundle.json').read_bytes())
    if meta['manifest']['sha256'] != source_sha or meta['statement']['request'] != request:
        raise ValueError('manifest changed while packing')
    del meta['expected_result']
    meta.update(schema=SCHEMA, decision=DECISION, source_manifest_sha256=source_sha)
    for src, target in (('single_check_portable/runtime.py', 'code/runtime.py'),
                        ('single_check_portable/launcher.py', 'verify.py')):
        tick(); content = (ROOT/src).read_bytes(); (d/target).write_bytes(content)
        meta['checker_sources'][src] = digest(content)
    meta['checker_sources']['single_check_portable/pack.py'] = digest(Path(__file__).read_bytes())
    meta['files'] = {}
    for p in d.rglob('*'):
        if p.is_file() and p != d/'bundle.json':
            tick(); meta['files'][str(p.relative_to(d))] = digest(p.read_bytes())
    tick(); (d/'bundle.json').write_bytes(original_bytes(meta)); tick()
    return {**info, 'schema': SCHEMA, 'decision': DECISION, 'source_manifest_sha256': source_sha,
            'bundle_sha256': digest((d/'bundle.json').read_bytes()),
            'statement_sha256': digest(compact(meta['statement'])),
            'pack_seconds': time.monotonic()-started,
            'bundle_bytes': sum(p.stat().st_size for p in d.rglob('*') if p.is_file()),
            'mathematical_precheck_executed': False, 'checked_result': None}
