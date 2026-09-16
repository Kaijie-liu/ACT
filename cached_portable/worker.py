"""Isolated precheck/package phases, sharing the caller's original request clock."""
import argparse
from pathlib import Path
import sys
import time

from portable_proof.runtime import strict_json, original_bytes, digest


def execute(stage, root):
    job = strict_json((root/'job.json').read_bytes())
    def tick():
        if time.monotonic() >= job['started_monotonic']+298:
            raise TimeoutError('request work budget exhausted')
    tick(); source = Path(job['source'])
    raw = (source/'manifest.json').read_bytes()
    if digest(raw) != job['manifest_sha256']:
        raise ValueError('source manifest changed')
    m = strict_json(raw)
    if m['request'] != job['request']:
        raise ValueError('request/source binding mismatch')
    if stage == 'precheck':
        from exact_matrix_cache.checker import check_manifest
        from moe_evidence.storage import loader
        result = check_manifest(m, job['request'], loader(source, tick),
                                enabled=job['cache_enabled'], tick=tick)
        name = 'precheck.json'
    elif stage == 'package':
        from cached_portable.pack import pack
        result = pack(source, root/'portable', strict_json((root/'precheck.json').read_bytes())['result'],
                      enabled=job['cache_enabled'], tick=tick)
        name = 'packing.json'
    else:
        raise ValueError('unknown evidence-tail stage')
    tick()
    with (root/name).open('xb') as f: f.write(original_bytes(result))
    tick()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('precheck', 'package'))
    p.add_argument('root', type=Path); a = p.parse_args()
    try: execute(a.stage, a.root)
    except TimeoutError: sys.exit(3)
