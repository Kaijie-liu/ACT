"""One packaging phase. The isolated child is the only mathematical checker."""
import argparse
from pathlib import Path
import sys
import time

from portable_proof.runtime import strict_json, original_bytes, digest


def execute(stage, root):
    if stage != 'package': raise ValueError('no precheck stage in V3')
    job = strict_json((root/'job.json').read_bytes())
    def tick():
        if time.monotonic() >= job['started_monotonic']+298:
            raise TimeoutError('request work budget exhausted')
    tick(); source = Path(job['source']); raw = (source/'manifest.json').read_bytes()
    if digest(raw) != job['manifest_sha256']:
        raise ValueError('source manifest changed')
    from single_check_portable.pack import pack
    info = pack(source, job['request'], root/'portable', enabled=job['cache_enabled'], tick=tick)
    if info['source_manifest_sha256'] != job['manifest_sha256']:
        raise ValueError('packaged a different source')
    tick()
    with (root/'packing.json').open('xb') as f: f.write(original_bytes(info))
    tick()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('package',))
    p.add_argument('root', type=Path); a = p.parse_args()
    try: execute(a.stage, a.root)
    except TimeoutError: sys.exit(3)
